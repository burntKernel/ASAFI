"""
ASAFI - Autonomous Spacecraft Avionics Fire Interceptor
Vision Pipeline: /dev/video0 -> YOLO Inference -> Visual Overlays & Centroid Tracking
Optimized for Raspberry Pi 4B CPU execution.
"""

import os
import cv2
import time
import base64
import logging
import threading
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASAFI.Vision")


class VisionPipeline:
    def __init__(self, model_path="models/best.pt", camera_index=0, imgsz=320, conf_threshold=0.35):
        self.model_path = model_path
        self.camera_index = camera_index
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

        self.cap = None
        self.model = None
        self.is_running = False
        self.worker_thread = None
        self.lock = threading.Lock()

        # Latest frame buffers & telemetry
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_jpeg_bytes = None
        self.latest_base64_frame = ""
        self.fps = 0.0
        self.inference_ms = 0.0
        self.detections = []
        self.target_detected = False
        self.primary_target = None  # {cx, cy, bbox, conf, label}

        self._init_model()
        self._init_camera()

    def _init_model(self):
        """Attempts to load Ultralytics YOLO model with fallback to color-based detection if PyTorch is absent."""
        try:
            from ultralytics import YOLO
            abs_model_path = os.path.abspath(self.model_path)
            if not os.path.exists(abs_model_path):
                # Try relative to repo root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(script_dir, "models", "best.pt")
                if os.path.exists(candidate):
                    abs_model_path = candidate

            if os.path.exists(abs_model_path):
                logger.info(f"[VISION] Loading YOLO model from {abs_model_path}...")
                self.model = YOLO(abs_model_path)
                logger.info(f"[VISION] YOLO model loaded successfully! Classes: {self.model.names}")
            else:
                logger.warning(f"[VISION] Model file not found at {abs_model_path}. Fallback detector active.")
                self.model = None
        except Exception as e:
            logger.warning(f"[VISION] Ultralytics/Torch unavailable or failed to load: {e}. Fallback detector active.")
            self.model = None

    def _init_camera(self):
        """Initializes V4L2 webcam at /dev/video0."""
        logger.info(f"[VISION] Opening camera index {self.camera_index} (/dev/video{self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Fallback without CAP_V4L2
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(f"[VISION] Failed to open /dev/video{self.camera_index}! Check camera connection.")
            return

        # Optimization for Raspberry Pi: lowest latency buffer and 640x480 resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, test_frame = self.cap.read()
        if ret and test_frame is not None:
            h, w = test_frame.shape[:2]
            logger.info(f"[VISION] Camera online! Native resolution: {w}x{h}")
        else:
            logger.warning("[VISION] Camera opened but initial frame read returned None.")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_and_infer_loop, daemon=True)
        self.worker_thread.start()
        logger.info("[VISION] Vision pipeline worker thread started.")

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info("[VISION] Vision pipeline stopped.")

    def _detect_fallback(self, frame):
        """
        Lightweight fallback detector (detects flame-like color contours)
        used if Ultralytics/PyTorch is loading or not installed.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Flame color range in HSV: yellow/orange/red high-intensity
        lower_flame = np.array([0, 80, 200], dtype=np.uint8)
        upper_flame = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_flame, upper_flame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 450:  # Noise threshold
                x, y, w, h = cv2.boundingRect(c)
                conf = min(0.95, 0.50 + (area / 10000.0) * 0.45)
                results.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": round(float(conf), 2),
                    "class_name": "fire",
                    "cx": int(x + w / 2),
                    "cy": int(y + h / 2),
                    "width": int(w),
                    "height": int(h)
                })
        return results

    def _infer_yolo(self, frame):
        """Runs YOLO model inference on frame."""
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device="cpu",
            verbose=False
        )
        self.inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                w = int(x2 - x1)
                h = int(y2 - y1)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 2),
                    "class_name": cls_name,
                    "cx": cx,
                    "cy": cy,
                    "width": w,
                    "height": h
                })
        return detections

    def _annotate_frame(self, frame, detections, fps):
        """Draws clean, professional tactical overlays on the live frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        center_x, center_y = w // 2, h // 2

        # 1. Optical Boresight Crosshair at optical center
        cross_len = 16
        cv2.line(annotated, (center_x - cross_len, center_y), (center_x + cross_len, center_y), (0, 240, 255), 1)
        cv2.line(annotated, (center_x, center_y - cross_len), (center_x, center_y + cross_len), (0, 240, 255), 1)
        cv2.circle(annotated, (center_x, center_y), 8, (0, 240, 255), 1)

        # 2. Draw Target Bounding Boxes & Centroids
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["cx"], det["cy"]
            conf = det["confidence"]
            label = det["class_name"].upper()

            is_fire = "fire" in label.lower()
            box_color = (40, 40, 255) if is_fire else (0, 165, 255)  # Red for fire, orange for smoke

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Corner accents
            c_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(annotated, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + c_len), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1 + c_len, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - c_len), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)

            # Vector line from boresight center to target centroid
            cv2.line(annotated, (center_x, center_y), (cx, cy), box_color, 1, cv2.LINE_AA)

            # Centroid crosshair
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

            # Label banner
            label_text = f"{label} {int(conf * 100)}%"
            (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + txt_w + 10, max(22, y1)), box_color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 5, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 3. Tactical HUD Banner (top)
        cv2.putText(
            annotated,
            f"ASAFI OPTICAL | {fps:.1f} FPS",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 128),
            2,
            cv2.LINE_AA
        )

        detector_type = "YOLOv8" if self.model else "COLOR-FALLBACK"
        status_txt = f"{detector_type} | TARGETS: {len(detections)}"
        cv2.putText(
            annotated,
            status_txt,
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA
        )

        return annotated

    def _capture_and_infer_loop(self):
        """Worker loop running continuous acquisition and inference."""
        prev_time = time.perf_counter()

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # Inference
            if self.model:
                try:
                    detections = self._infer_yolo(frame)
                except Exception as e:
                    logger.error(f"[VISION] YOLO inference error: {e}")
                    detections = self._detect_fallback(frame)
            else:
                detections = self._detect_fallback(frame)

            # Sort detections by confidence, pick primary target
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            primary = detections[0] if len(detections) > 0 else None

            # Calculate FPS
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            calc_fps = (1.0 / dt) if dt > 0 else 0.0

            # Render overlay annotations
            annotated = self._annotate_frame(frame, detections, calc_fps)

            # Encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buf = cv2.imencode(".jpg", annotated, encode_param)

            if ret:
                jpeg_bytes = buf.tobytes()
                b64_str = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")

                with self.lock:
                    self.latest_raw_frame = frame
                    self.latest_annotated_frame = annotated
                    self.latest_jpeg_bytes = jpeg_bytes
                    self.latest_base64_frame = b64_str
                    self.fps = calc_fps
                    self.detections = detections
                    self.target_detected = len(detections) > 0
                    self.primary_target = primary

    def get_latest_data(self):
        """Thread-safe snapshot of the latest vision results."""
        with self.lock:
            return {
                "base64_frame": self.latest_base64_frame,
                "fps": round(self.fps, 1),
                "inference_ms": round(self.inference_ms, 1),
                "target_detected": self.target_detected,
                "detections": list(self.detections),
                "primary_target": dict(self.primary_target) if self.primary_target else None
            }

    def generate_mjpeg_stream(self):
        """Generator for HTTP MJPEG streaming."""
        while self.is_running:
            with self.lock:
                jpeg = self.latest_jpeg_bytes
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.033)  # ~30 FPS
