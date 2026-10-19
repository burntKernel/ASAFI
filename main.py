"""
ASAFI - Autonomous Spacecraft Avionics Fire Interceptor
Complete Single-File Autonomous Mission Server & Closed-Loop Control System

Contains:
1. Hardware Abstraction Layer (PCA9685 Gimbal + MLX90614 IR Sensor with MOCK fallbacks)
2. Vision Pipeline (OpenCV Capture + YOLO Inference Engine + HUD Overlay)
3. Autonomous State Machine (Scanning -> Target Tracking -> Thermal Verification -> Lock)
4. FastAPI Mission Server & WebSocket / MJPEG Streaming Backend
"""

import os
import sys
import json
import math
import time
import cv2
import base64
import asyncio
import logging
import argparse
import threading
import numpy as np
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASAFI.UnifiedSingleFile")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(REPO_DIR, "dashboard")

# Check SMBus2 availability for hardware I2C
try:
    import smbus2
    HAS_SMBUS2 = True
except ImportError:
    HAS_SMBUS2 = False
    logger.warning("[HARDWARE] smbus2 module not installed. Hardware I2C disabled.")


# =====================================================================
# SECTION 1: HARDWARE DRIVERS (PCA9685 & MLX90614 WITH MOCK FALLBACKS)
# =====================================================================

class PCA9685Driver:
    """PCA9685 PWM Driver using smbus2 directly."""
    PCA9685_MODE1 = 0x00
    PCA9685_PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus_num=1, address=0x40):
        self.bus_num = bus_num
        self.address = address
        self.bus = smbus2.SMBus(bus_num)
        self.init_device()

    def init_device(self):
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, 0x00)
        time.sleep(0.005)
        old_mode = self.bus.read_byte_data(self.address, self.PCA9685_MODE1)
        sleep_mode = (old_mode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, sleep_mode)
        self.bus.write_byte_data(self.address, self.PCA9685_PRESCALE, 121)  # 50Hz for servos
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode | 0xA1)

    def set_pwm(self, channel, on_tick, off_tick):
        reg = self.LED0_ON_L + 4 * channel
        data = [on_tick & 0xFF, (on_tick >> 8) & 0xFF, off_tick & 0xFF, (off_tick >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, reg, data)

    def set_servo_angle(self, channel, angle_deg, min_us=500, max_us=2500, angle_range=180.0):
        clamped_angle = max(0.0, min(angle_range, float(angle_deg)))
        pulse_us = min_us + (clamped_angle / angle_range) * (max_us - min_us)
        off_tick = int(round(pulse_us * 4096.0 / 20000.0))
        self.set_pwm(channel, 0, off_tick)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


class GimbalController:
    def __init__(self, bus_num=1, address=0x40, force_mock=False):
        self.bus_num = bus_num
        self.address = address
        self.is_mock = force_mock
        self.driver = None

        self.pan = 0.0      # -90 (left) to +90 (right)
        self.tilt = 0.0     # -45 (down) to +45 (up)
        self.kp = 0.04      # Proportional tracking gain

        self._init_hardware()

    def _init_hardware(self):
        if self.is_mock:
            logger.info("[GIMBAL] MOCK mode enabled.")
            return

        if not HAS_SMBUS2 or not os.path.exists(f"/dev/i2c-{self.bus_num}"):
            logger.warning(f"[GIMBAL] /dev/i2c-{self.bus_num} missing. Falling back to MOCK mode.")
            self.is_mock = True
            return

        try:
            self.driver = PCA9685Driver(bus_num=self.bus_num, address=self.address)
            self._write_servos(0.0, 0.0)
            logger.info(f"[GIMBAL] PCA9685 initialized on /dev/i2c-{self.bus_num} @ 0x{self.address:02X}")
        except Exception as e:
            logger.warning(f"[GIMBAL] PCA9685 unavailable ({e}). Falling back to MOCK mode.")
            self.is_mock = True
            self.driver = None

    def _write_servos(self, pan_deg, tilt_deg):
        if self.is_mock or not self.driver:
            return
        try:
            self.driver.set_servo_angle(0, pan_deg + 90.0)
            self.driver.set_servo_angle(1, tilt_deg + 90.0)
        except Exception as e:
            logger.error(f"[GIMBAL] I2C write error: {e}. Switching to MOCK.")
            self.is_mock = True

    def update_tracking(self, cx, cy, frame_w=640, frame_h=480):
        dx = cx - (frame_w / 2.0)
        dy = cy - (frame_h / 2.0)

        self.pan += self.kp * dx
        self.tilt -= self.kp * dy

        self.pan = max(-90.0, min(90.0, self.pan))
        self.tilt = max(-45.0, min(45.0, self.tilt))

        self._write_servos(self.pan, self.tilt)

        return {
            "pan": round(self.pan, 2),
            "tilt": round(self.tilt, 2),
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "hardware": "PHYSICAL" if not self.is_mock else "MOCK"
        }

    def center(self):
        self.pan = 0.0
        self.tilt = 0.0
        self._write_servos(0.0, 0.0)

    def get_status(self):
        return {
            "pan": round(self.pan, 2),
            "tilt": round(self.tilt, 2),
            "hardware": "PHYSICAL" if not self.is_mock else "MOCK"
        }


class ThermalSensor:
    MLX90614_TA = 0x06
    MLX90614_TOBJ1 = 0x07

    def __init__(self, bus_num=3, address=0x5A, force_mock=False):
        self.bus_num = bus_num
        self.address = address
        self.is_mock = force_mock
        self.bus = None

        self.ambient_temp = 24.2
        self.object_temp = 24.8
        self.threshold_c = 55.0

        self._init_hardware()

    def _init_hardware(self):
        if self.is_mock or not HAS_SMBUS2:
            self.is_mock = True
            return

        candidate_buses = [self.bus_num, 1] if self.bus_num != 1 else [1]
        connected = False

        for b in candidate_buses:
            if os.path.exists(f"/dev/i2c-{b}"):
                try:
                    bus = smbus2.SMBus(b)
                    raw = bus.read_word_data(self.address, self.MLX90614_TA)
                    temp = (raw * 0.02) - 273.15
                    if -40.0 <= temp <= 125.0:
                        self.bus = bus
                        self.bus_num = b
                        connected = True
                        logger.info(f"[THERMAL] MLX90614 connected on /dev/i2c-{b} @ 0x{self.address:02X}")
                        break
                except Exception:
                    pass

        if not connected:
            logger.warning(f"[THERMAL] MLX90614 not detected. Falling back to MOCK mode.")
            self.is_mock = True

    def read(self, fire_detected=False):
        if not self.is_mock and self.bus:
            try:
                raw_ta = self.bus.read_word_data(self.address, self.MLX90614_TA)
                raw_to = self.bus.read_word_data(self.address, self.MLX90614_TOBJ1)
                self.ambient_temp = round((raw_ta * 0.02) - 273.15, 1)
                self.object_temp = round((raw_to * 0.02) - 273.15, 1)
                return {
                    "ambient_c": self.ambient_temp,
                    "object_c": self.object_temp,
                    "threshold_c": self.threshold_c,
                    "fire_confirmed": self.object_temp >= self.threshold_c,
                    "hardware": "PHYSICAL"
                }
            except Exception as e:
                logger.error(f"[THERMAL] Read error: {e}. Switching to MOCK.")
                self.is_mock = True

        if fire_detected:
            self.object_temp += (68.5 - self.object_temp) * 0.3
        else:
            self.object_temp += (24.8 - self.object_temp) * 0.1

        return {
            "ambient_c": round(self.ambient_temp, 1),
            "object_c": round(self.object_temp, 1),
            "threshold_c": self.threshold_c,
            "fire_confirmed": self.object_temp >= self.threshold_c,
            "hardware": "MOCK"
        }


class HardwareManager:
    def __init__(self, force_mock=False):
        self.gimbal = GimbalController(bus_num=1, address=0x40, force_mock=force_mock)
        self.thermal = ThermalSensor(bus_num=3, address=0x5A, force_mock=force_mock)

    def update(self, fire_detected, cx=None, cy=None, frame_w=640, frame_h=480):
        if fire_detected and cx is not None and cy is not None:
            gimbal_data = self.gimbal.update_tracking(cx, cy, frame_w, frame_h)
        else:
            gimbal_data = self.gimbal.get_status()

        thermal_data = self.thermal.read(fire_detected=fire_detected)

        return {
            "gimbal": gimbal_data,
            "thermal": thermal_data,
            "hardware_mode": {
                "gimbal": "PHYSICAL" if not self.gimbal.is_mock else "MOCK",
                "thermal": "PHYSICAL" if not self.thermal.is_mock else "MOCK"
            }
        }


# =====================================================================
# SECTION 2: VISION PIPELINE (CAMERA + YOLO + HUD OVERLAY)
# =====================================================================

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

        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_jpeg_bytes = None
        self.latest_base64_frame = ""
        self.fps = 0.0
        self.inference_ms = 0.0
        self.detections = []
        self.target_detected = False
        self.primary_target = None

        self._init_model()
        self._init_camera()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            abs_model_path = os.path.abspath(self.model_path)
            if os.path.exists(abs_model_path):
                logger.info(f"[VISION] Loading YOLO model from {abs_model_path}...")
                self.model = YOLO(abs_model_path)
                logger.info(f"[VISION] YOLO model loaded! Classes: {self.model.names}")
            else:
                logger.warning(f"[VISION] Model missing at {abs_model_path}. Color detector active.")
                self.model = None
        except Exception as e:
            logger.warning(f"[VISION] Ultralytics/Torch unavailable: {e}. Color detector active.")
            self.model = None

    def _init_camera(self):
        logger.info(f"[VISION] Opening camera /dev/video{self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(f"[VISION] Failed to open camera /dev/video{self.camera_index}!")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_and_infer_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _detect_fallback(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_flame = np.array([0, 80, 200], dtype=np.uint8)
        upper_flame = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_flame, upper_flame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 450:
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

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 2),
                    "class_name": cls_name,
                    "cx": cx,
                    "cy": cy,
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })
        return detections

    def _annotate_frame(self, frame, detections, fps):
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Optical Crosshair
        cv2.line(annotated, (center_x - 16, center_y), (center_x + 16, center_y), (0, 240, 255), 1)
        cv2.line(annotated, (center_x, center_y - 16), (center_x, center_y + 16), (0, 240, 255), 1)
        cv2.circle(annotated, (center_x, center_y), 8, (0, 240, 255), 1)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["cx"], det["cy"]
            conf = det["confidence"]
            label = det["class_name"].upper()

            box_color = (40, 40, 255) if "fire" in label.lower() else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.line(annotated, (center_x, center_y), (cx, cy), box_color, 1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

            label_text = f"{label} {int(conf * 100)}%"
            cv2.putText(annotated, label_text, (x1 + 5, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(annotated, f"ASAFI | {fps:.1f} FPS", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
        return annotated

    def _capture_and_infer_loop(self):
        prev_time = time.perf_counter()

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            if self.model:
                try:
                    detections = self._infer_yolo(frame)
                except Exception:
                    detections = self._detect_fallback(frame)
            else:
                detections = self._detect_fallback(frame)

            detections.sort(key=lambda d: d["confidence"], reverse=True)
            primary = detections[0] if len(detections) > 0 else None

            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            calc_fps = (1.0 / dt) if dt > 0 else 0.0

            annotated = self._annotate_frame(frame, detections, calc_fps)
            ret, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

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
        while self.is_running:
            with self.lock:
                jpeg = self.latest_jpeg_bytes
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.033)


# =====================================================================
# SECTION 3: AUTONOMOUS CLOSED-LOOP CONTROL ENGINE
# =====================================================================

class ASAFIAutonomousSystem:
    def __init__(self, hardware_mgr: HardwareManager, vision_pipe: VisionPipeline):
        self.hw = hardware_mgr
        self.vision = vision_pipe
        self.is_running = False
        self.worker_thread = None

        self.state = "SCANNING"  # SCANNING, TRACKING, VERIFYING, LOCKED
        self.scan_angle = 0.0
        self.scan_speed = 0.05
        self.lock_start_time = 0.0
        self.verification_hold_sec = 2.0
        self.alignment_tolerance_px = 45.0
        self.fire_temp_threshold_c = 55.0

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.hw.gimbal.center()
        self.worker_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.worker_thread.start()
        logger.info("[AUTONOMOUS] Closed-loop control engine started.")

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        self.hw.gimbal.center()

    def _sweep_area(self):
        self.scan_angle += self.scan_speed
        pan_target = 60.0 * math.sin(self.scan_angle)
        self.hw.gimbal.pan = pan_target
        self.hw.gimbal.tilt = 0.0
        self.hw.gimbal._write_servos(pan_target, 0.0)

    def _main_loop(self):
        while self.is_running:
            loop_start = time.perf_counter()

            vdata = self.vision.get_latest_data()
            target_found = vdata["target_detected"]
            primary_target = vdata["primary_target"]

            if self.state == "SCANNING":
                if target_found and primary_target:
                    logger.info(f"[STATE] Target found! -> TRACKING.")
                    self.state = "TRACKING"
                    self.lock_start_time = time.perf_counter()
                else:
                    self._sweep_area()

            elif self.state in ["TRACKING", "VERIFYING", "LOCKED"]:
                if not target_found or not primary_target:
                    logger.info("[STATE] Target lost -> Resuming SCANNING.")
                    self.state = "SCANNING"
                    continue

                cx, cy = primary_target["cx"], primary_target["cy"]
                tracking_status = self.hw.gimbal.update_tracking(cx, cy, frame_w=640, frame_h=480)
                distance_from_boresight = math.hypot(tracking_status["dx"], tracking_status["dy"])

                is_centered = distance_from_boresight <= self.alignment_tolerance_px

                if self.state == "TRACKING":
                    if is_centered:
                        logger.info("[STATE] Boresight aligned -> THERMAL VERIFICATION.")
                        self.state = "VERIFYING"
                        self.lock_start_time = time.perf_counter()

                elif self.state == "VERIFYING":
                    if (time.perf_counter() - self.lock_start_time) >= self.verification_hold_sec:
                        thermal_status = self.hw.thermal.read(fire_detected=True)
                        obj_temp = thermal_status["object_c"]

                        if obj_temp >= self.fire_temp_threshold_c:
                            logger.info(f"🔥 [REAL FIRE CONFIRMED] {obj_temp}°C >= {self.fire_temp_threshold_c}°C! LOCKED.")
                            self.state = "LOCKED"
                        else:
                            logger.warning(f"⚠️ [FALSE POSITIVE] Temp low ({obj_temp}°C). Resuming SCANNING.")
                            self.state = "SCANNING"

                elif self.state == "LOCKED":
                    self.hw.thermal.read(fire_detected=True)

            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            time.sleep(max(0.001, (1000.0 / 30.0 - elapsed_ms) / 1000.0))


# =====================================================================
# SECTION 4: FASTAPI SERVER & WEBSOCKET DASHBOARD BACKEND
# =====================================================================

hardware_mgr: HardwareManager = None
vision_pipe: VisionPipeline = None
auto_system: ASAFIAutonomousSystem = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hardware_mgr, vision_pipe, auto_system
    logger.info("==================================================")
    logger.info("    PROJECT ASAFI - UNIFIED MISSION SERVER        ")
    logger.info("==================================================")

    hardware_mgr = HardwareManager(force_mock=app.state.force_mock)
    model_path = os.path.join(REPO_DIR, app.state.model_path)
    vision_pipe = VisionPipeline(
        model_path=model_path,
        camera_index=app.state.camera_index,
        imgsz=app.state.imgsz,
        conf_threshold=0.35
    )
    vision_pipe.start()

    auto_system = ASAFIAutonomousSystem(hardware_mgr, vision_pipe)
    auto_system.start()

    logger.info(f"[SERVER] Dashboard active at http://{app.state.host}:{app.state.port}/")
    yield

    if auto_system:
        auto_system.stop()
    if vision_pipe:
        vision_pipe.stop()


app = FastAPI(title="ASAFI Mission Server", lifespan=lifespan)

if os.path.exists(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ASAFI Dashboard not found</h1>", status_code=404)


@app.get("/styles.css")
async def get_styles():
    css_file = os.path.join(DASHBOARD_DIR, "styles.css")
    if os.path.exists(css_file):
        return FileResponse(css_file, media_type="text/css")
    return HTMLResponse(status_code=404)


@app.get("/app.js")
async def get_app_js():
    js_file = os.path.join(DASHBOARD_DIR, "app.js")
    if os.path.exists(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    return HTMLResponse(status_code=404)


@app.get("/video_feed")
async def video_feed():
    if not vision_pipe:
        return HTMLResponse("Vision pipeline inactive", status_code=503)
    return StreamingResponse(
        vision_pipe.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
async def api_status():
    vision_data = vision_pipe.get_latest_data() if vision_pipe else {}
    hw_status = hardware_mgr.gimbal.get_status() if hardware_mgr else {}
    thermal_status = hardware_mgr.thermal.read() if hardware_mgr else {}

    return {
        "status": "ONLINE",
        "system_state": auto_system.state if auto_system else "OFFLINE",
        "vision": {
            "fps": vision_data.get("fps", 0),
            "inference_ms": vision_data.get("inference_ms", 0),
            "target_detected": vision_data.get("target_detected", False),
            "detections_count": len(vision_data.get("detections", []))
        },
        "hardware": {
            "gimbal": hw_status,
            "thermal": thermal_status
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"[WS] Client connected: {websocket.client.host}")

    try:
        while True:
            if not vision_pipe or not hardware_mgr or not auto_system:
                await asyncio.sleep(0.05)
                continue

            vision_data = vision_pipe.get_latest_data()
            target_detected = vision_data["target_detected"]

            packet = {
                "type": "FRAME_UPDATE",
                "frame": vision_data["base64_frame"],
                "fps": vision_data["fps"],
                "inference_ms": vision_data["inference_ms"],
                "target_detected": target_detected,
                "primary_target": vision_data["primary_target"],
                "telemetry": {
                    "gimbal": hardware_mgr.gimbal.get_status(),
                    "thermal": hardware_mgr.thermal.read(fire_detected=target_detected),
                    "system_state": auto_system.state
                }
            }

            await websocket.send_text(json.dumps(packet))
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {websocket.client.host}")
    except Exception as e:
        logger.warning(f"[WS] WebSocket error: {e}")


def main():
    parser = argparse.ArgumentParser(description="ASAFI Unified Mission Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO resolution")
    parser.add_argument("--mock-hardware", action="store_true", help="Force MOCK mode")
    args = parser.parse_args()

    app.state.host = args.host
    app.state.port = args.port
    app.state.camera_index = args.camera
    app.state.model_path = args.model
    app.state.imgsz = args.imgsz
    app.state.force_mock = args.mock_hardware

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()