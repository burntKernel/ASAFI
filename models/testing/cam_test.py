import cv2
import time
from ultralytics import YOLO

# ASAFI configuration
MODEL_PATH = r"models\best.pt"
CAMERA_INDEX = 1  # External USB webcam
CONFIDENCE = 0.35
IMAGE_SIZE = 640

def main():
    print("Loading ASAFI fire/smoke model...")
    model = YOLO(MODEL_PATH)
    print("Model classes:", model.names)

    print(f"Opening USB webcam at index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Check that the USB webcam is connected."
        )

    print("Live detection started. Press Q in the video window to quit.")

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False
            )

            annotated_frame = results[0].plot()

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                annotated_frame,
                f"ASAFI | FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "ASAFI - Live Fire and Smoke Detection",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("ASAFI webcam test stopped.")

if __name__ == "__main__":
    main()
