# ASAFI: Autonomous Spacecraft Avionics Fire Interceptor

ASAFI is a closed-loop fire detection and tracking system built around a Raspberry Pi-style mission computer. The project combines a live camera feed, YOLO-based object detection, a thermal verification step, and a real-time telemetry dashboard to simulate and support autonomous fire interception logic.

This repository is not just a static model demo; it contains the full mission control stack in one Python entry point: `main.py`. The application starts a live vision pipeline, monitors hardware state, runs an autonomous scanning/tracking loop, and serves a web dashboard with live video and telemetry.

---

## Project Summary

The current implementation includes:

- Real-time camera capture using OpenCV
- YOLO inference with a trained model at `models/best.pt`
- Fallback flame detection using HSV color segmentation when the model is unavailable
- PCA9685 and servo gimbal control logic with mock fallback for development
- MLX90614 thermal sensor abstraction with mock-mode fallback
- Autonomous state machine: `SCANNING -> TRACKING -> VERIFYING -> LOCKED`
- FastAPI server for dashboard, MJPEG video feed, and WebSocket telemetry
- Tactical mission dashboard in the `dashboard/` folder

> The code currently focuses on detection, tracking, thermal confirmation, and telemetry streaming. Actual suppression-valve hardware control is not implemented in the current codebase, even though the concept is part of the broader project vision.

---

## Repository Structure

```text
ASAFI/
├── main.py                 # Unified backend: vision, hardware, control logic, server
├── dashboard/
│   ├── index.html          # Mission dashboard UI
│   ├── app.js              # WebSocket client and HUD telemetry logic
│   └── styles.css          # Tactical dashboard styling
├── models/
│   ├── best.pt             # YOLO model weights
│   └── readme.md           # Model-specific notes
├── LICENSE                 # Project license
├── README.md               # Project documentation
├── __pycache__/            # Generated Python cache files
└── .git/                   # Git metadata
```

---

## Core File-by-File Breakdown

### `main.py`
This is the central application file. It contains the full runtime stack in one place and is structured into major sections:

1. Hardware drivers
   - `PCA9685Driver`: low-level PWM driver for servo control over I2C
   - `GimbalController`: pan/tilt positioning logic and mock fallback
   - `ThermalSensor`: MLX90614 read logic with thermal threshold logic
   - `HardwareManager`: combines gimbal and thermal subsystems

2. Vision pipeline
   - `VisionPipeline`: opens the camera, runs YOLO inference, annotates frames, and exposes latest detections
   - Includes fallback flame detection using HSV thresholding
   - Produces annotated JPEG frames and base64 video payloads

3. Autonomous control engine
   - `ASAFIAutonomousSystem`: scans the scene, aligns on a target, verifies temperature, and enters a locked state when a real fire is confirmed
   - State transitions are based on centroid tracking and thermal threshold validation

4. FastAPI server
   - Serves the dashboard at `/`
   - Exposes `/video_feed` for MJPEG streaming
   - Exposes `/api/status` for telemetry snapshots
   - Exposes `/ws` for live dashboard updates

### `dashboard/index.html`
This file builds the mission HUD interface. It includes:

- live camera view
- system status badges
- target lock indicators
- FPS and latency readouts
- gimbal angle and thermal display
- tactical overlay styling for a spacecraft/avionics feel

### `dashboard/app.js`
This client performs the browser-side WebSocket communication. It:

- connects to the FastAPI backend WebSocket
- receives `FRAME_UPDATE` packets
- updates the UI with the video frame and telemetry
- renders target centroids, confidence, and gimbal status

### `dashboard/styles.css`
Defines the dark aerospace dashboard theme, including:

- scanline effect
- HUD framing corners
- status badges, labels, and metric layout
- target and telemetry styling
- responsive layout for the streaming view

### `models/best.pt`
This is the trained YOLO model used for fire/object detection. If the file is missing or unavailable, the app falls back to an HSV-based flame detector instead of failing.

---

## System Architecture

### Hardware Layer
- Camera input from `/dev/video0` or another OpenCV-supported source
- PCA9685 PWM board for servo actuation
- MG996R-style pan/tilt servos for target tracking
- MLX90614 thermal sensor for object temperature verification
- Supported in physical mode when I2C devices are available
- Automatically falls back to `MOCK` mode when hardware is missing

### Software Layer
- Python + OpenCV for image processing
- Ultralytics YOLO for inference
- FastAPI + Uvicorn for serving the dashboard backend
- WebSockets for real-time telemetry streaming
- Threading for the camera loop and control loop

### Mission Logic
The autonomous loop behaves like this:

1. Start the camera and detection model
2. Search the scene in scanning mode
3. Detect the highest-confidence target
4. Use gimbal feedback to center the target
5. Verify object temperature against the threshold
6. Enter `LOCKED` state when the target is both centered and hot enough
7. Stream the current state to the dashboard in real time

---

## Runtime Flow

```text
Camera Feed
   ↓
VisionPipeline
   ↓
YOLO inference / fallback flame detection
   ↓
Target selection + centroid tracking
   ↓
ASAFIAutonomousSystem
   ↓
Gimbal update + thermal verification
   ↓
FastAPI dashboard / WebSocket telemetry
```

---

## How to Run

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn opencv-python numpy ultralytics smbus2
```

If you are working on a development machine without Raspberry Pi hardware, you can run the app in mock mode:

```bash
python main.py --mock-hardware
```

### 3. Launch the mission server

```bash
python main.py --host 0.0.0.0 --port 8000 --camera 0
```

### 4. Open the dashboard

Visit:

```text
http://localhost:8000/
```

The UI will show:

- live video feed
- detection confidence
- centroid position
- gimbal state
- thermal readings

---

## Operational Notes

- The project is designed to run both with real I2C hardware and without it.
- If the camera or sensors are unavailable, the logic gracefully degrades to mock behavior.
- The app uses `best.pt` when present; otherwise it falls back to color-based flame detection.
- Current telemetry and mission logic are tuned for live demonstration and rapid iteration rather than final flight-grade deployment.

---

## Current Scope and Limitations

The current repository implements a strong prototype and mission-control platform, but it does not yet include a production-ready suppression subsystem or full safety-certified flight logic. The major implemented components are:

- computer vision-based fire detection
- target tracking and alignment
- thermal confirmation loop
- mission dashboard and telemetry interface

This means the project is best described as an autonomous proof-of-concept for fire interception rather than a complete hardware-suppression deployment package.

---

## Team / Project Context

This project was structured as a collaborative avionics and mission-control effort with separate workstreams for:

- computer vision and sensing
- servo and actuation logic
- dashboard and telemetry interface
- documentation and product presentation

The implementation in this repository reflects that split, with the unified backend orchestrating all of it from a single application.

---

## License

This project is released under the MIT-style license included in the repository. See `LICENSE` for details.
