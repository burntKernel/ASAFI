"""
ASAFI - Autonomous Spacecraft Avionics Fire Interceptor
FastAPI Mission Server & Video Streaming Backend
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add repository root to path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from hardware import HardwareManager
from vision import VisionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASAFI.Server")

# Global instances
hardware_mgr: HardwareManager = None
vision_pipe: VisionPipeline = None
DASHBOARD_DIR = os.path.join(REPO_DIR, "dashboard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hardware_mgr, vision_pipe
    logger.info("==================================================")
    logger.info("   PROJECT ASAFI - AVIONICS MISSION SERVER        ")
    logger.info("==================================================")

    # Initialize Hardware Manager (Graceful I2C fallback to MOCK)
    hardware_mgr = HardwareManager(force_mock=app.state.force_mock)

    # Initialize Vision Pipeline (/dev/video0 + YOLO)
    model_path = os.path.join(REPO_DIR, app.state.model_path)
    vision_pipe = VisionPipeline(
        model_path=model_path,
        camera_index=app.state.camera_index,
        imgsz=app.state.imgsz,
        conf_threshold=0.35
    )
    vision_pipe.start()

    logger.info(f"[SERVER] Dashboard available at http://{app.state.host}:{app.state.port}/")
    yield

    logger.info("[SERVER] Shutting down ASAFI Mission Server...")
    if vision_pipe:
        vision_pipe.stop()


app = FastAPI(title="ASAFI Mission Server", lifespan=lifespan)

# Mount dashboard static directory
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
    """HTTP MJPEG stream fallback."""
    if not vision_pipe:
        return HTMLResponse("Vision pipeline inactive", status_code=503)
    return StreamingResponse(
        vision_pipe.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
async def api_status():
    """System health and telemetry status endpoint."""
    vision_data = vision_pipe.get_latest_data() if vision_pipe else {}
    hw_status = hardware_mgr.gimbal.get_status() if hardware_mgr else {}
    thermal_status = hardware_mgr.thermal.read() if hardware_mgr else {}

    return {
        "status": "ONLINE",
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
    """
    High-speed WebSocket telemetry & video stream endpoint.
    Transmits annotated frame + detection & hardware telemetry to client.
    """
    await websocket.accept()
    logger.info(f"[WS] Client connected from {websocket.client.host}")

    try:
        while True:
            if not vision_pipe or not hardware_mgr:
                await asyncio.sleep(0.05)
                continue

            vision_data = vision_pipe.get_latest_data()
            target_detected = vision_data["target_detected"]
            primary = vision_data["primary_target"]

            # Update pan-tilt tracking and read thermal sensor
            if target_detected and primary:
                hw_telemetry = hardware_mgr.update(
                    fire_detected=True,
                    cx=primary["cx"],
                    cy=primary["cy"],
                    frame_w=640,
                    frame_h=480
                )
            else:
                hw_telemetry = hardware_mgr.update(fire_detected=False)

            # Build WebSocket packet
            packet = {
                "type": "FRAME_UPDATE",
                "frame": vision_data["base64_frame"],
                "fps": vision_data["fps"],
                "inference_ms": vision_data["inference_ms"],
                "target_detected": target_detected,
                "primary_target": primary,
                "telemetry": hw_telemetry
            }

            await websocket.send_text(json.dumps(packet))

            # Stream at ~25-30 FPS
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {websocket.client.host}")
    except Exception as e:
        logger.warning(f"[WS] Connection error: {e}")


def main():
    parser = argparse.ArgumentParser(description="ASAFI Avionics Mission Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 Camera index (default: 0 for /dev/video0)")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution (default: 320 for Pi CPU)")
    parser.add_argument("--mock-hardware", action="store_true", help="Force MOCK hardware mode")
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
