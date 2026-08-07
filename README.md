# ASAFI: Autonomous Spacecraft Avionics Fire Interceptor

ASAFI is an autonomous avionics payload designed to detect thermal anomalies and fires using optical computer vision and infrared sensing, track target coordinates via a 2-axis pan-tilt gimbal, and deploy localized cold-gas suppression pulses via a micro-solenoid valve.

---

## 🛠️ System Architecture

* **Core Processing:** Raspberry Pi 4 / 5
* **Computer Vision:** OpenCV / YOLO Fire Detection Pipeline
* **Thermal Verification:** MLX90614 Non-Contact IR Sensor (I²C)
* **Actuation:** PCA9685 PWM Driver + MG996R Servos (2-Axis Pan-Tilt)
* **Suppression Mechanism:** 12V Micro-Solenoid Valve + 5V Relay Module
* **Power Delivery:** 12V 5A DC Power Input + Dual Buck Converters

---

## 👥 Team Structure & Allocation

| Pair | Members | Domain Responsibility |
| :--- | :--- | :--- |
| **Pair 1** | Manas & Pradyumna | Computer Vision (OpenCV/YOLO), Thermal Sensor Fusion & Servo/Solenoid Actuation |
| **Pair 2** | Kavyansh & Deepanshu | Mission Control Dashboard, WebSockets Telemetry & Remote Override System |
| **Pair 3** | Paakhi & Rahul | Interactive Landing Page, Technical Documentation, Poster & ECS Reimbursement |

---

## 🚀 Getting Started

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install opencv-python numpy smbus2 websockets
