# ASAFI — Fire & Smoke Detection Setup

## 1. Requirements

- Python 3.11
- USB webcam
- Trained YOLO model: `best.pt`
- Windows 10/11 for the commands below

## 2. Installation

Clone the repository and open a terminal in the project directory.

Create and activate a virtual environment (Windows PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the required libraries:

```powershell
python -m pip install ultralytics opencv-python
```

### Main dependencies

- `ultralytics` — loads the trained YOLO model and performs inference.
- `opencv-python` — webcam capture and live video display.
- `torch` — deep-learning inference backend, installed as an Ultralytics dependency when required.
- `numpy` — numerical processing dependency.

A CUDA-enabled GPU is optional. CPU inference is supported for local testing.

## 3. Model Location

Place the trained model in the repository's `Models` directory, or update the model path in the testing script accordingly.

Example structure:

```text
ASAFI/
├── Models/
│   ├── best.pt
│   └── testing/
│       └── cam_test.py
└── .venv/
```

**Important:** The model path configured in `cam_test.py` must match the actual location of `best.pt`.

## 4. Verify the Installation

Run the following command after activating the virtual environment:

```powershell
python -c "import torch, cv2; from ultralytics import YOLO; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

This checks whether the required libraries are installed and whether CUDA is available.

## 5. Run Live Webcam Detection

From the repository root, run:

```powershell
python -u Models/testing/cam_test.py
```

The webcam window should display detected fire and smoke with bounding boxes, class labels, and confidence scores.

- Camera index `1` is configured for an external USB webcam.
- If the camera does not open, try index `0`.
- Press **Q** to close the webcam window.

### CPU and GPU Configuration

If PyTorch is installed as a CPU-only build, configure the model inference call in `cam_test.py` to use:

```python
device="cpu"
```

Do not use `device=0` unless a compatible CUDA-enabled PyTorch installation and NVIDIA GPU are available.

## 6. Troubleshooting

**Model not found:** Verify that `best.pt` exists at the path configured in the testing script.

**Module not found:** Activate the virtual environment and rerun the dependency installation commands.

**CUDA device error:** Set `device="cpu"` when using CPU-only PyTorch.

**Webcam not detected:** Check the camera connection and try another camera index.

**Slow inference:** CPU inference may be slower than GPU inference.

## Note

This setup is for local software testing of fire and smoke detection. Raspberry Pi deployment and hardware integration will be configured separately.
