"""
Configuration file for Smart Parking System on Raspberry Pi 5.
Contains hardware settings, AI model paths, detection thresholds, and camera options.
"""

from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YOLOV5_DIR = BASE_DIR / "yolov5"
DB_PATH = BASE_DIR / "parking_system.db"

# Model weights
PLATE_DETECTOR_MODEL = MODELS_DIR / "LP_detector_nano_61.pt"
PLATE_OCR_MODEL = MODELS_DIR / "LP_ocr_nano_62.pt"
CAR_TRACKING_MODEL = MODELS_DIR / "best.pt"
FACE_MODEL_NAME = "buffalo_sc"

# Camera Settings
# Set CAMERA_SOURCE to an integer (0, 1) for USB webcam, or a video path/RTSP string
CAMERA_SOURCE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20
# Set to True if using the official Raspberry Pi Camera Module (v2, v3, HQ) with libcamera/Picamera2
USE_PICAMERA2 = False

# Face Recognition Settings (InsightFace buffalo_sc)
FACE_DET_THRESHOLD = 0.50
FACE_MATCH_THRESHOLD = 0.45  # Cosine similarity threshold for driver verification
FACE_DET_SIZE = (320, 320)   # Optimal detection resolution for Cortex-A76 CPU
FACE_MIN_SIZE = 45           # Minimum width/height in pixels to reject blurry distant faces

# License Plate Recognition Settings (ALPR)
PLATE_DET_CONF = 0.25   # Lowered threshold to detect plates reliably on webcams/Pi
PLATE_OCR_CONF = 0.25   # Lowered OCR threshold to read all plate characters
PLATE_IMG_SIZE = 640
ALPR_CAPTURE_WIDTH = 1920
ALPR_CAPTURE_HEIGHT = 1080

# Entry & Exit Gate Boxes (2 ô ra vào ở phía trên giữa 2 ô màu xanh): [x1, y1, x2, y2] relative to 640x480
ENTRY_LANE_BOX = [241, 42, 322, 117]   # Ô Vào (↓ - Phía trái): Lật 180° khi chụp ALPR
EXIT_LANE_BOX = [329, 42, 411, 117]    # Ô Ra (↑ - Phía phải): Giữ nguyên (không lật) khi chụp ALPR

# Legacy compatibility
GATE_LANE_ROI_BOX = [235, 10, 400, 110]

# Car Tracking Settings
CAR_CONF_THRESHOLD = 0.45
CAR_IOU_THRESHOLD = 0.60
CAR_TRACKING_IMGSIZE = 320   # 320x320 optimal for Raspberry Pi 5 CPU (10-25 FPS)
TRACK_SCORE_THRESHOLD = 0.20
TRACK_MAX_MISSED = 35

# ROI & Crossing Line Parking Settings
# 1. Crossing Line (Vạch cắt ngang nằm ngay dưới ô màu xanh tại Y = 110)
ENABLE_CROSSING_LINE_FILTER = True
CROSSING_LINE_Y = 147  # Horizontal crossing line right below the top green patches
CROSSING_LINE_COLOR = (0, 255, 255)  # Cyan line

# 2. 4 Parking Zones (Khu A, B, C, D) with 3 slots each
# Zone A: Top-Left (Dưới ô xanh bên trái)
# Zone B: Top-Right (Dưới ô xanh bên phải)
# Zone C: Bottom-Left (Khu C dưới cùng bên trái)
# Zone D: Bottom-Right (Khu D dưới cùng bên phải)
PARKING_ZONES = {
    "A": {
        "name": "Khu A (Trái Trên)",
        "slots": {
            "A1": [[77, 147], [136, 147], [136, 234], [77, 234]],
            "A2": [[136, 147], [195, 147], [195, 234], [136, 234]],
            "A3": [[195, 147], [255, 147], [255, 234], [195, 234]],
        },
    },
    "B": {
        "name": "Khu B (Phải Trên)",
        "slots": {
            "B1": [[395, 147], [454, 147], [454, 234], [395, 234]],
            "B2": [[454, 147], [514, 147], [514, 234], [454, 234]],
            "B3": [[514, 147], [574, 147], [574, 234], [514, 234]],
        },
    },
    "C": {
        "name": "Khu C (Trái Dưới)",
        "slots": {
            "C1": [[77, 352], [136, 352], [136, 440], [77, 440]],
            "C2": [[136, 352], [195, 352], [195, 440], [136, 440]],
            "C3": [[195, 352], [255, 352], [255, 440], [195, 440]],
        },
    },
    "D": {
        "name": "Khu D (Phải Dưới)",
        "slots": {
            "D1": [[395, 352], [454, 352], [454, 440], [395, 440]],
            "D2": [[454, 352], [514, 352], [514, 440], [454, 440]],
            "D3": [[514, 352], [574, 352], [574, 440], [514, 440]],
        },
    },
}

# Legacy ROI Filter compatibility (combined area of all parking spaces)
ENABLE_ROI_FILTER = True
ROI_FILTER_TYPE = "centroid"
PARKING_ROI_POLYGONS = [
    [[85, 15], [570, 15], [570, 290], [85, 290]],
]

# Visualization styles
SHOW_ROI_BOUNDARY = True
SHOW_SLOT_LABELS = True
SLOT_VACANT_COLOR = (0, 255, 0)     # Green for empty spot
SLOT_OCCUPIED_COLOR = (0, 0, 255)   # Red for occupied spot
ROI_FILL_ALPHA = 0.00   # 100% transparent interior (bên trong trong suốt, chỉ vẽ viền)

# Performance & Multiprocessing (Raspberry Pi 5 has 4 Cortex-A76 cores)
ONNX_NUM_THREADS = 4
SKIP_FRAMES = 0  # Process every frame (0), or skip frames (1, 2) to reduce CPU load

# GUI / Display
ENABLE_GUI = True  # Set to False when running headless over SSH without X11

