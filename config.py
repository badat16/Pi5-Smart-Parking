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
CAMERA_SOURCE = 1
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
PLATE_DET_CONF = 0.50
PLATE_OCR_CONF = 0.55
PLATE_IMG_SIZE = 640

# Car Tracking Settings
CAR_CONF_THRESHOLD = 0.45
CAR_IOU_THRESHOLD = 0.60
TRACK_SCORE_THRESHOLD = 0.20
TRACK_MAX_MISSED = 35

# Performance & Multiprocessing (Raspberry Pi 5 has 4 Cortex-A76 cores)
ONNX_NUM_THREADS = 4
SKIP_FRAMES = 0  # Process every frame (0), or skip frames (1, 2) to reduce CPU load

# GUI / Display
ENABLE_GUI = True  # Set to False when running headless over SSH without X11
