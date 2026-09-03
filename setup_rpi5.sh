#!/usr/bin/env bash
# =============================================================================
# Automated Setup Script for Smart Parking System on Raspberry Pi 5
# Tested on Raspberry Pi OS (64-bit Debian 12 Bookworm)
# =============================================================================

set -e

echo "=================================================================="
echo "  Setting up Smart Parking Management System on Raspberry Pi 5   "
echo "=================================================================="

# 1. Update system packages
echo -e "\n[1/5] Updating system packages and installing OS dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    libcap-dev \
    ffmpeg \
    libatlas-base-dev \
    v4l-utils \
    libcamera-tools \
    git \
    wget \
    unzip

# 2. Setup Python Virtual Environment
echo -e "\n[2/5] Creating Python virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
    echo "[OK] Virtual environment created."
else
    echo "[INFO] Virtual environment already exists."
fi

# 3. Activate environment & upgrade pip
echo -e "\n[3/5] Activating virtual environment & upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install Python dependencies
echo -e "\n[4/5] Installing Python libraries from requirements.txt..."
pip install -r requirements.txt

# 5. Download and verify AI models
echo -e "\n[5/5] Downloading and verifying AI models (InsightFace & YOLO nano)..."
python3 models/download_models.py

# Make scripts executable
chmod +x main_parking_system.py demo_face.py demo_plate.py demo_tracking.py

echo -e "\n=================================================================="
echo "  [SUCCESS] CÀI ĐẶT HOÀN TẤT TRÊN RASPBERRY PI 5!                "
echo "=================================================================="
echo "Để chạy hệ thống, hãy thực hiện:"
echo "  1. Kích hoạt môi trường:  source .venv/bin/activate"
echo "  2. Chạy cổng thông minh:  python3 main_parking_system.py"
echo "  3. Chạy test Biển số:     python3 demo_plate.py"
echo "  4. Chạy test Khuôn mặt:   python3 demo_face.py"
echo "  5. Chạy test Tracking xe: python3 demo_tracking.py"
echo "=================================================================="
