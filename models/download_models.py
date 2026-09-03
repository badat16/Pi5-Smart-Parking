"""
Helper script to download and verify all AI models on Raspberry Pi 5.
Downloads InsightFace 'buffalo_sc' to ~/.insightface/models/buffalo_sc/
and checks all local YOLO weights.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
INSIGHTFACE_HOME = Path.home() / ".insightface" / "models"
BUFFALO_SC_DIR = INSIGHTFACE_HOME / "buffalo_sc"

BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

LOCAL_WEIGHTS = [
    MODELS_DIR / "LP_detector_nano_61.pt",
    MODELS_DIR / "LP_ocr_nano_62.pt",
    MODELS_DIR / "best.pt",
]


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()


def setup_insightface_model():
    BUFFALO_SC_DIR.mkdir(parents=True, exist_ok=True)
    onnx_files = list(BUFFALO_SC_DIR.glob("*.onnx"))

    if len(onnx_files) >= 2:
        print(f"[OK] InsightFace 'buffalo_sc' already installed at: {BUFFALO_SC_DIR}")
        for f in onnx_files:
            print(f"     - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        return True

    print(f"\n[INFO] Downloading InsightFace 'buffalo_sc' model pack...")
    zip_path = BUFFALO_SC_DIR / "buffalo_sc.zip"

    try:
        urllib.request.urlretrieve(BUFFALO_SC_URL, zip_path, reporthook=download_progress)
        print("\n[INFO] Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(BUFFALO_SC_DIR)

        if zip_path.exists():
            zip_path.unlink()

        print(f"[SUCCESS] InsightFace buffalo_sc installed at: {BUFFALO_SC_DIR}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download buffalo_sc: {e}")
        print(f"Please manually download {BUFFALO_SC_URL} and extract to {BUFFALO_SC_DIR}")
        return False


def verify_local_weights():
    print("\n--- Verifying Local YOLO Nano Weights ---")
    all_ok = True
    for w in LOCAL_WEIGHTS:
        if w.exists() and w.stat().st_size > 1000:
            print(f"[OK] Found weight: {w.name} ({w.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"[MISSING] Weight file not found or corrupted: {w}")
            all_ok = False
    return all_ok


def main():
    print("==================================================")
    print(" Smart Parking RPi 5 - Model Verifier & Downloader ")
    print("==================================================")
    weights_ok = verify_local_weights()
    face_ok = setup_insightface_model()

    if weights_ok and face_ok:
        print("\n>>> All AI models are ready to run on Raspberry Pi 5! <<<")
    else:
        print("\n>>> WARNING: Some models are missing. Please check above logs. <<<")


if __name__ == "__main__":
    main()
