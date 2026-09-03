"""
Demo: License Plate Recognition (ALPR) on Raspberry Pi 5.
Supports live camera feed or single test image.
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np

import config
from core.plate_engine import PlateEngine
from core.camera_stream import CameraStream


def main():
    parser = argparse.ArgumentParser(description="ALPR Demo for Raspberry Pi 5")
    parser.add_argument("-i", "--image", type=str, default="", help="Path to single image file for testing")
    parser.add_argument("-s", "--source", default="", help="Camera index (0, 1) or video path")
    args = parser.parse_args()

    engine = PlateEngine(
        detector_path=str(config.PLATE_DETECTOR_MODEL),
        ocr_path=str(config.PLATE_OCR_MODEL),
        yolov5_dir=str(config.YOLOV5_DIR),
        det_conf=config.PLATE_DET_CONF,
        ocr_conf=config.PLATE_OCR_CONF,
        img_size=config.PLATE_IMG_SIZE,
        device="cpu",
    )

    # 1. Single Image Mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[Error] Image not found: {img_path}")
            return

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Error] Could not read image: {img_path}")
            return

        t0 = time.time()
        detections = engine.detect_and_read(img)
        dt = (time.time() - t0) * 1000.0

        print(f"\n--- ALPR Detection Result ({dt:.1f} ms) ---")
        if len(detections) == 0:
            print("No license plates detected.")
        else:
            for idx, d in enumerate(detections, 1):
                print(f"Plate #{idx}: '{d['plate']}' (Confidence: {d['conf']:.2f}, Box: {d['bbox']})")

        engine.draw_plates(img, detections)
        cv2.putText(
            img,
            f"Infer: {dt:.1f}ms | Plates: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if config.ENABLE_GUI:
            cv2.imshow("ALPR Image Test (RPi 5)", img)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Live Camera Mode
    cam_src = args.source if args.source else config.CAMERA_SOURCE
    camera = CameraStream(
        source=cam_src,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
        use_picamera2=config.USE_PICAMERA2,
    )

    print("=== ALPR Live Stream (RPi 5) ===")
    print("Press 'Q' to exit.")

    prev_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            detections = engine.detect_and_read(frame)
            engine.draw_plates(frame, detections)

            # Top HUD
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Detected Plates: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if config.ENABLE_GUI:
                cv2.imshow("ALPR Camera Demo (RPi 5)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if len(detections) > 0:
                    for d in detections:
                        print(f"[{time.strftime('%H:%M:%S')}] Detected Plate: {d['plate']} (conf={d['conf']:.2f})")
                time.sleep(0.03)

    finally:
        camera.release()
        if config.ENABLE_GUI:
            cv2.destroyAllWindows()
        print("[ALPR Demo] Exited.")


if __name__ == "__main__":
    main()
