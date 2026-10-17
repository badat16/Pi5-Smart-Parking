"""
Demo: Basic Camera License Plate Recognition (ALPR) on Raspberry Pi 5.
Tests basic license plate detection and OCR capability from camera stream or static image.
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
    parser = argparse.ArgumentParser(description="Basic ALPR Camera Demo for Raspberry Pi 5")
    parser.add_argument("-i", "--image", type=str, default="", help="Path to single image file for testing")
    parser.add_argument("-s", "--source", default="", help="Camera index (0, 1) or video path")
    args = parser.parse_args()

    print("[INFO] Initializing Plate Engine...")
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

        if getattr(config, "ENABLE_GUI", True):
            cv2.imshow("ALPR Image Test", img)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Live Camera Detection & OCR Mode
    cam_src = args.source if args.source else config.CAMERA_SOURCE
    # Try converting numeric string to int for webcam index
    try:
        cam_src = int(cam_src)
    except (ValueError, TypeError):
        pass

    camera = CameraStream(
        source=cam_src,
        width=getattr(config, "CAMERA_WIDTH", 640),
        height=getattr(config, "CAMERA_HEIGHT", 480),
        fps=getattr(config, "CAMERA_FPS", 30),
        use_picamera2=getattr(config, "USE_PICAMERA2", False),
    )

    if not camera.is_opened():
        print(f"\n[ERROR] Could not open camera source '{cam_src}'.")
        print("Troubleshooting options:")
        print("  1. If using USB webcam, try: python3 demo_plate.py --source 0 (or 1, 2)")
        print("  2. Check connected video devices: v4l2-ctl --list-devices or ls -l /dev/video*")
        print("  3. If using RPi Camera Module, set USE_PICAMERA2 = True in config.py")
        print("  4. Test a static image: python3 demo_plate.py --image test_images/1.jpg\n")
        camera.release()
        return

    print("\n=======================================================")
    print("  BASIC CAMERA ALPR DEMO (DETECTION & OCR)")
    print("=======================================================")
    print("  - Streaming from camera source...")
    print("  - [Q / ESC]  : Quit application")
    print("  - [SPACE / S]: Print current detection details to console\n")

    enable_gui = getattr(config, "ENABLE_GUI", True)

    try:
        prev_time = time.time()
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            detections = engine.detect_and_read(frame)
            infer_dt = (time.time() - t0) * 1000.0

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            # Draw bounding boxes & text on frame
            engine.draw_plates(frame, detections)

            # Draw overlay status banner
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)
            status_text = f"FPS: {fps:.1f} | Infer: {infer_dt:.1f}ms | Plates: {len(detections)}"
            cv2.putText(
                frame,
                status_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Print to terminal if plate detected
            for d in detections:
                if d["plate"] != "unknown":
                    print(f"[DETECTED] Plate: '{d['plate']}' | Conf: {d['conf']:.2f} | BBox: {d['bbox']}")

            key = -1
            if enable_gui:
                cv2.imshow("ALPR Basic Camera Test", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                time.sleep(0.03)

            if key in (27, ord("q"), ord("Q")):
                print("\n[INFO] Exiting...")
                break

            if key in (32, ord("s"), ord("S"), ord("c"), ord("C")):
                print(f"\n--- Snapshot Detections ({infer_dt:.1f} ms) ---")
                if not detections:
                    print("No plates detected in current frame.")
                else:
                    for idx, d in enumerate(detections, 1):
                        print(f"  #{idx}: Plate='{d['plate']}', Conf={d['conf']:.2f}, Box={d['bbox']}")
                print("-------------------------------------------\n")

    finally:
        camera.release()
        if enable_gui:
            cv2.destroyAllWindows()
        print("[ALPR Demo] Camera released. Done.")


if __name__ == "__main__":
    main()
