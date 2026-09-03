"""
Clean Standalone Camera Test Script for Raspberry Pi 5.
Displays pure live camera stream without any text or HUD overlay.
"""

import argparse
import time
import cv2
import config
from core.camera_stream import CameraStream


def main():
    parser = argparse.ArgumentParser(description="Clean Camera Stream Test")
    parser.add_argument("-s", "--source", default="", help="Camera index (0, 1, 2) or video file/stream")
    parser.add_argument("-w", "--width", type=int, default=config.CAMERA_WIDTH, help="Camera width")
    parser.add_argument("-H", "--height", type=int, default=config.CAMERA_HEIGHT, help="Camera height")
    args = parser.parse_args()

    cam_src = args.source if args.source else config.CAMERA_SOURCE

    print("\n=======================================================")
    print("  Raspberry Pi 5 Clean Camera Stream                   ")
    print("=======================================================")
    print(f"  - Nguồn Camera: {cam_src}")
    print("  - Phím bấm:")
    print("      [S]   : Lưu ảnh chụp khung hình (test_images/camera_snapshot.jpg)")
    print("      [Q/ESC]: Tắt camera và thoát\n")

    camera = CameraStream(
        source=cam_src,
        width=args.width,
        height=args.height,
        fps=config.CAMERA_FPS,
        use_picamera2=config.USE_PICAMERA2,
    )

    if not camera.is_opened():
        print(f"[LỖI] Không thể mở nguồn camera '{cam_src}'.")
        print("Gợi ý:")
        print("  1. Thử cổng camera khác: python3 test_camera.py --source 1")
        print("  2. Kiểm tra thiết bị: v4l2-ctl --list-devices hoặc ls /dev/video*")
        camera.release()
        return

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Pure clean camera display without any text overlays
            cv2.imshow("Camera Live Stream", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                break

            if key == ord("s") or key == ord("S"):
                out_path = "test_images/camera_snapshot.jpg"
                cv2.imwrite(out_path, frame)
                print(f"[Snapshot] Đã lưu ảnh chụp khung hình vào {out_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[Camera Test] Đã tắt camera.")


if __name__ == "__main__":
    main()
