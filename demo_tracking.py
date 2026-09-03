"""
Demo: Vehicle Detection & Lightweight Tracking on Raspberry Pi 5.
Powered by Ultralytics YOLOv8 nano & multi-object trajectory tracking.
"""

import argparse
import time
import cv2

import config
from core.vehicle_tracker import VehicleTracker
from core.camera_stream import CameraStream


def main():
    parser = argparse.ArgumentParser(description="Vehicle Tracking Demo for Raspberry Pi 5")
    parser.add_argument("-s", "--source", default="", help="Camera index (0, 1) or video file path")
    parser.add_argument("--conf", type=float, default=config.CAR_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=config.CAR_IOU_THRESHOLD, help="NMS IoU threshold")
    args = parser.parse_args()

    tracker = VehicleTracker(
        model_path=str(config.CAR_TRACKING_MODEL),
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        track_score=config.TRACK_SCORE_THRESHOLD,
        max_missed=config.TRACK_MAX_MISSED,
    )

    cam_src = args.source if args.source else config.CAMERA_SOURCE
    camera = CameraStream(
        source=cam_src,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
        use_picamera2=config.USE_PICAMERA2,
    )

    if not camera.is_opened():
        print(f"\n[ERROR] Could not open camera source '{cam_src}'.")
        print("Troubleshooting options:")
        print("  1. If using a USB webcam, try index 1 or 2: python3 demo_tracking.py --source 1")
        print("  2. Check connected video devices: v4l2-ctl --list-devices or ls -l /dev/video*")
        print("  3. If using RPi Camera Module ( ribbon ), set USE_PICAMERA2 = True in config.py")
        print("  4. If testing with a video file: python3 demo_tracking.py --source /path/to/video.mp4\n")
        camera.release()
        return

    print("=== Vehicle Tracking Demo (RPi 5) ===")
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

            tracks = tracker.track(frame)
            tracker.draw_tracks(frame, tracks)

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Vehicles Tracked: {len(tracks)}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if config.ENABLE_GUI:
                cv2.imshow("Vehicle Tracking Demo (RPi 5)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.03)

    finally:
        camera.release()
        if config.ENABLE_GUI:
            cv2.destroyAllWindows()
        print("[Tracking Demo] Exited.")


if __name__ == "__main__":
    main()
