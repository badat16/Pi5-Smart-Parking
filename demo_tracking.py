"""
Pure Vehicle Tracking & Parking Occupancy System for Raspberry Pi 5.
Focuses exclusively on real-time vehicle detection, trajectory tracking,
and 12-slot 4-zone occupancy monitoring (A1..D3).
"""

import argparse
import time
import cv2
import numpy as np

import config
from core.vehicle_tracker import VehicleTracker
from core.camera_stream import CameraStream


def main():
    parser = argparse.ArgumentParser(description="Pure Vehicle Tracking & Parking Occupancy Monitoring (RPi 5)")
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
        imgsz=config.CAR_TRACKING_IMGSIZE,
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
        print("  1. Try index 0 or 1: python3 demo_tracking.py --source 0")
        print("  2. Check connected devices: ls -l /dev/video*")
        camera.release()
        return

    print("\n=======================================================================")
    print("   HỆ THỐNG THEO DÕI XE & QUẢN LÝ TRẠNG THÁI Ô ĐẬU THÔNG MINH (RPi 5)  ")
    print("=======================================================================")
    print("  - Chế độ      : TẬP TRUNG PURITY TRACKING (Tốc độ tối đa trên Pi 5)")
    print("  - Giám sát    : 4 Khu A, B, C, D (Tổng 12 Ô đỗ xe A1..D3)")
    print("  - Ranh giới   : Vạch tracking Y và Vùng Active Zone")
    print("  - Phím [Q]    : Thoát ứng dụng\n")

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

            h, w = frame.shape[:2]

            # 1. Execute Vehicle Tracking & Slot Occupancy Detection
            tracks = tracker.track(frame)
            occupancy = tracker.get_slot_occupancy(tracks)

            # 2. Draw Slot Boundaries, Active Tracking Box & Vehicle Trackers
            display_frame = tracker.draw_tracks(frame, tracks)

            # 3. Calculate Zone Occupancy Statistics
            zone_counts = {"A": [0, 0], "B": [0, 0], "C": [0, 0], "D": [0, 0]}
            total_occupied = 0

            for slot_id, slot_info in occupancy.items():
                z_id = slot_info["zone_id"]
                if z_id in zone_counts:
                    zone_counts[z_id][1] += 1  # Total slots in zone
                    if slot_info["occupied"]:
                        zone_counts[z_id][0] += 1  # Occupied count
                        total_occupied += 1

            total_slots = len(occupancy)
            total_vacant = max(0, total_slots - total_occupied)

            # 4. Top Status Dashboard Banner
            cv2.rectangle(display_frame, (0, 0), (w, 36), (15, 15, 15), -1)
            status_text = (
                f"KHU A: {zone_counts['A'][0]}/{zone_counts['A'][1]}  |  "
                f"KHU B: {zone_counts['B'][0]}/{zone_counts['B'][1]}  |  "
                f"KHU C: {zone_counts['C'][0]}/{zone_counts['C'][1]}  |  "
                f"KHU D: {zone_counts['D'][0]}/{zone_counts['D'][1]}  |  "
                f"DANG DO: {total_occupied}/{total_slots}  |  TRONG: {total_vacant}"
            )
            cv2.putText(
                display_frame,
                status_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 5. Bottom HUD Bar
            cv2.rectangle(display_frame, (0, h - 30), (w, h), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | TRACKING XE: {len(tracks)} xe | CHE DO: PURE VEHICLE TRACKING | [Q]: THOAT",
                (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if config.ENABLE_GUI:
                cv2.imshow("Pure Vehicle Tracking & Parking System (RPi 5)", display_frame)
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            else:
                time.sleep(0.03)

    finally:
        camera.release()
        if config.ENABLE_GUI:
            cv2.destroyAllWindows()
        print("[Tracking System] Exited cleanly.")


if __name__ == "__main__":
    main()
