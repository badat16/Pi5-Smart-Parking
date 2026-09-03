"""
Interactive Proportional Grid & ROI Selector Tool for Smart Parking System.
Allows dragging a bounding box to automatically calculate & scale all 12 parking slots (A1..D3),
Entry/Exit gate boxes, and horizontal tracking line proportionally!
"""

import argparse
import os
import re
import sys
import cv2
import numpy as np

import config
from core.camera_stream import CameraStream


def compute_proportional_layout(x1, y1, x2, y2):
    """
    Given outer parking boundary [x1, y1, x2, y2], calculate proportional coordinates for:
    - 12 Parking Slots (A1..A3, B1..B3, C1..C3, D1..D3)
    - Entry & Exit Gate Boxes (Ô Vào ↓, Ô Ra ↑)
    - Crossing Line Y
    """
    w = max(10, x2 - x1)
    h = max(10, y2 - y1)

    crossing_y = y1

    # Vertical distribution
    top_slots_y1 = y1
    top_slots_y2 = int(y1 + 0.3 * h)

    bot_slots_y1 = int(y2 - 0.3 * h)
    bot_slots_y2 = y2

    # Horizontal distribution
    left_x1 = x1
    left_x2 = int(x1 + 0.36 * w)

    right_x1 = int(x2 - 0.36 * w)
    right_x2 = x2

    mid_x1 = int(x1 + 0.36 * w)
    mid_x2 = int(x2 - 0.36 * w)
    mid_w = mid_x2 - mid_x1

    # Entry (↓) and Exit (↑) lane boxes in the top middle
    entry_box = [mid_x1, y1 - int(0.22 * h), mid_x1 + int(mid_w * 0.48), y1]
    exit_box = [mid_x1 + int(mid_w * 0.52), y1 - int(0.22 * h), mid_x2, y1]

    # Slot widths
    l_slot_w = (left_x2 - left_x1) / 3.0
    r_slot_w = (right_x2 - right_x1) / 3.0

    zones = {
        "A": {
            "name": "Khu A (Trái Trên)",
            "slots": {
                "A1": [[int(left_x1), top_slots_y1], [int(left_x1 + l_slot_w), top_slots_y1], [int(left_x1 + l_slot_w), top_slots_y2], [int(left_x1), top_slots_y2]],
                "A2": [[int(left_x1 + l_slot_w), top_slots_y1], [int(left_x1 + 2*l_slot_w), top_slots_y1], [int(left_x1 + 2*l_slot_w), top_slots_y2], [int(left_x1 + l_slot_w), top_slots_y2]],
                "A3": [[int(left_x1 + 2*l_slot_w), top_slots_y1], [left_x2, top_slots_y1], [left_x2, top_slots_y2], [int(left_x1 + 2*l_slot_w), top_slots_y2]],
            },
        },
        "B": {
            "name": "Khu B (Phải Trên)",
            "slots": {
                "B1": [[right_x1, top_slots_y1], [int(right_x1 + r_slot_w), top_slots_y1], [int(right_x1 + r_slot_w), top_slots_y2], [right_x1, top_slots_y2]],
                "B2": [[int(right_x1 + r_slot_w), top_slots_y1], [int(right_x1 + 2*r_slot_w), top_slots_y1], [int(right_x1 + 2*r_slot_w), top_slots_y2], [int(right_x1 + r_slot_w), top_slots_y2]],
                "B3": [[int(right_x1 + 2*r_slot_w), top_slots_y1], [right_x2, top_slots_y1], [right_x2, top_slots_y2], [int(right_x1 + 2*r_slot_w), top_slots_y2]],
            },
        },
        "C": {
            "name": "Khu C (Trái Dưới)",
            "slots": {
                "C1": [[int(left_x1), bot_slots_y1], [int(left_x1 + l_slot_w), bot_slots_y1], [int(left_x1 + l_slot_w), bot_slots_y2], [int(left_x1), bot_slots_y2]],
                "C2": [[int(left_x1 + l_slot_w), bot_slots_y1], [int(left_x1 + 2*l_slot_w), bot_slots_y1], [int(left_x1 + 2*l_slot_w), bot_slots_y2], [int(left_x1 + l_slot_w), bot_slots_y2]],
                "C3": [[int(left_x1 + 2*l_slot_w), bot_slots_y1], [left_x2, bot_slots_y1], [left_x2, bot_slots_y2], [int(left_x1 + 2*l_slot_w), bot_slots_y2]],
            },
        },
        "D": {
            "name": "Khu D (Phải Dưới)",
            "slots": {
                "D1": [[right_x1, bot_slots_y1], [int(right_x1 + r_slot_w), bot_slots_y1], [int(right_x1 + r_slot_w), bot_slots_y2], [right_x1, bot_slots_y2]],
                "D2": [[int(right_x1 + r_slot_w), bot_slots_y1], [int(right_x1 + 2*r_slot_w), bot_slots_y1], [int(right_x1 + 2*r_slot_w), bot_slots_y2], [int(right_x1 + r_slot_w), bot_slots_y2]],
                "D3": [[int(right_x1 + 2*r_slot_w), bot_slots_y1], [right_x2, bot_slots_y1], [right_x2, bot_slots_y2], [int(right_x1 + 2*r_slot_w), bot_slots_y2]],
            },
        },
    }

    return zones, crossing_y


def save_to_config_file(zones, crossing_y):
    """Update config.py directly with the newly calculated proportional layout."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.py")
    if not os.path.exists(cfg_path):
        print(f"[Error] config.py not found at {cfg_path}")
        return False

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Format PARKING_ZONES string
    zones_str = "PARKING_ZONES = {\n"
    for z_id, z_data in zones.items():
        zones_str += f'    "{z_id}": {{\n'
        zones_str += f'        "name": "{z_data["name"]}",\n'
        zones_str += '        "slots": {\n'
        for s_id, pts in z_data["slots"].items():
            zones_str += f'            "{s_id}": {pts},\n'
        zones_str += "        },\n    },\n"
    zones_str += "}"

    # Replace CROSSING_LINE_Y
    content = re.sub(r"CROSSING_LINE_Y\s*=\s*\d+", f"CROSSING_LINE_Y = {crossing_y}", content)
    # Replace PARKING_ZONES up to Legacy ROI comment
    content = re.sub(r"PARKING_ZONES\s*=\s*\{.*?\n\}(?=\n\n# Legacy)", zones_str, content, flags=re.DOTALL)
    if "PARKING_ZONES = {" not in content:
        content = re.sub(r"PARKING_ZONES\s*=\s*\{.*?\n\}", zones_str, content, flags=re.DOTALL)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[SUCCESS] Updated config.py with new proportional parking grid layout!")
    print(f"  - CROSSING_LINE_Y = {crossing_y}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Proportional Parking Grid Selector & Auto-Calibrator")
    parser.add_argument("-s", "--source", default="", help="Camera index or image/video file path")
    args = parser.parse_args()

    cam_src = args.source if args.source else config.CAMERA_SOURCE

    camera = None
    frame = None
    if isinstance(cam_src, str) and (cam_src.endswith(".jpg") or cam_src.endswith(".png")):
        frame = cv2.imread(cam_src)
    else:
        print(f"[Init] Đang mở camera stream liên tục từ nguồn: {cam_src}...")
        camera = CameraStream(
            source=cam_src,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
            use_picamera2=config.USE_PICAMERA2,
        )
        if not camera.is_opened():
            print(f"[Error] Không thể mở camera source {cam_src}")
            return

        # Fetch initial frame for resolution
        for _ in range(20):
            ret, frame = camera.read()
            if ret and frame is not None:
                break
            time.sleep(0.02)

    if frame is None and camera is None:
        print("[Error] Failed to capture frame.")
        return

    h, w = (480, 640) if frame is None else frame.shape[:2]

    # Default outer boundary: x1, y1, x2, y2
    outer_box = [5, 110, 635, 475]
    dragging = False
    drag_start = (0, 0)

    print("\n===============================================================")
    print("   CÔNG CỤ TỰ ĐỘNG CHIA TỈ LỆ KHUÔN TRACKING & 12 Ô ĐẬU A1..D3  ")
    print("===============================================================")
    print("  - Stream camera LIÊN TỤC (Live Stream) giúp căn chỉnh góc camera")
    print("  - KÉO CHUỘT TRÁI : Kéo chọn Vùng Khung Bãi Đỗ Xe mới (Tracking Box)")
    print("  - Phím [S]       : TỰ ĐỘNG LƯU cấu hình 12 ô A1..D3 chuẩn tỉ lệ vào config.py")
    print("  - Phím [R]       : Khôi phục khung bãi đỗ mặc định")
    print("  - Phím [Q]       : Thoát công cụ\n")

    def mouse_callback(event, x, y, flags, param):
        nonlocal outer_box, dragging, drag_start
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
            outer_box = [x, y, x + 10, y + 10]
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            outer_box[2] = max(drag_start[0] + 10, x)
            outer_box[3] = max(drag_start[1] + 10, y)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            outer_box[2] = max(drag_start[0] + 10, x)
            outer_box[3] = max(drag_start[1] + 10, y)

    cv2.namedWindow("Proportional Parking Grid Calibrator", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Proportional Parking Grid Calibrator", mouse_callback)

    try:
        while True:
            if camera is not None:
                ret, live_frame = camera.read()
                if not ret or live_frame is None:
                    time.sleep(0.01)
                    continue
                frame = live_frame

            display = frame.copy()
            x1, y1, x2, y2 = outer_box

            # Compute proportional grid
            zones, crossing_y = compute_proportional_layout(x1, y1, x2, y2)

            # 1. Draw Outer Tracking Boundary Box (Vạch Cyan)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(display, (0, crossing_y), (w, crossing_y), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"VẠCH TRACKING Y={crossing_y}", (10, max(20, crossing_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

            # 2. Draw 12 Parking Slots (A1..D3)
            for z_id, z_data in zones.items():
                for s_id, pts in z_data["slots"].items():
                    poly = np.array(pts, dtype=np.int32)
                    cv2.polylines(display, [poly], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
                    cv2.putText(display, s_id, (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # HUD Instructions
            cv2.rectangle(display, (0, h - 35), (w, h), (20, 20, 20), -1)
            cv2.putText(display, "CAMERA TRỰC TIẾP | KÉO CHUỘT ĐỔI KHUNG | [S] LƯU CONFIG | [Q] THOÁT", (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Proportional Parking Grid Calibrator", display)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord("s"), ord("S")):
                save_to_config_file(zones, crossing_y)

            elif key in (ord("r"), ord("R")):
                outer_box = [5, 110, 635, 475]
                print("[Calibrator] Reset to default boundary box.")

            elif key in (27, ord("q"), ord("Q")):
                break
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
