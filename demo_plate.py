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

    # 2. Live Camera Mode (Capture Full HD 1920x1080 on button press for Entry/Exit Boxes)
    cam_src = args.source if args.source else config.CAMERA_SOURCE
    camera = CameraStream(
        source=cam_src,
        width=getattr(config, "ALPR_CAPTURE_WIDTH", 1920),
        height=getattr(config, "ALPR_CAPTURE_HEIGHT", 1080),
        fps=config.CAMERA_FPS,
        use_picamera2=config.USE_PICAMERA2,
    )

    if not camera.is_opened():
        print(f"\n[ERROR] Could not open camera source '{cam_src}'.")
        print("Troubleshooting options:")
        print("  1. If using USB webcam, try index 1 or 2: python3 demo_plate.py --source 1")
        print("  2. Check connected video devices: v4l2-ctl --list-devices or ls -l /dev/video*")
        print("  3. If using RPi Camera Module (ribbon), set USE_PICAMERA2 = True in config.py")
        print("  4. Test a static image: python3 demo_plate.py --image test_images/1.jpg\n")
        camera.release()
        return

    print("\n=======================================================")
    print("  ALPR FULL HD CAPTURE: Ô VÀO (LẬT 180°) & Ô RA (GIỮ NGUYÊN)")
    print("=======================================================")
    print("  - [SPACE / C / ENTER] : Chụp đúng 2 ô ra vào để đọc biển số")
    print("  - [R]                 : Xem lại camera live stream")
    print("  - [Q]                 : Thoát chương trình\n")

    captured_result_frame = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Entry & Exit Box coordinates for current preview resolution
            s_x, s_y = w / 640.0, h / 480.0
            e_box_raw = getattr(config, "ENTRY_LANE_BOX", [238, 10, 315, 115])
            x_box_raw = getattr(config, "EXIT_LANE_BOX", [322, 10, 398, 115])

            en_x1, en_y1, en_x2, en_y2 = int(e_box_raw[0]*s_x), int(e_box_raw[1]*s_y), int(e_box_raw[2]*s_x), int(e_box_raw[3]*s_y)
            ex_x1, ex_y1, ex_x2, ex_y2 = int(x_box_raw[0]*s_x), int(x_box_raw[1]*s_y), int(x_box_raw[2]*s_x), int(x_box_raw[3]*s_y)

            # Show captured result or live preview
            if captured_result_frame is not None:
                display_frame = captured_result_frame
            else:
                display_frame = frame.copy()

                # Draw Entry Box (Ô Vào ↓ - Left)
                cv2.rectangle(display_frame, (en_x1, en_y1), (en_x2, en_y2), (0, 255, 255), 2)
                cv2.putText(display_frame, "Ô VÀO (v)", (en_x1, max(12, en_y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                # Draw Exit Box (Ô Ra ↑ - Right)
                cv2.rectangle(display_frame, (ex_x1, ex_y1), (ex_x2, ex_y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "Ô RA (^)", (ex_x1, max(12, ex_y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                # Preview HUD Banner
                cv2.rectangle(display_frame, (0, h - 35), (w, h), (20, 20, 20), -1)
                cv2.putText(
                    display_frame,
                    "[SPACE / C]: CHỤP 2 Ô RA VÀO (Ô VÀO LẬT 180° | Ô RA GIỮ NGUYÊN) | [Q]: THOÁT",
                    (15, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            key = -1
            if config.ENABLE_GUI:
                cv2.imshow("ALPR Entry/Exit Capture Demo (RPi 5)", display_frame)
                key = cv2.waitKey(10) & 0xFF
            else:
                time.sleep(0.03)

            if key == ord("q") or key == ord("Q"):
                break

            # Trigger capture & ALPR detection on key press
            if key in (32, ord("c"), ord("C"), ord("s"), ord("S"), 13):
                print("\n[CAP] Đang chụp đúng 2 ô ra vào để xử lý đọc biển số...")

                # Ensure resolution is Full HD (1920x1080) for maximum ALPR precision
                if (w, h) != (1920, 1080):
                    hd_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                else:
                    hd_frame = frame.copy()

                hd_h, hd_w = hd_frame.shape[:2]

                # Full HD scale for Entry & Exit boxes
                hs_x, hs_y = hd_w / 640.0, hd_h / 480.0
                hen_x1, hen_y1 = max(0, int(e_box_raw[0]*hs_x)), max(0, int(e_box_raw[1]*hs_y))
                hen_x2, hen_y2 = min(hd_w, int(e_box_raw[2]*hs_x)), min(hd_h, int(e_box_raw[3]*hs_y))

                hex_x1, hex_y1 = max(0, int(x_box_raw[0]*hs_x)), max(0, int(x_box_raw[1]*hs_y))
                hex_x2, hex_y2 = min(hd_w, int(x_box_raw[2]*hs_x)), min(hd_h, int(x_box_raw[3]*hs_y))

                # 1. CẮT Ô VÀO (Entry Lane ↓): LẬT NGƯỢC HÌNH 180°
                crop_entry_raw = hd_frame[hen_y1:hen_y2, hen_x1:hen_x2].copy()
                crop_entry = cv2.flip(crop_entry_raw, -1) if crop_entry_raw.size > 0 else crop_entry_raw

                # 2. CẮT Ô RA (Exit Lane ↑): GIỮ NGUYÊN HÌNH (KHÔNG LẬT)
                crop_exit = hd_frame[hex_y1:hex_y2, hex_x1:hex_x2].copy()

                t0 = time.time()
                # Run ALPR Detection on Entry Crop (Flipped first, then raw)
                det_entry = engine.detect_and_read(crop_entry) if crop_entry.size > 0 else []
                if not det_entry and crop_entry_raw.size > 0:
                    det_entry = engine.detect_and_read(crop_entry_raw)
                if not det_entry and crop_entry.size > 0:
                    # Direct OCR fallback on entry crop
                    ocr_txt = engine.recognize_crop(crop_entry)
                    if ocr_txt != "unknown":
                        det_entry = [{"bbox": (5, 5, crop_entry.shape[1] - 5, crop_entry.shape[0] - 5), "plate": ocr_txt, "conf": 0.75, "crop": crop_entry}]

                # Run ALPR Detection on Exit Crop (Normal first, then flipped)
                det_exit = engine.detect_and_read(crop_exit) if crop_exit.size > 0 else []
                if not det_exit and crop_exit.size > 0:
                    crop_exit_flip = cv2.flip(crop_exit, -1)
                    det_exit = engine.detect_and_read(crop_exit_flip)
                if not det_exit and crop_exit.size > 0:
                    # Direct OCR fallback on exit crop
                    ocr_txt = engine.recognize_crop(crop_exit)
                    if ocr_txt != "unknown":
                        det_exit = [{"bbox": (5, 5, crop_exit.shape[1] - 5, crop_exit.shape[0] - 5), "plate": ocr_txt, "conf": 0.75, "crop": crop_exit}]

                dt = (time.time() - t0) * 1000.0

                # Draw plates on cropped views
                if crop_entry.size > 0:
                    engine.draw_plates(crop_entry, det_entry)
                if crop_exit.size > 0:
                    engine.draw_plates(crop_exit, det_exit)

                # Draw boxes on main HD frame
                cv2.rectangle(hd_frame, (hen_x1, hen_y1), (hen_x2, hen_y2), (0, 255, 255), 3)
                cv2.putText(hd_frame, "Ô VÀO (v) [LẬT 180°]", (hen_x1, max(25, hen_y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(hd_frame, (hex_x1, hex_y1), (hex_x2, hex_y2), (0, 255, 0), 3)
                cv2.putText(hd_frame, "Ô RA (^) [GIỮ NGUYÊN]", (hex_x1, max(25, hex_y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)

                # -------------------------------------------------------------
                # Panel Inset Left: Ô VÀO (LẬT 180°)
                # -------------------------------------------------------------
                panel_w, panel_h = 440, 360
                if crop_entry.size > 0:
                    zoom_entry = cv2.resize(crop_entry, (panel_w, panel_h))
                    cv2.rectangle(zoom_entry, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 255), 3)
                    cv2.rectangle(zoom_entry, (0, 0), (panel_w, 32), (20, 20, 20), -1)

                    plate_in_txt = det_entry[0]["plate"] if det_entry else "KHÔNG THẤY"
                    cv2.putText(
                        zoom_entry,
                        f"Ô VÀO (LẬT 180°): {plate_in_txt}",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    # Overlay Top-Left
                    hd_frame[60 : 60 + panel_h, 20 : 20 + panel_w] = zoom_entry

                # -------------------------------------------------------------
                # Panel Inset Right: Ô RA (GIỮ NGUYÊN)
                # -------------------------------------------------------------
                if crop_exit.size > 0:
                    zoom_exit = cv2.resize(crop_exit, (panel_w, panel_h))
                    cv2.rectangle(zoom_exit, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 0), 3)
                    cv2.rectangle(zoom_exit, (0, 0), (panel_w, 32), (20, 20, 20), -1)

                    plate_out_txt = det_exit[0]["plate"] if det_exit else "KHÔNG THẤY"
                    cv2.putText(
                        zoom_exit,
                        f"Ô RA (GIỮ NGUYÊN): {plate_out_txt}",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    # Overlay Top-Right
                    hd_frame[60 : 60 + panel_h, hd_w - panel_w - 20 : hd_w - 20] = zoom_exit

                # Summary Text
                p_in = det_entry[0]["plate"] if det_entry else "No"
                p_out = det_exit[0]["plate"] if det_exit else "No"
                summary_msg = f"ĐÃ CHỤP 2 Ô | Xe Vào (Lật 180°): {p_in} | Xe Ra (Giữ Nguyên): {p_out} | ({dt:.1f}ms)"

                cv2.rectangle(hd_frame, (0, 0), (hd_w, 50), (15, 15, 15), -1)
                cv2.putText(
                    hd_frame,
                    summary_msg,
                    (20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.rectangle(hd_frame, (0, hd_h - 40), (hd_w, hd_h), (20, 20, 20), -1)
                cv2.putText(
                    hd_frame,
                    "ĐÃ CẮT & ĐỌC BIỂN SỐ 2 Ô RA VÀO! [SPACE/C] Chụp tiếp | [R] Live camera | [Q] Thoát",
                    (20, hd_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                captured_result_frame = hd_frame
                print(f"[ALPR SUCCESS] {summary_msg}")
                if det_entry:
                    print(f" -> Ô Vào (Lật 180°): Biển số '{det_entry[0]['plate']}' (conf={det_entry[0]['conf']:.2f})")
                if det_exit:
                    print(f" -> Ô Ra (Giữ Nguyên): Biển số '{det_exit[0]['plate']}' (conf={det_exit[0]['conf']:.2f})")

            # Resume live camera stream on key 'R'
            if key in (ord("r"), ord("R")):
                captured_result_frame = None
                print("[LIVE] Trở lại camera live stream.")

    finally:
        camera.release()
        if config.ENABLE_GUI:
            cv2.destroyAllWindows()
        print("[ALPR Demo] Exited.")


if __name__ == "__main__":
    main()
