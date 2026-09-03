"""
=============================================================================
Smart Parking Management & Vehicle Verification System (Raspberry Pi 5 Edition)
Integrates License Plate Recognition (ALPR) & Driver Face Recognition.
=============================================================================
"""

import argparse
import sys
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

import config
from database import ParkingDatabase
from core.camera_stream import CameraStream
from core.face_engine import FaceEngine
from core.plate_engine import PlateEngine


class SmartParkingGate:
    def __init__(self, headless: bool = False):
        self.headless = headless or not config.ENABLE_GUI
        print("\n=======================================================")
        print("  SMART PARKING MANAGEMENT SYSTEM (Raspberry Pi 5)   ")
        print("=======================================================")

        print("[Init] Loading Database...")
        self.db = ParkingDatabase(config.DB_PATH)

        print("[Init] Initializing Face Engine (InsightFace buffalo_sc)...")
        self.face_engine = FaceEngine(
            model_name=config.FACE_MODEL_NAME,
            det_thresh=config.FACE_DET_THRESHOLD,
            det_size=config.FACE_DET_SIZE,
            min_face_size=config.FACE_MIN_SIZE,
            num_threads=config.ONNX_NUM_THREADS,
        )

        print("[Init] Initializing Plate Engine (YOLOv5 Nano)...")
        self.plate_engine = PlateEngine(
            detector_path=str(config.PLATE_DETECTOR_MODEL),
            ocr_path=str(config.PLATE_OCR_MODEL),
            yolov5_dir=str(config.YOLOV5_DIR),
            det_conf=config.PLATE_DET_CONF,
            ocr_conf=config.PLATE_OCR_CONF,
            img_size=config.PLATE_IMG_SIZE,
            device="cpu",
        )

        print("[Init] Starting Camera Stream...")
        self.camera = CameraStream(
            source=config.CAMERA_SOURCE,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
            use_picamera2=config.USE_PICAMERA2,
        )

        if not self.camera.is_opened():
            print(f"\n[ERROR] Could not open camera source '{config.CAMERA_SOURCE}'.")
            print("Troubleshooting options:")
            print("  1. Change CAMERA_SOURCE in config.py to 1 or 2 (for USB webcams)")
            print("  2. Check connected devices: v4l2-ctl --list-devices or ls -l /dev/video*")
            print("  3. If using RPi Camera Module (ribbon), set USE_PICAMERA2 = True in config.py\n")

        self.last_status_msg = "SYSTEM READY"
        self.last_status_color = (0, 255, 0)
        self.last_transaction_info = "No recent transactions"
        self.barrier_status = "BARRIER: CLOSED"

    def handle_checkin(self, frame: np.ndarray, detected_plates: List[Dict], faces: List) -> None:
        """Process vehicle arrival at the entry gate."""
        print("\n--- [GATE IN / CHECK-IN] XỬ LÝ XE VÀO ---")

        # 1. Check plate
        plate_str = "UNKNOWN"
        for p in detected_plates:
            if p["plate"] != "unknown":
                plate_str = p["plate"]
                break

        # 2. Check face
        primary_face = self.face_engine.choose_primary_face(faces)
        embedding = None
        if primary_face is not None:
            embedding = self.face_engine.get_embedding(primary_face)
        else:
            print("[Warning] Không phát hiện được khuôn mặt rõ ràng của tài xế.")

        # If plate is unknown, allow console input or auto-generate
        if plate_str == "UNKNOWN":
            print(f"Không nhận diện được biển số tự động.")
            if not self.headless:
                manual_plate = input("Nhập biển số xe thủ công (hoặc Enter để bỏ qua): ").strip().upper()
                if manual_plate:
                    plate_str = manual_plate

        # Register in database
        driver_name = ""
        session_id = self.db.register(driver_name, plate_str, embedding)

        self.last_status_msg = f"CHECK-IN OK: ID #{session_id} | Plate: {plate_str}"
        self.last_status_color = (0, 255, 0)
        self.barrier_status = "BARRIER: OPEN (CHECK-IN)"
        self.last_transaction_info = f"IN: #{session_id} | Plate: {plate_str} | {datetime.now().strftime('%H:%M:%S')}"

        print(f"[SUCCESS] {self.last_status_msg}")
        print(f"[GATE] Cổng barrier MỞ cho xe #{session_id} vào bãi.")

    def handle_checkout(self, frame: np.ndarray, detected_plates: List[Dict], faces: List) -> None:
        """Process vehicle departure at the exit gate with dual plate + face verification."""
        print("\n--- [GATE OUT / CHECK-OUT] XỬ LÝ XE RA & ĐỐI SOÁT ---")

        # 1. Get plate
        plate_str = "UNKNOWN"
        for p in detected_plates:
            if p["plate"] != "unknown":
                plate_str = p["plate"]
                break

        primary_face = self.face_engine.choose_primary_face(faces)
        exit_emb = None
        if primary_face is not None:
            exit_emb = self.face_engine.get_embedding(primary_face)

        # 2. Search database session
        session = None
        if plate_str != "UNKNOWN":
            session = self.db.find_active_by_plate(plate_str)

        # If not found by plate, try searching by face embedding
        if session is None and exit_emb is not None:
            print("[Info] Không tìm thấy theo biển số, tiến hành quét theo nhận diện khuôn mặt...")
            session = self.db.find_best_face_active(exit_emb)

        if session is None:
            # Fallback manual plate
            if not self.headless:
                manual = input("Không tìm thấy phiên. Nhập biển số thủ công: ").strip().upper()
                if manual:
                    session = self.db.find_active_by_plate(manual)

        if session is None:
            self.last_status_msg = "CHECK-OUT FAIL: Không tìm thấy phiên gửi xe phù hợp!"
            self.last_status_color = (0, 0, 255)
            self.barrier_status = "BARRIER: LOCKED"
            print(f"[ERROR] {self.last_status_msg}")
            return

        session_id = session["id"]
        saved_plate = session["plate"]
        saved_emb = session["embedding"]

        # 3. Dual Face Verification
        sim_score = 0.0
        face_match = False

        if saved_emb is not None and exit_emb is not None:
            sim_score = self.db.compare_embeddings(saved_emb, exit_emb)
            face_match = sim_score >= config.FACE_MATCH_THRESHOLD
        elif saved_emb is None:
            # Face was not registered at check-in
            face_match = True
            sim_score = 1.0

        if face_match:
            res = self.db.close_session(session_id, match_score=sim_score)
            self.last_status_msg = f"CHECK-OUT OK: Plate {saved_plate} (Sim: {sim_score:.2f}) | {res['duration_minutes']} mins"
            self.last_status_color = (0, 255, 0)
            self.barrier_status = "BARRIER: OPEN (CHECK-OUT)"
            self.last_transaction_info = f"OUT: #{session_id} | {saved_plate} | {res['duration_minutes']}m | Sim: {sim_score:.2f}"

            print("\n[SUCCESS] XÁC THỰC HỢP LỆ! Xe được phép ra bãi.")
            print(f"-> Biển số: {saved_plate}")
            print(f"-> Độ khớp khuôn mặt tài xế: {sim_score:.3f} (Ngưỡng: {config.FACE_MATCH_THRESHOLD})")
            print(f"-> Thời gian gửi: {res['time_in']} -> {res['time_out']} ({res['duration_minutes']} phút)")
            print("[GATE] Cổng barrier MỞ cho xe ra.")
        else:
            self.last_status_msg = f"CẢNH BÁO: SAI KHUÔN MẶT! Sim={sim_score:.2f} (< {config.FACE_MATCH_THRESHOLD})"
            self.last_status_color = (0, 0, 255)
            self.barrier_status = "BARRIER: LOCKED (ALERT)"
            self.last_transaction_info = f"ALERT: #{session_id} Plate {saved_plate} | Face mismatch ({sim_score:.2f})"

            print("\n[CẢNH BÁO GIAN LẬN / SAI TÀI XẾ!]")
            print(f"-> Biển số: {saved_plate}")
            print(f"-> Khuôn mặt hiện tại không trùng khớp với người gửi lúc vào (Sim: {sim_score:.3f})")
            print("[GATE] Cổng barrier KHÓA chặt. Yêu cầu kiểm tra an ninh!")

    def draw_dashboard(
        self,
        frame: np.ndarray,
        fps: float,
        detected_plates: List[Dict],
        faces: List,
    ) -> np.ndarray:
        """Render a modern smart parking HUD overlay."""
        h, w = frame.shape[:2]

        # Top banner
        header_overlay = frame.copy()
        cv2.rectangle(header_overlay, (0, 0), (w, 100), (25, 25, 25), -1)
        cv2.addWeighted(header_overlay, 0.75, frame, 0.25, 0, frame)

        # Title & Stats
        cv2.putText(
            frame,
            f"SMART PARKING RPi 5 | FPS: {fps:.1f} | Active Cars: {self.db.active_count()}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Status line
        cv2.putText(
            frame,
            f"Status: {self.last_status_msg}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            self.last_status_color,
            2,
            cv2.LINE_AA,
        )

        # Sub-status & barrier
        cv2.putText(
            frame,
            f"{self.barrier_status} | {self.last_transaction_info}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Bottom Controls bar
        footer_overlay = frame.copy()
        cv2.rectangle(footer_overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(footer_overlay, 0.8, frame, 0.2, 0, frame)

        cv2.putText(
            frame,
            "[I / R]: Check-in (Vào) | [O / X]: Check-out (Ra) | [Q]: Thoát",
            (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw Plate & Face annotations
        self.plate_engine.draw_plates(frame, detected_plates)
        self.face_engine.draw_faces(frame, faces)

        return frame

    def run(self) -> None:
        """Main system loop."""
        print("\n>>> System running. Press 'I' for Check-in, 'O' for Check-out, 'Q' to quit. <<<")
        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                # Run ALPR & Face Detection in parallel/sequence
                detected_plates = self.plate_engine.detect_and_read(frame)
                faces = self.face_engine.analyze(frame)

                # Render Dashboard
                display_frame = self.draw_dashboard(frame, fps, detected_plates, faces)

                key = -1
                if not self.headless:
                    cv2.imshow("Smart Parking Gate System (Raspberry Pi 5)", display_frame)
                    key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                # Check-in (Key 'I' or 'R')
                if key in (ord("i"), ord("r")):
                    self.handle_checkin(frame, detected_plates, faces)

                # Check-out (Key 'o' or 'x')
                if key in (ord("o"), ord("x")):
                    self.handle_checkout(frame, detected_plates, faces)

        finally:
            self.camera.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.db.close()
            print("\n[System] Shutdown completed gracefully.")


def main():
    parser = argparse.ArgumentParser(description="Smart Parking Management System on Raspberry Pi 5")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    args = parser.parse_args()

    app = SmartParkingGate(headless=args.headless)
    app.run()


if __name__ == "__main__":
    main()
