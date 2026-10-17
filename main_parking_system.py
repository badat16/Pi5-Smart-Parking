"""
=============================================================================
Smart Parking Management & Verification System (Raspberry Pi 5 Edition)
Dual-Window Multi-Camera Architecture:
  - Window 1 (Camera 0): Real-Time Vehicle Tracking & Slot Occupancy (30 FPS)
  - Window 2 (Camera 1): Entry/Exit Gate Camera & Face Query (Throttled 10 FPS)
=============================================================================
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# Suppress PyTorch and InsightFace deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

import config
from database import ParkingDatabase
from core.camera_stream import CameraStream
from core.face_engine import FaceEngine
from core.plate_engine import PlateEngine
from core.vehicle_tracker import VehicleTracker


class DualCameraSmartParking:
    def __init__(self, cam_tracking_src=None, cam_gate_src=None, headless: bool = False):
        self.headless = headless or not config.ENABLE_GUI

        print("\n=======================================================================")
        print("  SMART PARKING SYSTEM - DUAL CAMERA & TWO WINDOW MODE (Raspberry Pi 5) ")
        print("=======================================================================")

        # 1. Initialize Database
        print("[Init] Loading Database...")
        self.db = ParkingDatabase(config.DB_PATH)

        # 2. Initialize Vehicle Tracker for Camera 0 (Overhead Tracking)
        print("[Init] Initializing Vehicle Tracker (YOLO ONNX)...")
        self.tracker = VehicleTracker(
            model_path=str(config.CAR_TRACKING_MODEL),
            conf_thresh=config.CAR_CONF_THRESHOLD,
            iou_thresh=config.CAR_IOU_THRESHOLD,
            track_score=config.TRACK_SCORE_THRESHOLD,
            max_missed=config.TRACK_MAX_MISSED,
            imgsz=config.CAR_TRACKING_IMGSIZE,
        )

        # 3. Initialize Face Engine & Plate Engine for Camera 1 (Gate)
        print("[Init] Initializing Face Engine (InsightFace buffalo_sc)...")
        self.face_engine = FaceEngine(
            model_name=config.FACE_MODEL_NAME,
            det_thresh=config.FACE_DET_THRESHOLD,
            det_size=config.FACE_DET_SIZE,
            min_face_size=config.FACE_MIN_SIZE,
            num_threads=config.ONNX_NUM_THREADS,
        )

        print("[Init] Initializing Plate Engine (YOLOv5 LP)...")
        self.plate_engine = PlateEngine(
            detector_path=str(config.PLATE_DETECTOR_MODEL),
            ocr_path=str(config.PLATE_OCR_MODEL),
            yolov5_dir=str(config.YOLOV5_DIR),
            det_conf=config.PLATE_DET_CONF,
            ocr_conf=config.PLATE_OCR_CONF,
            img_size=config.PLATE_IMG_SIZE,
            device="cpu",
        )

        # 4. Initialize Camera 0 (Tracking Camera - Full 30 FPS)
        track_src = cam_tracking_src if cam_tracking_src is not None else getattr(config, "TRACKING_CAMERA_SOURCE", 0)
        track_fps = getattr(config, "TRACKING_CAMERA_FPS", 30)
        print(f"[Init] Starting Tracking Camera (Camera {track_src} at {track_fps} FPS)...")
        self.cam_tracking = CameraStream(
            source=track_src,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=track_fps,
            use_picamera2=config.USE_PICAMERA2,
        )

        # 5. Initialize Camera 1 (Gate Camera - Throttled 10 FPS)
        gate_src = cam_gate_src if cam_gate_src is not None else getattr(config, "GATE_CAMERA_SOURCE", 1)
        gate_fps = getattr(config, "GATE_CAMERA_FPS", 10)
        print(f"[Init] Starting Gate Camera (Camera {gate_src} at {gate_fps} FPS)...")
        self.cam_gate = CameraStream(
            source=gate_src,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=gate_fps,
            use_picamera2=False,
        )

        # Fallback if Gate Camera is not plugged in
        if not self.cam_gate.is_opened():
            print(f"[Warning] Gate Camera ({gate_src}) not found. Falling back to Tracking Camera for Gate view.")
            self.cam_gate = self.cam_tracking

        cam0_src = getattr(self.cam_tracking, "actual_source", track_src)
        cam1_src = getattr(self.cam_gate, "actual_source", gate_src)
        print(f"[Init] Tracking Camera running on /dev/video{cam0_src}")
        print(f"[Init] Gate Camera running on /dev/video{cam1_src}")

        if cam0_src == cam1_src:
            print(f"\n[INFO] Both windows are currently sharing camera device /dev/video{cam0_src}.")
            print("  To use 2 separate physical webcams, pass your second webcam index:")
            print("  Example: python3 main_parking_system.py --cam-tracking 0 --cam-gate 2 (or 4)\n")

        # System Status state
        self.last_status_msg = "SYSTEM READY - DUAL CAMERA ACTIVE"
        self.last_status_color = (0, 255, 0)
        self.last_transaction_info = "No recent transactions"
        self.barrier_status = "BARRIER: CLOSED"
        self.last_gate_result_overlay = None

        # Tracking Session binding state
        self.pending_sessions: List[Dict[str, Any]] = []
        self.track_session_map: Dict[int, Dict[str, Any]] = {}
        self.current_occupancy: Dict[str, Dict[str, Any]] = {}

        # Gate window FPS rate limiter (10 FPS)
        self.last_gate_frame_time = 0.0
        self.cached_gate_display = None

    def handle_checkin(self, gate_frame: np.ndarray) -> None:
        """Execute Face + LP detection ON-DEMAND when Check-in button is pressed."""
        print("\n--- [GATE IN / CHECK-IN] EXECUTE FACE + LP RECOGNITION ---")

        t0 = time.time()
        detected_plates = self.plate_engine.detect_and_read(gate_frame)
        faces = self.face_engine.analyze(gate_frame)
        dt = (time.time() - t0) * 1000.0

        # Face Requirement Check
        primary_face = self.face_engine.choose_primary_face(faces)
        embedding = None
        if primary_face is not None:
            embedding = self.face_engine.get_embedding(primary_face)

        if embedding is None:
            self.last_status_msg = "CHECK-IN THAT BAI: CHUA NHAN DIEN DUOC KHUON MAT! VUI LONG THU LAI [I]"
            self.last_status_color = (0, 0, 255)
            self.barrier_status = "BARRIER: LOCKED (CANH BAO)"
            print("\n[ERROR] Check-in That bai: Khong tim thay khuon mat tai xe rõ rang.")
            print("[GUIDE] Vui long nhin thang vao Camera cong va an [I] de thu lai.")

            # Snapshot error overlay
            res_frame = gate_frame.copy()
            self.plate_engine.draw_plates(res_frame, detected_plates)
            self.face_engine.draw_faces(res_frame, faces)
            self.last_gate_result_overlay = res_frame
            return

        # Extract plate string
        plate_str = "UNKNOWN"
        for p in detected_plates:
            if p["plate"] != "unknown":
                plate_str = p["plate"]
                break

        if plate_str == "UNKNOWN" and not self.headless:
            manual_plate = input("Nhap bien so xe thu cong (Enter de bo qua): ").strip().upper()
            if manual_plate:
                plate_str = manual_plate

        # Register session in database
        driver_name = ""
        session_id = self.db.register(driver_name, plate_str, embedding)

        session_data = {
            "id": session_id,
            "plate": plate_str,
            "embedding": embedding,
            "time_in": datetime.now().strftime("%H:%M:%S"),
        }
        self.pending_sessions.append(session_data)

        self.last_status_msg = f"CHECK-IN OK: ID #{session_id} | Plate: {plate_str} ({dt:.0f}ms)"
        self.last_status_color = (0, 255, 0)
        self.barrier_status = "BARRIER: OPEN (CHECK-IN)"
        self.last_transaction_info = f"IN: #{session_id} | Plate: {plate_str} | {session_data['time_in']}"

        print(f"[SUCCESS] {self.last_status_msg}")
        print(f"[GATE] Cong barrier MO cho xe #{session_id} vao bai. (Pending auto-assign to next car track)")

        res_frame = gate_frame.copy()
        self.plate_engine.draw_plates(res_frame, detected_plates)
        self.face_engine.draw_faces(res_frame, faces)
        self.last_gate_result_overlay = res_frame

    def handle_checkout(self, gate_frame: np.ndarray) -> None:
        """Execute Dual Face + LP verification ON-DEMAND when Check-out button is pressed."""
        print("\n--- [GATE OUT / CHECK-OUT] EXECUTE DUAL FACE + LP VERIFICATION ---")

        t0 = time.time()
        detected_plates = self.plate_engine.detect_and_read(gate_frame)
        faces = self.face_engine.analyze(gate_frame)
        dt = (time.time() - t0) * 1000.0

        plate_str = "UNKNOWN"
        for p in detected_plates:
            if p["plate"] != "unknown":
                plate_str = p["plate"]
                break

        primary_face = self.face_engine.choose_primary_face(faces)
        exit_emb = None
        if primary_face is not None:
            exit_emb = self.face_engine.get_embedding(primary_face)

        session = None
        if plate_str != "UNKNOWN":
            session = self.db.find_active_by_plate(plate_str)

        if session is None and exit_emb is not None:
            print("[Info] Quet phien gui theo nhan dien khuon mat...")
            session = self.db.find_best_face_active(exit_emb)

        if session is None and not self.headless:
            manual = input("Khong tim thay phien. Nhap bien so thu cong: ").strip().upper()
            if manual:
                session = self.db.find_active_by_plate(manual)

        if session is None:
            self.last_status_msg = f"CHECK-OUT FAIL: Khong tim thay phien gui ({dt:.0f}ms)"
            self.last_status_color = (0, 0, 255)
            self.barrier_status = "BARRIER: LOCKED"
            print(f"[ERROR] {self.last_status_msg}")
            return

        session_id = session["id"]
        saved_plate = session["plate"]
        saved_emb = session.get("embedding", None)

        sim_score = 0.0
        face_match = False

        if saved_emb is not None and exit_emb is not None:
            sim_score = self.db.compare_embeddings(saved_emb, exit_emb)
            face_match = sim_score >= config.FACE_MATCH_THRESHOLD
        elif saved_emb is None:
            face_match = True
            sim_score = 1.0

        if face_match:
            res = self.db.close_session(session_id, match_score=sim_score)
            self.last_status_msg = f"CHECK-OUT OK: Plate {saved_plate} (Sim: {sim_score:.2f})"
            self.last_status_color = (0, 255, 0)
            self.barrier_status = "BARRIER: OPEN (CHECK-OUT)"
            self.last_transaction_info = f"OUT: #{session_id} | {saved_plate} | {res['duration_minutes']}m | Sim: {sim_score:.2f}"
            print(f"[SUCCESS] XAC THUC HOP LE! Xe ra bai. {self.last_status_msg}")

            # Clean up track mapping for this session
            keys_to_remove = [k for k, v in self.track_session_map.items() if v["id"] == session_id]
            for k in keys_to_remove:
                del self.track_session_map[k]
        else:
            self.last_status_msg = f"CANH BAO: SAI KHUON MAT! Sim={sim_score:.2f} (< {config.FACE_MATCH_THRESHOLD})"
            self.last_status_color = (0, 0, 255)
            self.barrier_status = "BARRIER: LOCKED (ALERT)"
            self.last_transaction_info = f"ALERT: #{session_id} Plate {saved_plate} | Face mismatch ({sim_score:.2f})"
            print(f"[ALERT] {self.last_status_msg}")

        res_frame = gate_frame.copy()
        self.plate_engine.draw_plates(res_frame, detected_plates)
        self.face_engine.draw_faces(res_frame, faces)
        self.last_gate_result_overlay = res_frame

    def handle_find_my_car(self, gate_frame: np.ndarray) -> None:
        """Scan driver face at gate camera and query which parking slot (A1..D3) their car is in."""
        print("\n--- [FIND MY CAR / QUET MAT TIM VI TRI XE] ---")
        t0 = time.time()
        faces = self.face_engine.analyze(gate_frame)
        primary_face = self.face_engine.choose_primary_face(faces)

        if primary_face is None:
            self.last_status_msg = "TIM XE THAT BAI: CHUA NHAN DIEN DUOC KHUON MAT! THU LAI [F]"
            self.last_status_color = (0, 0, 255)
            print("[ERROR] Tim xe that bai: Khong tim thay khuon mat tai xe.")
            return

        query_emb = self.face_engine.get_embedding(primary_face)
        session = self.db.find_best_face_active(query_emb)

        if session is None or session.get("score", 0.0) < config.FACE_MATCH_THRESHOLD:
            sim_str = f"{session.get('score', 0.0):.2f}" if session else "0.00"
            self.last_status_msg = f"TIM XE THAT BAI: KHONG TIM THAY XE TRONG BAI (Sim: {sim_str})"
            self.last_status_color = (0, 0, 255)
            print(f"[ERROR] Khong tim thay phien gui xe trung khop khuon mat nay (Sim: {sim_str}).")
            return

        session_id = session["id"]
        saved_plate = session["plate"]

        # Locate assigned slot
        current_slot = "DANG DI CHUYEN / CHUA VAO O"
        for tr_id, sess_info in self.track_session_map.items():
            if sess_info["id"] == session_id:
                for s_id, s_info in self.current_occupancy.items():
                    if s_info.get("vehicle_id") == tr_id:
                        current_slot = f"O DO {s_id} (Khu {s_info.get('zone_id', '')})"
                        break
                break

        self.last_status_msg = f"TIM XE: Bien so [{saved_plate}] -> Vi tri: [{current_slot}]"
        self.last_status_color = (0, 255, 255)
        print(f"[SUCCESS] TIM XE THANH CONG!")
        print(f"  -> Chu xe: #{session_id} | Bien so: {saved_plate}")
        print(f"  -> Vi tri hien tai trong bai: {current_slot}")

        res_frame = gate_frame.copy()
        self.face_engine.draw_faces(res_frame, faces)
        self.last_gate_result_overlay = res_frame

    def draw_gate_dashboard(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Render HUD overlay on Gate Camera (Camera 1)."""
        h, w = frame.shape[:2]
        display = frame.copy()

        # Top Banner
        cv2.rectangle(display, (0, 0), (w, 75), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"GATE CAMERA (CAM 1) | FPS: {fps:.1f} (10 FPS Target) | Active Cars: {self.db.active_count()}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Status: {self.last_status_msg}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self.last_status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"{self.barrier_status} | {self.last_transaction_info}",
            (12, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Bottom HUD Controls
        cv2.rectangle(display, (0, h - 30), (w, h), (15, 15, 15), -1)
        cv2.putText(
            display,
            "[I]: Check-in | [O]: Check-out | [F]: Tim Xe qua Mat | [Q]: Thoat",
            (12, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return display

    def draw_tracking_dashboard(self, frame: np.ndarray, fps: float, tracks: List, occupancy: Dict) -> np.ndarray:
        """Render HUD overlay on Tracking Camera (Camera 0)."""
        h, w = frame.shape[:2]
        display = self.tracker.draw_tracks(frame, tracks, session_map=self.track_session_map)

        # Calculate Zone Statistics
        zone_counts = {"A": [0, 0], "B": [0, 0], "C": [0, 0], "D": [0, 0]}
        total_occupied = 0

        for slot_id, slot_info in occupancy.items():
            z_id = slot_info["zone_id"]
            if z_id in zone_counts:
                zone_counts[z_id][1] += 1
                if slot_info["occupied"]:
                    zone_counts[z_id][0] += 1
                    total_occupied += 1

        total_slots = len(occupancy)
        total_vacant = max(0, total_slots - total_occupied)

        # Top Banner
        cv2.rectangle(display, (0, 0), (w, 36), (15, 15, 15), -1)
        pending_txt = f" | Pending: {len(self.pending_sessions)}" if self.pending_sessions else ""
        status_text = (
            f"KHU A: {zone_counts['A'][0]}/{zone_counts['A'][1]} | "
            f"KHU B: {zone_counts['B'][0]}/{zone_counts['B'][1]} | "
            f"KHU C: {zone_counts['C'][0]}/{zone_counts['C'][1]} | "
            f"KHU D: {zone_counts['D'][0]}/{zone_counts['D'][1]} | "
            f"DANG DO: {total_occupied}/{total_slots} | TRONG: {total_vacant}{pending_txt}"
        )
        cv2.putText(
            display,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Bottom HUD
        cv2.rectangle(display, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"TRACKING CAM 0 | FPS: {fps:.1f} | TRACKING XE: {len(tracks)} xe | [Q]: THOAT",
            (10, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return display

    def run(self) -> None:
        """Main dual-window loop."""
        print("\n>>> System running. Window 1: Vehicle Tracking (Cam 0 - 30 FPS) | Window 2: Gate (Cam 1 - 10 FPS) <<<")
        print(">>> Press 'I' for Check-in, 'O' for Check-out, 'F' for Find My Car (Face Scan), 'Q' to Quit. <<<\n")

        prev_time = time.time()

        try:
            while True:
                # 1. Read frames from both cameras
                ret_track, frame_track = self.cam_tracking.read()
                ret_gate, frame_gate = self.cam_gate.read()

                if not ret_track or frame_track is None:
                    time.sleep(0.01)
                    continue

                if not ret_gate or frame_gate is None:
                    frame_gate = frame_track.copy()

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                # 2. Window 1: Real-Time Vehicle & Slot Tracking (Camera 0)
                tracks = self.tracker.track(frame_track)
                self.current_occupancy = self.tracker.get_slot_occupancy(tracks)

                # Auto-assign pending check-in sessions to newly detected vehicle tracks
                for tr in tracks:
                    tr_id = tr.track_id
                    if tr_id not in self.track_session_map:
                        if self.pending_sessions:
                            sess_info = self.pending_sessions.pop(0)
                            self.track_session_map[tr_id] = sess_info
                            print(f"[TRACK BIND] Tracker Car #{tr_id} da duoc gan cho Session #{sess_info['id']} (Bien so: {sess_info['plate']})")

                display_tracking = self.draw_tracking_dashboard(frame_track, fps, tracks, self.current_occupancy)

                # 3. Window 2: Gate Preview (Camera 1) - Rate limited to 10 FPS for max tracking performance
                now = time.time()
                if now - self.last_gate_frame_time >= 0.10:
                    self.cached_gate_display = self.draw_gate_dashboard(frame_gate, fps)
                    self.last_gate_frame_time = now

                display_gate = self.cached_gate_display if self.cached_gate_display is not None else self.draw_gate_dashboard(frame_gate, fps)

                # 4. Render GUI windows
                key = -1
                if not self.headless:
                    cv2.imshow("1. Smart Parking - Vehicle Tracking (Camera 0)", display_tracking)
                    cv2.imshow("2. Smart Parking - Gate Check-in/out (Camera 1)", display_gate)
                    key = cv2.waitKey(10) & 0xFF
                else:
                    time.sleep(0.03)

                # Key Controls
                if key in (ord("q"), ord("Q"), 27):
                    break

                # Check-in (ON-DEMAND Face + LP + Face Check)
                if key in (ord("i"), ord("r"), ord("I"), ord("R")):
                    self.handle_checkin(frame_gate)

                # Check-out (ON-DEMAND Dual Verification)
                if key in (ord("o"), ord("x"), ord("O"), ord("X")):
                    self.handle_checkout(frame_gate)

                # Find My Car (Face Scan query slot)
                if key in (ord("f"), ord("s"), ord("F"), ord("S")):
                    self.handle_find_my_car(frame_gate)

        finally:
            self.cam_tracking.release()
            if self.cam_gate is not self.cam_tracking:
                self.cam_gate.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.db.close()
            print("\n[System] Shutdown completed gracefully.")


def main():
    parser = argparse.ArgumentParser(description="Smart Parking Dual Camera System (RPi 5)")
    parser.add_argument("--cam-tracking", type=int, default=0, help="Camera index for vehicle tracking (default 0)")
    parser.add_argument("--cam-gate", type=int, default=1, help="Camera index for entry/exit gate (default 1)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    app = DualCameraSmartParking(
        cam_tracking_src=args.cam_tracking,
        cam_gate_src=args.cam_gate,
        headless=args.headless,
    )
    app.run()


if __name__ == "__main__":
    main()
