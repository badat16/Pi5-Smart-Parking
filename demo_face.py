"""
Demo: Driver Face Recognition on Raspberry Pi 5.
Supports Check-in ('R'), Verification ('C'), and Check-out ('X').
"""

import time
import cv2
import numpy as np

import config
from database import ParkingDatabase
from core.face_engine import FaceEngine
from core.camera_stream import CameraStream


def draw_hud(frame: np.ndarray, status: str, active_count: int, fps: float):
    # Overlay top HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Active Sessions: {active_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Status: {status}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[R]: Check-in | [C]: Check | [X]: Check-out | [Q]: Quit",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def main():
    print("=== Demo Driver Face Recognition (RPi 5) ===")
    engine = FaceEngine(
        model_name=config.FACE_MODEL_NAME,
        det_thresh=config.FACE_DET_THRESHOLD,
        det_size=config.FACE_DET_SIZE,
        min_face_size=config.FACE_MIN_SIZE,
        num_threads=config.ONNX_NUM_THREADS,
    )
    db = ParkingDatabase(config.DB_PATH)

    camera = CameraStream(
        source=config.CAMERA_SOURCE,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
        use_picamera2=config.USE_PICAMERA2,
    )

    status_msg = "Ready. Position face in front of camera."
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            faces = engine.analyze(frame)
            primary_face = engine.choose_primary_face(faces)
            engine.draw_faces(frame, faces)

            draw_hud(frame, status_msg, db.active_count(), fps)

            if config.ENABLE_GUI:
                cv2.imshow("Face Recognition Demo (RPi 5)", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key == ord("q"):
                break

            # Key R: Register / Check-in
            if key == ord("r"):
                if primary_face is None:
                    status_msg = "CHECK-IN FAIL: Can 1 khuon mat ro rang!"
                    print(f"\n[!] {status_msg}")
                    continue

                emb = engine.get_embedding(primary_face)
                print("\n--- [CHECK-IN] DANG KY XE VAO ---")
                person_name = input("Ten/Ma nguoi gui (Enter de dung ID): ").strip()
                plate = input("Bien so xe: ").strip().upper() or "UNKNOWN"
                session_id = db.register(person_name, plate, emb)
                status_msg = f"CHECK-IN OK: ID #{session_id} | Plate: {plate}"
                print(f"[OK] {status_msg}")

            # Key C: Check similarity without closing session
            # Key X: Check similarity and close session (Check-out)
            if key in (ord("c"), ord("x")):
                if primary_face is None:
                    status_msg = "CHECK FAIL: Can 1 khuon mat ro rang!"
                    print(f"\n[!] {status_msg}")
                    continue

                emb = engine.get_embedding(primary_face)
                match = db.find_best_face_active(emb)

                if match is None:
                    status_msg = "NO MATCH: Khong co phien nao dang gui!"
                    print(f"\n[!] {status_msg}")
                    continue

                score = match["score"]
                if score >= config.FACE_MATCH_THRESHOLD:
                    status_msg = (
                        f"MATCH ({score:.3f}): {match['person_name']} | Plate: {match['plate']}"
                    )
                    print(f"\n--- [KET QUA DOI SOAT] ---")
                    print(f"Xac thuc thanh cong! Driver: {match['person_name']}, Plate: {match['plate']}, Sim: {score:.3f}")
                    print(f"Time In: {match['time_in']}")

                    if key == ord("x"):
                        res = db.close_session(match["id"], match_score=score)
                        status_msg += f" | CHECK-OUT: {res['duration_minutes']} mins"
                        print(f"[EXIT] Da dong phien #{match['id']}. Thoi gian do: {res['duration_minutes']} phut.")
                else:
                    status_msg = f"UNKNOWN DRIVER! Best Sim = {score:.3f} (< {config.FACE_MATCH_THRESHOLD})"
                    print(f"\n[!] CANH BAO: {status_msg}")

    finally:
        camera.release()
        if config.ENABLE_GUI:
            cv2.destroyAllWindows()
        db.close()
        print("[Face Demo] Finished.")


if __name__ == "__main__":
    main()
