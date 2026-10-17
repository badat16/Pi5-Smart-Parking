"""
Fast ALPR Demo for Raspberry Pi 5 using YOLOv5 Detector + Fast-Plate-OCR (ONNX Engine).
Combines YOLOv5 Plate Detection with fast ONNX-based OCR for ultra-low latency.
"""

import argparse
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

import config
from core.camera_stream import CameraStream

try:
    from fast_plate_ocr import LicensePlateRecognizer
    HAS_FAST_OCR = True
except ImportError:
    HAS_FAST_OCR = False


class FastPlateEngine:
    def __init__(
        self,
        detector_path: str = str(config.PLATE_DETECTOR_MODEL),
        yolov5_dir: str = str(config.YOLOV5_DIR),
        det_conf: float = config.PLATE_DET_CONF,
        img_size: int = config.PLATE_IMG_SIZE,
        ocr_model: str = "cct-xs-v2-global-model",
        device: str = "cpu",
    ):
        if not HAS_FAST_OCR:
            raise ImportError(
                "fast-plate-ocr is not installed. Please run: pip install 'fast-plate-ocr[onnx]'"
            )

        self.det_conf = det_conf
        self.img_size = img_size
        self.device = device

        print(f"[FastPlateEngine] Loading YOLOv5 Detector on {device}...")
        self.detector = torch.hub.load(
            str(Path(yolov5_dir).resolve()),
            "custom",
            path=str(Path(detector_path).resolve()),
            source="local",
            force_reload=False,
            device=device,
        )
        self.detector.conf = self.det_conf
        print("[FastPlateEngine] YOLOv5 Detector loaded successfully.")

        print(f"[FastPlateEngine] Initializing fast-plate-ocr ({ocr_model}) with ONNX runtime...")
        # Auto-detect CUDA/CPU ONNX providers
        self.ocr = LicensePlateRecognizer(
            hub_ocr_model=ocr_model,
            device="auto" if device != "cpu" else "cpu",
        )
        print("[FastPlateEngine] Fast-Plate-OCR ONNX engine ready.")

    def detect_and_read(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        h_img, w_img = frame.shape[:2]
        plates_res = self.detector(frame, size=self.img_size)
        list_plates = plates_res.pandas().xyxy[0].values.tolist()

        detected = []
        for p in list_plates:
            x1 = max(0, int(p[0]))
            y1 = max(0, int(p[1]))
            x2 = min(w_img, int(p[2]))
            y2 = min(h_img, int(p[3]))
            conf = float(p[4])

            if (x2 - x1) < 15 or (y2 - y1) < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            try:
                # Fast ONNX prediction
                pred = self.ocr.run_one(crop, return_confidence=True)
                plate_text = pred.plate
                ocr_conf = float(np.mean(pred.char_probs)) if hasattr(pred, "char_probs") and len(pred.char_probs) > 0 else 0.0
                region = getattr(pred, "region", "")
            except Exception:
                plate_text = "unknown"
                ocr_conf = 0.0
                region = ""

            detected.append({
                "bbox": (x1, y1, x2, y2),
                "plate": plate_text if plate_text else "unknown",
                "conf": conf,
                "ocr_conf": ocr_conf,
                "region": region,
                "crop": crop,
            })

        return detected

    @staticmethod
    def draw_plates(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            plate = d["plate"]
            conf = d["conf"]
            ocr_conf = d.get("ocr_conf", 0.0)

            color = (0, 220, 0) if plate != "unknown" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"LP: {plate} ({ocr_conf:.0%})" if plate != "unknown" else f"LP ({conf:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame


def main():
    parser = argparse.ArgumentParser(description="Fast ALPR Demo (YOLOv5 + fast-plate-ocr ONNX)")
    parser.add_argument("-i", "--image", type=str, default="", help="Path to single image file for testing")
    parser.add_argument("-s", "--source", default="", help="Camera index (0, 1) or video path")
    parser.add_argument("-k", "--skip", type=int, default=3, help="Inference skip interval in frames (default: 3)")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="cct-xs-v2-global-model",
        help="fast-plate-ocr model (default: cct-xs-v2-global-model, options: cct-s-v2-global-model, global-plates-mobile-vit-v2-model)",
    )
    args = parser.parse_args()

    print("[INFO] Initializing Fast ALPR Engine...")
    engine = FastPlateEngine(
        detector_path=str(config.PLATE_DETECTOR_MODEL),
        yolov5_dir=str(config.YOLOV5_DIR),
        det_conf=config.PLATE_DET_CONF,
        img_size=config.PLATE_IMG_SIZE,
        ocr_model=args.model,
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

        print(f"\n--- FAST ALPR Result ({dt:.1f} ms) ---")
        if len(detections) == 0:
            print("No license plates detected.")
        else:
            for idx, d in enumerate(detections, 1):
                reg_str = f" [Region: {d['region']}]" if d.get('region') else ""
                print(f"Plate #{idx}: '{d['plate']}' (OCR Conf: {d['ocr_conf']:.2%}, Det Conf: {d['conf']:.2f}){reg_str}")

        engine.draw_plates(img, detections)
        cv2.putText(
            img,
            f"Infer: {dt:.1f}ms | Plates: {len(detections)} | ONNX",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if getattr(config, "ENABLE_GUI", True):
            cv2.imshow("Fast ALPR Image Test", img)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Live Camera Mode
    cam_src = args.source if args.source else config.CAMERA_SOURCE
    camera = CameraStream(
        source=cam_src,
        width=getattr(config, "CAMERA_WIDTH", 640),
        height=getattr(config, "CAMERA_HEIGHT", 480),
        fps=getattr(config, "CAMERA_FPS", 30),
        use_picamera2=getattr(config, "USE_PICAMERA2", False),
    )

    if not camera.is_opened():
        print(f"\n[ERROR] Could not open camera source '{cam_src}'.")
        camera.release()
        return

    skip_interval = max(1, args.skip)
    print("\n=======================================================")
    print("  FAST ALPR DEMO (YOLOv5 DETECTOR + FAST-PLATE-OCR ONNX)")
    print("=======================================================")
    print(f"  - OCR Engine  : fast-plate-ocr [{args.model}]")
    print(f"  - Optimization: Running inference every {skip_interval} frame(s)")
    print("  - [Q / ESC]  : Quit application\n")

    fps_history = deque(maxlen=30)
    frame_count = 0
    cached_detections = []
    last_infer_dt = 0.0
    enable_gui = getattr(config, "ENABLE_GUI", True)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            curr_time = time.time()
            fps_history.append(curr_time)
            if len(fps_history) > 1:
                time_span = fps_history[-1] - fps_history[0]
                fps = (len(fps_history) - 1) / time_span if time_span > 0 else 0.0
            else:
                fps = 0.0

            frame_count += 1
            is_infer_frame = ((frame_count - 1) % skip_interval == 0)

            if is_infer_frame:
                t0 = time.time()
                cached_detections = engine.detect_and_read(frame)
                last_infer_dt = (time.time() - t0) * 1000.0

            detections = cached_detections
            infer_dt = last_infer_dt

            engine.draw_plates(frame, detections)

            # HUD status overlay
            hud_text = f"FPS: {fps:.1f} | Infer: {infer_dt:.1f}ms | Plates: {len(detections)} | Fast-OCR ONNX"
            cv2.putText(
                frame,
                hud_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if enable_gui:
                cv2.imshow("Fast ALPR Live Stream (ONNX)", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            else:
                if is_infer_frame and len(detections) > 0:
                    for d in detections:
                        if d["plate"] != "unknown":
                            print(f"[{time.strftime('%H:%M:%S')}] Plate: {d['plate']} (OCR: {d['ocr_conf']:.2%})")
                time.sleep(0.03)

    finally:
        camera.release()
        if enable_gui:
            cv2.destroyAllWindows()
        print("[Fast ALPR] Camera released. Done.")


if __name__ == "__main__":
    main()
