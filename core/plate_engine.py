"""
License Plate Recognition (ALPR) Engine for Raspberry Pi 5.
Combines YOLOv5 Nano Plate Detector, in-memory Deskew, and YOLOv5 Nano Character OCR.
"""

import os
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch

from .utils_rotate import deskew


class PlateEngine:
    def __init__(
        self,
        detector_path: str,
        ocr_path: str,
        yolov5_dir: str,
        det_conf: float = 0.50,
        ocr_conf: float = 0.55,
        img_size: int = 640,
        device: str = "cpu",
    ):
        self.det_conf = det_conf
        self.ocr_conf = ocr_conf
        self.img_size = img_size
        self.device = device

        detector_path = str(Path(detector_path).resolve())
        ocr_path = str(Path(ocr_path).resolve())
        yolov5_dir = str(Path(yolov5_dir).resolve())

        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Detector weight not found: {detector_path}")
        if not os.path.exists(ocr_path):
            raise FileNotFoundError(f"OCR weight not found: {ocr_path}")

        print(f"[PlateEngine] Loading YOLOv5 models on {device}...")
        
        # Load local YOLOv5 models without internet/hub update
        self.detector = torch.hub.load(
            yolov5_dir,
            "custom",
            path=detector_path,
            source="local",
            force_reload=False,
            device=device,
        )
        self.detector.conf = self.det_conf

        self.ocr = torch.hub.load(
            yolov5_dir,
            "custom",
            path=ocr_path,
            source="local",
            force_reload=False,
            device=device,
        )
        self.ocr.conf = self.ocr_conf
        print("[PlateEngine] Plate models loaded successfully.")

    @staticmethod
    def _linear_equation(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        if abs(x2 - x1) < 1e-5:
            return 0.0, y1
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b

    @staticmethod
    def _check_point_linear(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
        a, b = PlateEngine._linear_equation(x1, y1, x2, y2)
        y_pred = a * x + b
        return math.isclose(y_pred, y, abs_tol=3.0)

    def _read_characters(self, plate_crop: np.ndarray) -> str:
        """Run OCR on cropped plate and format 1-line or 2-line Vietnamese plate text."""
        if plate_crop is None or plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
            return "unknown"

        results = self.ocr(plate_crop)
        bb_list = results.pandas().xyxy[0].values.tolist()

        if len(bb_list) < 6 or len(bb_list) > 12:
            return "unknown"

        center_list = []
        y_sum = 0.0
        for bb in bb_list:
            x_c = (bb[0] + bb[2]) / 2.0
            y_c = (bb[1] + bb[3]) / 2.0
            y_sum += y_c
            char_label = str(bb[-1])
            center_list.append([x_c, y_c, char_label])

        # Find leftmost and rightmost points to test linearity
        l_point = min(center_list, key=lambda cp: cp[0])
        r_point = max(center_list, key=lambda cp: cp[0])

        lp_type = "1"  # 1-line plate by default
        if l_point[0] != r_point[0]:
            for ct in center_list:
                if not self._check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                    lp_type = "2"  # 2-line plate
                    break

        y_mean = y_sum / len(bb_list)

        if lp_type == "2":
            line_1 = []
            line_2 = []
            for c in center_list:
                if c[1] > y_mean:
                    line_2.append(c)
                else:
                    line_1.append(c)

            plate_text = ""
            for l1 in sorted(line_1, key=lambda x: x[0]):
                plate_text += str(l1[2])
            plate_text += "-"
            for l2 in sorted(line_2, key=lambda x: x[0]):
                plate_text += str(l2[2])
        else:
            plate_text = ""
            for l in sorted(center_list, key=lambda x: x[0]):
                plate_text += str(l[2])

        return plate_text.strip()

    def recognize_crop(self, crop_img: np.ndarray) -> str:
        """Deskew and OCR plate image directly in memory with multiple angle tests."""
        for change_cons in (0, 1):
            for center_thres in (0, 1):
                rotated = deskew(crop_img, change_cons=change_cons, center_thres=center_thres)
                text = self._read_characters(rotated)
                if text != "unknown" and len(text) >= 6:
                    return text
        return "unknown"

    def detect_and_read(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect plates in frame, deskew, and read characters.
        Returns list of dicts: [{'bbox': (x1, y1, x2, y2), 'plate': '29A12345', 'conf': 0.85, 'crop': crop_img}]
        """
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

            if (x2 - x1) < 20 or (y2 - y1) < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            plate_text = self.recognize_crop(crop)

            detected.append({
                "bbox": (x1, y1, x2, y2),
                "plate": plate_text,
                "conf": conf,
                "crop": crop,
            })

        return detected

    @staticmethod
    def draw_plates(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and OCR text on the frame."""
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            plate = d["plate"]
            conf = d["conf"]

            color = (0, 220, 0) if plate != "unknown" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"LP: {plate}" if plate != "unknown" else f"LP ({conf:.2f})"
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
