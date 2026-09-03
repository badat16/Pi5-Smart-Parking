"""
Vehicle Detection and Lightweight Multi-Object Tracker for Raspberry Pi 5.
Powered by Ultralytics YOLOv8 & centroid-velocity tracking.
"""

from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))


def move_box(box: Tuple[float, float, float, float], dx: float, dy: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class SimpleTrack:
    def __init__(self, track_id: int, box: Tuple[float, float, float, float], class_id: int, conf: float):
        self.track_id = track_id
        self.box = box
        self.class_id = class_id
        self.conf = conf
        self.missed = 0
        self.hits = 1
        self.age = 1
        self.velocity = (0.0, 0.0)

    def predicted_box(self) -> Tuple[float, float, float, float]:
        if self.missed <= 0:
            return self.box
        dx, dy = self.velocity
        return move_box(self.box, dx * self.missed, dy * self.missed)

    def update(self, box: Tuple[float, float, float, float], class_id: int, conf: float) -> None:
        prev_cx, prev_cy = centroid(self.box)
        new_cx, new_cy = centroid(box)
        self.velocity = (new_cx - prev_cx, new_cy - prev_cy)
        self.box = box
        self.class_id = class_id
        self.conf = conf
        self.missed = 0
        self.hits += 1
        self.age += 1

    def mark_missed(self) -> None:
        self.missed += 1
        self.age += 1


class LightweightTracker:
    def __init__(self, iou_threshold: float = 0.22, max_missed: int = 35, max_center_distance: float = 160.0):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_distance = max_center_distance
        self.tracks: List[SimpleTrack] = []
        self.next_id = 1

    def _score(self, track: SimpleTrack, box: Tuple[float, float, float, float], class_id: int) -> float:
        predicted = track.predicted_box()
        iou_score = iou(predicted, box)

        track_cx, track_cy = centroid(predicted)
        box_cx, box_cy = centroid(box)
        distance = np.hypot(track_cx - box_cx, track_cy - box_cy)

        box_w, box_h = box_size(box)
        normalized_limit = max(self.max_center_distance, box_w * 1.5, box_h * 1.5)
        distance_score = max(0.0, 1.0 - distance / normalized_limit)

        class_bonus = 0.15 if track.class_id == class_id else 0.0
        return 0.6 * iou_score + 0.3 * distance_score + class_bonus

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], int, float]]) -> List[SimpleTrack]:
        assigned_tracks = set()
        updated_tracks = []

        detections = sorted(detections, key=lambda item: item[2], reverse=True)

        for box, class_id, conf in detections:
            best_track = None
            best_score = 0.0

            for track in self.tracks:
                if track.track_id in assigned_tracks:
                    continue
                score = self._score(track, box, class_id)
                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is not None and best_score >= self.iou_threshold:
                best_track.update(box, class_id, conf)
                assigned_tracks.add(best_track.track_id)
                updated_tracks.append(best_track)
            else:
                new_track = SimpleTrack(self.next_id, box, class_id, conf)
                self.next_id += 1
                assigned_tracks.add(new_track.track_id)
                updated_tracks.append(new_track)

        for track in self.tracks:
            if track.track_id not in assigned_tracks:
                track.mark_missed()
                if track.missed <= self.max_missed:
                    updated_tracks.append(track)

        self.tracks = updated_tracks
        return self.tracks


class VehicleTracker:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.60,
        track_score: float = 0.20,
        max_missed: int = 35,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = YOLO(str(model_path))
        self.tracker = LightweightTracker(iou_threshold=track_score, max_missed=max_missed)
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def track(self, frame: np.ndarray) -> List[SimpleTrack]:
        """Detect and update tracks for frame."""
        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append(((x1, y1, x2, y2), class_id, conf))

        return self.tracker.update(detections)

    def draw_tracks(self, frame: np.ndarray, tracks: List[SimpleTrack]) -> np.ndarray:
        """Draw tracked vehicles bounding boxes and trajectory centroid."""
        for track in tracks:
            display_box = track.box if track.missed == 0 else track.predicted_box()
            x1, y1, x2, y2 = map(int, display_box)
            label_name = self.names.get(track.class_id, f"car_{track.class_id}") if isinstance(self.names, dict) else str(track.class_id)

            color = (0, 255, 0) if track.missed == 0 else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label_name} #{track.track_id} ({track.conf:.2f})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2,
                cv2.LINE_AA,
            )
            cx, cy = centroid(display_box)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
        return frame
