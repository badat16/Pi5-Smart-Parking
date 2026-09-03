"""
Vehicle Detection and Lightweight Multi-Object Tracker for Raspberry Pi 5.
Powered by Ultralytics YOLOv8 & centroid-velocity tracking.
"""

from typing import List, Tuple, Dict, Any, Optional
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


import os
import config


class VehicleTracker:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        iou_thresh: float = 0.60,
        track_score: float = 0.20,
        max_missed: int = 35,
        imgsz: int = 320,
        enable_roi: bool = None,
        roi_polygons: List[List[Tuple[int, int]]] = None,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        # Crossing Line Settings (Vạch cắt ngang)
        self.enable_crossing_line = getattr(config, "ENABLE_CROSSING_LINE_FILTER", True)
        self.crossing_line_y = getattr(config, "CROSSING_LINE_Y", 295)
        self.crossing_line_color = getattr(config, "CROSSING_LINE_COLOR", (0, 255, 255))

        # ROI & Parking Zones (4 Khu: A, B, C, D | 12 Ô đậu: A1..D3)
        self.enable_roi_filter = getattr(config, "ENABLE_ROI_FILTER", True) if enable_roi is None else enable_roi
        self.roi_filter_type = getattr(config, "ROI_FILTER_TYPE", "centroid")
        raw_polys = getattr(config, "PARKING_ROI_POLYGONS", []) if roi_polygons is None else roi_polygons
        self.roi_polygons = [np.array(poly, dtype=np.int32) for poly in raw_polys]

        # Load Parking Zones & 12 Slots configuration
        self.zones_config = getattr(config, "PARKING_ZONES", {})
        self.parsed_slots = {}
        for zone_id, zone_info in self.zones_config.items():
            z_name = zone_info.get("name", f"Khu {zone_id}")
            for slot_id, pts in zone_info.get("slots", {}).items():
                self.parsed_slots[slot_id] = {
                    "zone_id": zone_id,
                    "zone_name": z_name,
                    "poly": np.array(pts, dtype=np.int32),
                }

        # Prefer ONNX format for maximum CPU speed on Raspberry Pi 5
        onnx_path = str(model_path).rsplit(".", 1)[0] + ".onnx"
        if os.path.exists(onnx_path):
            self.model = YOLO(onnx_path, task="detect")
            print(f"[VehicleTracker] Loaded optimized ONNX model: {onnx_path}")
        else:
            self.model = YOLO(str(model_path))
            print(f"[VehicleTracker] Loaded PyTorch model: {model_path} (imgsz={imgsz})")

        self.tracker = LightweightTracker(iou_threshold=track_score, max_missed=max_missed)
        self.names = self.model.names if hasattr(self.model, "names") else {}

        # Dynamic bounds cache for resolution scaling
        self.scaled_slots = {}
        self.scaled_crossing_y = self.crossing_line_y
        self.min_x = 5
        self.max_x = 635
        self.max_y = 475
        self.last_frame_shape = (0, 0)

    def _update_bounds_for_frame(self, frame_h: int, frame_w: int):
        """Dynamically scale 640x480 configured slots and crossing line to match actual frame resolution."""
        if self.last_frame_shape == (frame_h, frame_w) and self.scaled_slots:
            return

        self.last_frame_shape = (frame_h, frame_w)
        s_x = frame_w / 640.0
        s_y = frame_h / 480.0

        self.scaled_crossing_y = int(self.crossing_line_y * s_y)

        all_pts = []
        self.scaled_slots = {}
        for slot_id, slot_data in self.parsed_slots.items():
            raw_poly = slot_data["poly"]
            scaled_poly = np.column_stack((raw_poly[:, 0] * s_x, raw_poly[:, 1] * s_y)).astype(np.int32)
            self.scaled_slots[slot_id] = {
                "zone_id": slot_data["zone_id"],
                "zone_name": slot_data["zone_name"],
                "poly": scaled_poly,
            }
            all_pts.extend(scaled_poly.tolist())

        if all_pts:
            pts_arr = np.array(all_pts)
            self.min_x = int(np.min(pts_arr[:, 0]))
            self.max_x = int(np.max(pts_arr[:, 0]))
            self.max_y = int(np.max(pts_arr[:, 1]))
        else:
            self.min_x = int(5 * s_x)
            self.max_x = int(635 * s_x)
            self.max_y = int(475 * s_y)

    def has_crossed_line(self, box: Tuple[int, int, int, int]) -> bool:
        """Check if vehicle box has crossed into the parking lot below the horizontal tracking line."""
        if not self.enable_crossing_line:
            return True
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cy >= self.scaled_crossing_y or y2 >= self.scaled_crossing_y

    def is_inside_roi(self, box: Tuple[int, int, int, int]) -> bool:
        """Check if vehicle centroid is strictly inside the rectangular ROI tracking box."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Strict check: centroid (cx, cy) MUST be strictly inside [min_x..max_x, scaled_crossing_y..max_y]
        in_horizontal = self.min_x <= cx <= self.max_x
        in_vertical = self.scaled_crossing_y <= cy <= self.max_y
        return in_horizontal and in_vertical

    def detect_vehicle_slot(self, box: Tuple[int, int, int, int]) -> Tuple[Optional[str], Optional[str]]:
        """Identify which Zone (A, B, C, D) and Slot (A1..D3) a vehicle is occupying."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        slots_to_check = self.scaled_slots if self.scaled_slots else self.parsed_slots

        for slot_id, slot_data in slots_to_check.items():
            poly = slot_data["poly"]
            dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
            if dist >= 0:
                return slot_id, slot_data["zone_name"]

        return None, None

    def get_slot_occupancy(self, tracks: List[SimpleTrack]) -> Dict[str, Dict[str, Any]]:
        """Get current occupancy status of all 12 parking slots."""
        occupancy = {}
        for slot_id, slot_data in self.parsed_slots.items():
            occupancy[slot_id] = {
                "zone_id": slot_data["zone_id"],
                "zone_name": slot_data["zone_name"],
                "occupied": False,
                "vehicle_id": None,
                "confidence": 0.0,
            }

        for track in tracks:
            display_box = track.box if track.missed == 0 else track.predicted_box()
            slot_id, zone_name = self.detect_vehicle_slot(display_box)
            if slot_id and slot_id in occupancy:
                occupancy[slot_id]["occupied"] = True
                occupancy[slot_id]["vehicle_id"] = track.track_id
                occupancy[slot_id]["confidence"] = track.conf
                track.assigned_slot = slot_id
                track.assigned_zone = zone_name
            else:
                track.assigned_slot = None
                track.assigned_zone = None

        return occupancy

    def track(self, frame: np.ndarray) -> List[SimpleTrack]:
        """Detect and update tracks for vehicles that crossed line and are in ROI."""
        h, w = frame.shape[:2]
        self._update_bounds_for_frame(h, w)

        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.imgsz, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                box_tuple = (x1, y1, x2, y2)

                # Track only if vehicle has crossed horizontal line and is inside active zone
                if self.is_inside_roi(box_tuple):
                    detections.append((box_tuple, class_id, conf))

        tracks = self.tracker.update(detections)
        self.get_slot_occupancy(tracks)
        return tracks

    def draw_crossing_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw horizontal tracking line and active tracking zone boundary box."""
        if not self.enable_crossing_line:
            return frame

        h, w = frame.shape[:2]
        self._update_bounds_for_frame(h, w)

        y_pos = self.scaled_crossing_y

        # 1. Horizontal Crossing Line
        cv2.line(frame, (10, y_pos), (w - 10, y_pos), self.crossing_line_color, 2, cv2.LINE_AA)

        # 2. Outer Boundary Frame matching EXACT slot boundaries!
        cv2.rectangle(frame, (self.min_x, y_pos), (self.max_x, self.max_y), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "VUNG TRACKING XE (ACTIVE ZONE)",
            (self.min_x + 8, y_pos + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def draw_parking_zones(self, frame: np.ndarray, occupancy: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Draw 4 Zones (A, B, C, D) & 12 Slots (A1..D3) overlay with vacancy color coding."""
        if not getattr(config, "SHOW_ROI_BOUNDARY", True) or not self.parsed_slots:
            return frame

        h, w = frame.shape[:2]
        self._update_bounds_for_frame(h, w)

        overlay = frame.copy()
        vacant_color = getattr(config, "SLOT_VACANT_COLOR", (0, 255, 0))
        occupied_color = getattr(config, "SLOT_OCCUPIED_COLOR", (0, 0, 255))
        fill_alpha = getattr(config, "ROI_FILL_ALPHA", 0.20)

        slots_to_draw = self.scaled_slots if self.scaled_slots else self.parsed_slots

        for slot_id, slot_info in occupancy.items():
            poly = slots_to_draw[slot_id]["poly"]
            is_occupied = slot_info["occupied"]
            color = occupied_color if is_occupied else vacant_color

            if fill_alpha > 0.0:
                cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
            label = f"{slot_id}"
            if is_occupied:
                label += f" [Car #{slot_info['vehicle_id']}]"

            cv2.putText(
                frame,
                label,
                (cx - 18, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 if not is_occupied else 0.40,
                (255, 255, 255) if is_occupied else color,
                2 if is_occupied else 1,
                cv2.LINE_AA,
            )

        # Draw Zone Headers (Zone A, B, C, D) positioned accurately
        s_x, s_y = w / 640.0, h / 480.0
        zone_header_pts = {
            "A": (int(100 * s_x), int(130 * s_y)),  # Khu A (Top-Left under green patch)
            "B": (int(470 * s_x), int(130 * s_y)),  # Khu B (Top-Right under green patch)
            "C": (int(100 * s_x), int(350 * s_y)),  # Khu C (Bottom-Left above zone C)
            "D": (int(470 * s_x), int(350 * s_y)),  # Khu D (Bottom-Right above zone D)
        }
        for z_code, pt in zone_header_pts.items():
            cv2.putText(
                frame,
                f"KHU {z_code}",
                pt,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if fill_alpha > 0.0:
            cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
        return frame

    def draw_tracks(self, frame: np.ndarray, tracks: List[SimpleTrack]) -> np.ndarray:
        """Draw crossing line, 4-Zone 12-Slot overlay, and tracked vehicle boxes."""
        # 1. Draw horizontal tracking crossing line
        self.draw_crossing_line(frame)

        # 2. Compute slot occupancy & render 12 parking spots
        occupancy = self.get_slot_occupancy(tracks)
        self.draw_parking_zones(frame, occupancy)

        # 3. Render vehicle bounding boxes with concise label: Car <id> <Khu> <conf>
        for track in tracks:
            display_box = track.box if track.missed == 0 else track.predicted_box()
            x1, y1, x2, y2 = map(int, display_box)

            color = (0, 255, 0) if track.missed == 0 else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if getattr(track, "assigned_slot", None) and track.assigned_slot in self.parsed_slots:
                display_label = f"Car {track.track_id} {track.assigned_slot} {track.conf:.2f}"
            else:
                display_label = f"Car {track.track_id} {track.conf:.2f}"

            # Draw background text box for readability
            text_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
            cv2.rectangle(frame, (x1, max(0, y1 - text_size[1] - 8)), (x1 + text_size[0] + 6, y1), (0, 0, 0), -1)
            cv2.putText(
                frame,
                display_label,
                (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255) if getattr(track, "assigned_slot", None) else color,
                1,
                cv2.LINE_AA,
            )
            cx, cy = centroid(display_box)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

        return frame


