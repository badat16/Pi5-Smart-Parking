"""
Generates a visualization preview showing the exact 4-Zone 12-Slot Parking Layout
and the Horizontal Crossing Line starting at the North edge of the green space.
"""

import cv2
import numpy as np
import config
from core.vehicle_tracker import VehicleTracker

def generate_preview():
    w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
    
    # Try loading an existing test image if available
    img = cv2.imread("test_images/101.jpg")
    if img is None:
        img = cv2.imread("test_images/1.jpg")
    if img is not None:
        img = cv2.resize(img, (w, h))
    else:
        # Synthetic mock frame
        img = np.ones((h, w, 3), dtype=np.uint8) * 80

    # Instantiate tracker without loading model weights for pure drawing visualization test
    tracker = VehicleTracker.__new__(VehicleTracker)
    tracker.enable_crossing_line = getattr(config, "ENABLE_CROSSING_LINE_FILTER", True)
    tracker.crossing_line_y = getattr(config, "CROSSING_LINE_Y", 295)
    tracker.crossing_line_color = getattr(config, "CROSSING_LINE_COLOR", (0, 255, 255))
    tracker.enable_roi_filter = getattr(config, "ENABLE_ROI_FILTER", True)
    tracker.roi_polygons = [np.array(poly, dtype=np.int32) for poly in getattr(config, "PARKING_ROI_POLYGONS", [])]
    
    tracker.zones_config = getattr(config, "PARKING_ZONES", {})
    tracker.parsed_slots = {}
    for zone_id, zone_info in tracker.zones_config.items():
        z_name = zone_info.get("name", f"Khu {zone_id}")
        for slot_id, pts in zone_info.get("slots", {}).items():
            tracker.parsed_slots[slot_id] = {
                "zone_id": zone_id,
                "zone_name": z_name,
                "poly": np.array(pts, dtype=np.int32),
            }

    # Simulate mock tracks for demo: Car #1 in A2, Car #2 in C1
    from core.vehicle_tracker import SimpleTrack
    mock_track_1 = SimpleTrack(1, (150, 25, 200, 90), 2, 0.92)
    mock_track_2 = SimpleTrack(2, (90, 210, 140, 280), 2, 0.88)
    mock_tracks = [mock_track_1, mock_track_2]

    # Draw elements
    tracker.names = {2: "car"}
    img = tracker.draw_tracks(img, mock_tracks)

    # Header Banner
    cv2.rectangle(img, (0, 0), (w, 35), (30, 30, 30), -1)
    cv2.putText(
        img,
        "SMART PARKING: 4 KHU (A, B, C, D) - 12 Ô ĐẬU & VẠCH TRACKING",
        (15, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    out_path = "test_images/roi_boundary_preview.jpg"
    cv2.imwrite(out_path, img)
    print(f"[Preview] Saved 4-Zone 12-Slot ROI preview to {out_path}")

if __name__ == "__main__":
    generate_preview()

