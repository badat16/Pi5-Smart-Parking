"""
License Plate Deskew & Contrast Enhancement Utilities.
Optimized for in-memory processing on CPU/Raspberry Pi 5.
"""

import math
import cv2
import numpy as np


def change_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance license plate contrast using CLAHE in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around its center."""
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    image_center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compute_skew(src_img: np.ndarray, center_thres: int = 0) -> float:
    """Compute plate tilt angle using Hough Line Transform."""
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    else:
        h, w = src_img.shape

    if h < 5 or w < 5:
        return 0.0

    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)

    if lines is None:
        return 0.0

    line_segments = lines.reshape(-1, 4)
    min_line = 100
    min_line_pos = 0

    for i, (x1, y1, x2, y2) in enumerate(line_segments):
        center_point_y = (y1 + y2) / 2.0
        if center_thres == 1 and center_point_y < 7:
            continue
        if center_point_y < min_line:
            min_line = center_point_y
            min_line_pos = i

    x1, y1, x2, y2 = line_segments[min_line_pos]
    ang = np.arctan2(y2 - y1, x2 - x1)
    if math.fabs(ang) <= math.pi / 6:
        return (ang * 180.0) / math.pi
    return 0.0


def deskew(src_img: np.ndarray, change_cons: int = 0, center_thres: int = 0) -> np.ndarray:
    """Deskew tilted license plate image."""
    if src_img is None or src_img.size == 0:
        return src_img

    processed = change_contrast(src_img) if change_cons == 1 else src_img
    angle = compute_skew(processed, center_thres)
    return rotate_image(src_img, angle)
