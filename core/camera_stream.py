"""
Multi-threaded Camera Stream Manager.
Supports USB Webcams, Picamera2 (Raspberry Pi Camera Module v2/v3/HQ),
RTSP streams, and video files with minimal latency.
"""

import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class CameraStream:
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_picamera2: bool = False,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_picamera2 = use_picamera2

        self.cap = None
        self.picam2 = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

        self._start_camera()

    def _start_camera(self) -> None:
        if self.use_picamera2:
            try:
                from picam2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self.ret = True
                print("[CameraStream] Picamera2 started successfully.")
            except Exception as e:
                print(f"[CameraStream] Picamera2 init failed: {e}. Falling back to OpenCV VideoCapture...")
                self.use_picamera2 = False

        if not self.use_picamera2:
            src = self.source
            if isinstance(src, str) and src.isdigit():
                src = int(src)

            self.actual_source = src
            candidate_sources = [src]
            if isinstance(src, int) and src != 0:
                for alt in [2, 4, 1, 3]:
                    if alt not in candidate_sources:
                        candidate_sources.append(alt)

            opened_src = None
            for candidate in candidate_sources:
                if isinstance(candidate, int):
                    cap_test = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
                    if not cap_test.isOpened():
                        cap_test = cv2.VideoCapture(candidate)
                else:
                    cap_test = cv2.VideoCapture(candidate)

                if cap_test.isOpened():
                    try:
                        cap_test.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass
                    cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap_test.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    ret, frame = cap_test.read()
                    if ret and frame is not None:
                        self.cap = cap_test
                        self.actual_source = candidate
                        self.ret = ret
                        self.frame = frame
                        opened_src = candidate
                        print(f"[CameraStream] Successfully opened video stream on camera source '{candidate}'")
                        break
                    else:
                        cap_test.release()

            if self.cap is None or not self.cap.isOpened():
                # Final fallback to 0 only if nothing else works
                print(f"[CameraStream] Requested source {src} unavailable. Attempting fallback to index 0...")
                self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.actual_source = 0
                    self.ret, self.frame = self.cap.read()

            if self.cap is None or not self.cap.isOpened():
                print(f"[CameraStream] WARNING: Could not open any camera source.")
                self.ret = False
                return

        # Start background capture thread for real-time fresh frames
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self) -> None:
        while not self.stopped:
            if self.use_picamera2 and self.picam2 is not None:
                try:
                    img_rgb = self.picam2.capture_array()
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    with self.lock:
                        self.frame = img_bgr
                        self.ret = True
                except Exception:
                    self.ret = False
                time.sleep(1.0 / max(1, self.fps))
            elif self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    if ret:
                        self.frame = frame
                if not ret:
                    time.sleep(0.01)

    def is_opened(self) -> bool:
        """Check if camera stream is successfully initialized and opened."""
        if self.use_picamera2:
            return self.picam2 is not None and self.ret
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the most recent frame."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self) -> None:
        """Stop background capture and release resources."""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
