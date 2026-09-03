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
            # Parse source
            src = self.source
            if isinstance(src, str) and src.isdigit():
                src = int(src)

            # Try V4L2 backend on Linux/Raspberry Pi
            try:
                self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(src)
            except Exception:
                self.cap = cv2.VideoCapture(src)

            if not self.cap.isOpened():
                print(f"[CameraStream] WARNING: Could not open camera source '{self.source}'.")
                self.ret = False
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.ret, self.frame = self.cap.read()

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
