"""
WebRTC Stream Server for Smart Parking System (Raspberry Pi 5).
Uses aiortc to provide ultra-low latency UDP streaming directly from Python memory.
Zero CPU encoding overhead when no clients are viewing.
"""

import asyncio
import fractions
import logging
import threading
import time
from typing import Dict, Any, Optional, Set

import av
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

import config

# Suppress excessive aioice / aiortc connection logs
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

pcs: Set[RTCPeerConnection] = set()
_webrtc_loop: Optional[asyncio.AbstractEventLoop] = None
_webrtc_thread: Optional[threading.Thread] = None


def make_placeholder(text: str = "STREAM CONNECTING...") -> np.ndarray:
    """Generate blank placeholder frame when camera is not ready yet."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (80, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2,
    )
    return img


class CameraStreamTrack(MediaStreamTrack):
    """
    aiortc MediaStreamTrack that pulls annotated OpenCV BGR frames from WebState
    and encodes them into H.264 / VP8 via PyAV/FFmpeg on demand.
    """
    kind = "video"

    def __init__(self, stream_type: str, web_state: Any, fps: int = 15):
        super().__init__()
        self.stream_type = stream_type
        self.web_state = web_state
        self.fps = max(5, min(30, fps))
        self.time_base = fractions.Fraction(1, 90000)
        self._interval = 1.0 / self.fps
        self._timestamp = 0
        self._start_time: Optional[float] = None
        self._placeholder = make_placeholder(f"{stream_type.upper()} CAMERA CONNECTING...")

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()

        # Pull raw BGR numpy frame from WebState
        frame = self.web_state.get_raw_frame(self.stream_type)
        if frame is None or frame.size == 0:
            frame = self._placeholder

        # Ensure standard 640x480 resolution
        if frame.shape[0] != 480 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480))

        # Convert numpy BGR to av.VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        loop = asyncio.get_event_loop()
        now = loop.time()

        if self._start_time is None:
            self._start_time = now
            self._timestamp = 0
        else:
            self._timestamp += int(90000 / self.fps)
            expected_time = self._start_time + (self._timestamp / 90000)
            delay = expected_time - now
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.2:
                # If falling behind by more than 200ms, resync to present time
                self._start_time = now
                self._timestamp = 0

        return self._timestamp, self.time_base


def get_webrtc_loop() -> asyncio.AbstractEventLoop:
    """Ensure dedicated background asyncio event loop for WebRTC."""
    global _webrtc_loop, _webrtc_thread
    if _webrtc_loop is None or not _webrtc_loop.is_running():
        _webrtc_loop = asyncio.new_event_loop()
        _webrtc_thread = threading.Thread(
            target=_webrtc_loop.run_forever,
            daemon=True,
            name="WebRTC-AsyncIO-Worker",
        )
        _webrtc_thread.start()
    return _webrtc_loop


async def process_offer(params: Dict[str, Any], web_state: Any) -> Dict[str, Any]:
    """Negotiate WebRTC SDP offer and bind requested camera track."""
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    stream_type = params.get("stream", "tracking")
    fps = getattr(config, "WEBRTC_FPS", 15)

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    track = CameraStreamTrack(stream_type=stream_type, web_state=web_state, fps=fps)
    pc.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


def handle_offer_sync(params: Dict[str, Any], web_state: Any, timeout: float = 8.0) -> Dict[str, Any]:
    """Thread-safe synchronous wrapper for Flask request handler."""
    loop = get_webrtc_loop()
    future = asyncio.run_coroutine_threadsafe(process_offer(params, web_state), loop)
    return future.result(timeout=timeout)


async def _close_all():
    coros = [pc.close() for pc in list(pcs)]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()


def close_all_peers():
    """Cleanly close all active WebRTC sessions."""
    loop = get_webrtc_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(_close_all(), loop)
