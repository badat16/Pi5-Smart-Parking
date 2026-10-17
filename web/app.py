"""
Flask Web Dashboard & API Server for Smart Parking System (Raspberry Pi 5).
Operates asynchronously using thread-safe shared state.
"""

import threading
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import psutil

import config


class WebState:
    """Thread-safe state container for Web Server <-> AI Engine communication."""

    def __init__(self):
        self._lock = threading.Lock()

        # Frame buffers (JPEG bytes)
        self.tracking_frame_jpeg: Optional[bytes] = None
        self.gate_frame_jpeg: Optional[bytes] = None

        # Telemetry & AI state
        self.fps: float = 0.0
        self.occupancy: Dict[str, Dict[str, Any]] = {}
        self.tracks: List[Dict[str, Any]] = []
        self.status_msg: str = "SYSTEM READY"
        self.status_color: Tuple[int, int, int] = (0, 255, 0)
        self.barrier_status: str = "BARRIER: CLOSED"
        self.last_transaction: str = "No recent transactions"
        self.active_cars_count: int = 0
        self.pending_sessions_count: int = 0
        self.recent_sessions: List[Dict[str, Any]] = []
        self.cam_tracking_active: bool = False
        self.cam_gate_active: bool = False

        # Action queue for remote web triggers (checkin, checkout, find_car)
        self.action_queue: List[str] = []

    def update_frames(self, tracking_frame: Optional[np.ndarray], gate_frame: Optional[np.ndarray]) -> None:
        """Encode OpenCV BGR frames to JPEG and update shared buffer."""
        quality = getattr(config, "WEB_STREAM_QUALITY", 70)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        track_bytes = None
        if tracking_frame is not None and tracking_frame.size > 0:
            ret, buf = cv2.imencode(".jpg", tracking_frame, encode_param)
            if ret:
                track_bytes = buf.tobytes()

        gate_bytes = None
        if gate_frame is not None and gate_frame.size > 0:
            ret, buf = cv2.imencode(".jpg", gate_frame, encode_param)
            if ret:
                gate_bytes = buf.tobytes()

        with self._lock:
            if track_bytes:
                self.tracking_frame_jpeg = track_bytes
                self.cam_tracking_active = True
            if gate_bytes:
                self.gate_frame_jpeg = gate_bytes
                self.cam_gate_active = True

    def update_telemetry(
        self,
        fps: float,
        occupancy: Dict[str, Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        status_msg: str,
        status_color: Tuple[int, int, int],
        barrier_status: str,
        last_transaction: str,
        active_cars_count: int,
        pending_sessions_count: int,
        recent_sessions: List[Dict[str, Any]],
    ) -> None:
        """Update system state variables safely."""
        with self._lock:
            self.fps = fps
            self.occupancy = occupancy
            self.tracks = tracks
            self.status_msg = status_msg
            self.status_color = status_color
            self.barrier_status = barrier_status
            self.last_transaction = last_transaction
            self.active_cars_count = active_cars_count
            self.pending_sessions_count = pending_sessions_count
            self.recent_sessions = recent_sessions

    def get_tracking_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.tracking_frame_jpeg

    def get_gate_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.gate_frame_jpeg

    def push_action(self, action: str) -> None:
        """Add remote trigger action to queue."""
        with self._lock:
            if action not in self.action_queue:
                self.action_queue.append(action)

    def pop_action(self) -> Optional[str]:
        """Fetch next pending remote trigger action."""
        with self._lock:
            if self.action_queue:
                return self.action_queue.pop(0)
            return None


# Global web state reference
web_state = WebState()


def get_cpu_temp() -> Optional[float]:
    """Read CPU temperature on Raspberry Pi 5 or Linux systems."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_raw = f.read().strip()
            return round(float(temp_raw) / 1000.0, 1)
    except Exception:
        pass

    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            return round(temps["cpu_thermal"][0].current, 1)
        elif "coretemp" in temps:
            return round(temps["coretemp"][0].current, 1)
    except Exception:
        pass

    return None


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    def mjpeg_generator(stream_type: str):
        target_fps = getattr(config, "WEB_STREAM_FPS", 15)
        interval = 1.0 / max(1, target_fps)

        # Blank 640x480 placeholder image if camera is not active yet
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank_img,
            "CAMERA OFFLINE / INITIALIZING...",
            (110, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )
        _, blank_buf = cv2.imencode(".jpg", blank_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        blank_bytes = blank_buf.tobytes()

        while True:
            t0 = time.time()
            if stream_type == "tracking":
                frame_bytes = web_state.get_tracking_jpeg()
            else:
                frame_bytes = web_state.get_gate_jpeg()

            if frame_bytes is None:
                frame_bytes = blank_bytes

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            dt = time.time() - t0
            time.sleep(max(0.005, interval - dt))

    @app.route("/video_feed/tracking")
    def video_feed_tracking():
        return Response(
            mjpeg_generator("tracking"),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/video_feed/gate")
    def video_feed_gate():
        return Response(
            mjpeg_generator("gate"),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/system")
    def api_system():
        cpu_usage = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        cpu_temp = get_cpu_temp()

        with web_state._lock:
            fps = web_state.fps
            cam_track_ok = web_state.cam_tracking_active
            cam_gate_ok = web_state.cam_gate_active

        return jsonify({
            "fps": round(fps, 1),
            "cpu_percent": round(cpu_usage, 1),
            "ram_percent": round(mem.percent, 1),
            "ram_used_mb": round(mem.used / (1024 * 1024), 1),
            "ram_total_mb": round(mem.total / (1024 * 1024), 1),
            "cpu_temp": cpu_temp,
            "camera_status": {
                "tracking_camera": "ONLINE" if cam_track_ok else "OFFLINE",
                "gate_camera": "ONLINE" if cam_gate_ok else "OFFLINE",
            },
        })

    @app.route("/api/parking")
    def api_parking():
        with web_state._lock:
            occ = web_state.occupancy.copy()
            pending_count = web_state.pending_sessions_count

        zone_summary = {"A": [0, 3], "B": [0, 3], "C": [0, 3], "D": [0, 3]}
        total_occupied = 0
        total_slots = len(occ) if occ else 12

        for slot_id, data in occ.items():
            z_id = data.get("zone_id", "A")
            if z_id in zone_summary:
                if data.get("occupied", False):
                    zone_summary[z_id][0] += 1
                    total_occupied += 1

        total_vacant = max(0, total_slots - total_occupied)

        return jsonify({
            "total_slots": total_slots,
            "total_occupied": total_occupied,
            "total_vacant": total_vacant,
            "pending_count": pending_count,
            "zone_summary": zone_summary,
            "slots": occ,
        })

    @app.route("/api/vehicle")
    def api_vehicle():
        with web_state._lock:
            tracks_data = web_state.tracks.copy()
            active_count = web_state.active_cars_count

        return jsonify({
            "active_cars_count": active_count,
            "tracks": tracks_data,
        })

    @app.route("/api/status")
    def api_status():
        with web_state._lock:
            msg = web_state.status_msg
            color_bgr = web_state.status_color
            barrier = web_state.barrier_status
            last_tx = web_state.last_transaction
            active_count = web_state.active_cars_count

        # Convert BGR tuple to HEX string
        hex_color = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

        return jsonify({
            "status_msg": msg,
            "status_color_hex": hex_color,
            "barrier_status": barrier,
            "last_transaction": last_tx,
            "active_cars_count": active_count,
        })

    @app.route("/api/sessions")
    def api_sessions():
        with web_state._lock:
            sessions = web_state.recent_sessions.copy()
        return jsonify({"recent_sessions": sessions})

    @app.route("/api/action/<action_name>", methods=["POST"])
    def api_action(action_name: str):
        action_name = action_name.lower().strip()
        valid_actions = ["checkin", "checkout", "find_car"]
        if action_name not in valid_actions:
            return jsonify({"success": False, "error": f"Invalid action. Choice: {valid_actions}"}), 400

        web_state.push_action(action_name)
        return jsonify({"success": True, "message": f"Action '{action_name}' queued successfully."})

    return app


def run_web_server(host: str = "0.0.0.0", port: int = 5000):
    app = create_app()
    # Suppress verbose Flask default logging for performance
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    print(f"\n[Web Server] Running Dashboard on http://0.0.0.0:{port} (Access via http://<PI_IP>:{port})")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def start_web_server_thread(host: str = "0.0.0.0", port: int = 5000) -> threading.Thread:
    thread = threading.Thread(target=run_web_server, args=(host, port), daemon=True)
    thread.start()
    return thread
