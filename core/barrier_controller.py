"""
Barrier Controller Module for Smart Parking System (Raspberry Pi 5).
Controls 2x SG90 9G 180° Micro Servos (Entry/Exit Gates) and 2x Push Buttons (Check-in/out).

Wiring Pinout (Raspberry Pi 5):
  - Entry Servo Signal: GPIO 5  (Pin 29)
  - Exit Servo Signal:  GPIO 6  (Pin 31)
  - Entry Button (NO):  GPIO 23 (Pin 16), COM -> GND (Pin 14)
  - Exit Button (NO):   GPIO 24 (Pin 18), COM -> GND (Pin 20)
  - Servo Power: 5V external power, GND tied to Pi GND (Pin 39)
"""

import threading
import time
import queue
import warnings
from typing import Optional, Callable, Dict, Any

import config

# Suppress software PWM fallback warning on Raspberry Pi
try:
    from gpiozero.exc import PWMSoftwareFallback
    warnings.filterwarnings("ignore", category=PWMSoftwareFallback)
except ImportError:
    pass


class BarrierController:
    """Thread-safe Hardware Controller for SG90 Servos and Entry/Exit Push Buttons."""

    def __init__(self, action_callback: Optional[Callable[[str], None]] = None):
        self.action_callback = action_callback
        self.action_queue: queue.Queue = queue.Queue()

        self.entry_state: str = "CLOSED"  # "CLOSED", "OPEN", "OPENING", "CLOSING"
        self.exit_state: str = "CLOSED"

        self._entry_lock = threading.Lock()
        self._exit_lock = threading.Lock()
        self._entry_timer: Optional[threading.Timer] = None
        self._exit_timer: Optional[threading.Timer] = None

        self._last_entry_press: float = 0.0
        self._last_exit_press: float = 0.0
        self._debounce_window: float = getattr(config, "BUTTON_COOLDOWN_TIME", 2.0)

        self.servo_entry = None
        self.servo_exit = None
        self.btn_entry = None
        self.btn_exit = None

        self.is_hardware_available = False
        self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize GPIO Servos and Push Buttons using gpiozero."""
        if not getattr(config, "ENABLE_HARDWARE_BARRIER", True):
            print("[Hardware] Hardware barrier disabled in config (Simulation mode).")
            return

        try:
            from gpiozero import AngularServo, Button

            entry_servo_pin = getattr(config, "SERVO_ENTRY_PIN", 5)
            exit_servo_pin = getattr(config, "SERVO_EXIT_PIN", 6)
            entry_btn_pin = getattr(config, "BUTTON_ENTRY_PIN", 23)
            exit_btn_pin = getattr(config, "BUTTON_EXIT_PIN", 24)

            min_pulse = getattr(config, "SERVO_MIN_PULSE_WIDTH", 0.0005)
            max_pulse = getattr(config, "SERVO_MAX_PULSE_WIDTH", 0.0024)
            bounce_time = getattr(config, "BUTTON_DEBOUNCE_TIME", 0.15)

            print(f"[Hardware] Initializing Servo Vào (GPIO {entry_servo_pin}) & Servo Ra (GPIO {exit_servo_pin})...")
            self.servo_entry = AngularServo(
                entry_servo_pin,
                min_angle=0,
                max_angle=180,
                min_pulse_width=min_pulse,
                max_pulse_width=max_pulse,
            )
            self.servo_exit = AngularServo(
                exit_servo_pin,
                min_angle=0,
                max_angle=180,
                min_pulse_width=min_pulse,
                max_pulse_width=max_pulse,
            )

            # Move both servos to closed position initially, then detach PWM to stop jitter
            closed_entry = getattr(config, "BARRIER_ENTRY_CLOSED_ANGLE", 0)
            closed_exit = getattr(config, "BARRIER_EXIT_CLOSED_ANGLE", 0)
            move_time = getattr(config, "BARRIER_MOVE_TIME", 0.8)

            self.servo_entry.angle = closed_entry
            self.servo_exit.angle = closed_exit
            time.sleep(move_time)
            self.servo_entry.detach()
            self.servo_exit.detach()

            print(f"[Hardware] Initializing Nút Vào (GPIO {entry_btn_pin}) & Nút Ra (GPIO {exit_btn_pin})...")
            self.btn_entry = Button(entry_btn_pin, pull_up=True, bounce_time=bounce_time)
            self.btn_exit = Button(exit_btn_pin, pull_up=True, bounce_time=bounce_time)

            # Register button click event handlers
            self.btn_entry.when_pressed = self._on_entry_button_pressed
            self.btn_exit.when_pressed = self._on_exit_button_pressed

            self.is_hardware_available = True
            print("[Hardware] ✓ GPIO SG90 Servos & Buttons initialized successfully (Raspberry Pi 5)!")

        except Exception as e:
            print(f"[Hardware Warning] Cannot initialize GPIO hardware ({e}). Running in SIMULATION mode.")
            self.is_hardware_available = False

    def _on_entry_button_pressed(self) -> None:
        """Callback executed when Entry Button (GPIO 23) is physically pressed."""
        now = time.time()
        if now - self._last_entry_press < self._debounce_window:
            return
        self._last_entry_press = now

        print("\n[BUTTON EVENT] >>> NÚT NHẤN VÀO (GPIO 23) ĐÃ BẤM -> KÍCH HOẠT CHECK-IN <<<")
        if self.action_callback:
            try:
                self.action_callback("checkin")
            except Exception as e:
                print(f"[Button Callback Error] {e}")
        else:
            self.action_queue.put("checkin")

    def _on_exit_button_pressed(self) -> None:
        """Callback executed when Exit Button (GPIO 24) is physically pressed."""
        now = time.time()
        if now - self._last_exit_press < self._debounce_window:
            return
        self._last_exit_press = now

        print("\n[BUTTON EVENT] >>> NÚT NHẤN RA (GPIO 24) ĐÃ BẤM -> KÍCH HOẠT CHECK-OUT <<<")
        if self.action_callback:
            try:
                self.action_callback("checkout")
            except Exception as e:
                print(f"[Button Callback Error] {e}")
        else:
            self.action_queue.put("checkout")

    def pop_action(self) -> Optional[str]:
        """Fetch next pending action from hardware queue."""
        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None

    def _move_servo_smooth(self, servo, target_angle: float, servo_name: str) -> None:
        """Move servo to angle and detach after movement to stop jitter/humming and conserve power."""
        if not self.is_hardware_available or servo is None:
            return

        def _worker():
            try:
                # Clamp angle to [0, 180]
                clamped_angle = max(0.0, min(180.0, float(target_angle)))
                servo.angle = clamped_angle
                # Wait for physical arm movement to complete
                move_delay = getattr(config, "BARRIER_MOVE_TIME", 0.8)
                time.sleep(move_delay)
                # Detach PWM signal so SG90 does NOT vibrate, hum, or overheat
                servo.detach()
            except Exception as e:
                print(f"[Hardware Warning] Error moving {servo_name}: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def open_entry_barrier(self, auto_close_delay: Optional[float] = None) -> None:
        """Open Entry Barrier (Vào) and schedule auto-close after delay."""
        if auto_close_delay is None:
            auto_close_delay = getattr(config, "BARRIER_AUTO_CLOSE_DELAY", 5.0)

        with self._entry_lock:
            if self._entry_timer is not None:
                self._entry_timer.cancel()
                self._entry_timer = None

            open_angle = getattr(config, "BARRIER_ENTRY_OPEN_ANGLE", 90)
            print(f"[BARRIER] >> MỞ BARRIER VÀO (Góc {open_angle}°). Tự đóng sau {auto_close_delay}s.")
            self.entry_state = "OPEN"
            self._move_servo_smooth(self.servo_entry, open_angle, "Servo Vào (Entry)")

            if auto_close_delay > 0:
                self._entry_timer = threading.Timer(auto_close_delay, self.close_entry_barrier)
                self._entry_timer.daemon = True
                self._entry_timer.start()

    def close_entry_barrier(self) -> None:
        """Close Entry Barrier (Vào) immediately."""
        with self._entry_lock:
            if self._entry_timer is not None:
                self._entry_timer.cancel()
                self._entry_timer = None

            closed_angle = getattr(config, "BARRIER_ENTRY_CLOSED_ANGLE", 0)
            print(f"[BARRIER] >> ĐÓNG BARRIER VÀO (Góc {closed_angle}°).")
            self.entry_state = "CLOSED"
            self._move_servo_smooth(self.servo_entry, closed_angle, "Servo Vào (Entry)")

    def open_exit_barrier(self, auto_close_delay: Optional[float] = None) -> None:
        """Open Exit Barrier (Ra) and schedule auto-close after delay."""
        if auto_close_delay is None:
            auto_close_delay = getattr(config, "BARRIER_AUTO_CLOSE_DELAY", 5.0)

        with self._exit_lock:
            if self._exit_timer is not None:
                self._exit_timer.cancel()
                self._exit_timer = None

            open_angle = getattr(config, "BARRIER_EXIT_OPEN_ANGLE", 90)
            print(f"[BARRIER] >> MỞ BARRIER RA (Góc {open_angle}°). Tự đóng sau {auto_close_delay}s.")
            self.exit_state = "OPEN"
            self._move_servo_smooth(self.servo_exit, open_angle, "Servo Ra (Exit)")

            if auto_close_delay > 0:
                self._exit_timer = threading.Timer(auto_close_delay, self.close_exit_barrier)
                self._exit_timer.daemon = True
                self._exit_timer.start()

    def close_exit_barrier(self) -> None:
        """Close Exit Barrier (Ra) immediately."""
        with self._exit_lock:
            if self._exit_timer is not None:
                self._exit_timer.cancel()
                self._exit_timer = None

            closed_angle = getattr(config, "BARRIER_EXIT_CLOSED_ANGLE", 0)
            print(f"[BARRIER] >> ĐÓNG BARRIER RA (Góc {closed_angle}°).")
            self.exit_state = "CLOSED"
            self._move_servo_smooth(self.servo_exit, closed_angle, "Servo Ra (Exit)")

    def close_all(self) -> None:
        """Close both entry and exit barriers."""
        self.close_entry_barrier()
        self.close_exit_barrier()

    def get_status(self) -> Dict[str, Any]:
        """Return real-time hardware status dictionary."""
        entry_btn_val = False
        exit_btn_val = False
        if self.btn_entry is not None:
            try:
                entry_btn_val = bool(self.btn_entry.is_pressed)
            except Exception:
                pass
        if self.btn_exit is not None:
            try:
                exit_btn_val = bool(self.btn_exit.is_pressed)
            except Exception:
                pass

        return {
            "hardware_enabled": self.is_hardware_available,
            "entry_barrier": self.entry_state,
            "exit_barrier": self.exit_state,
            "entry_button_pressed": entry_btn_val,
            "exit_button_pressed": exit_btn_val,
        }

    def cleanup(self) -> None:
        """Gracefully close hardware resources and release GPIO pins."""
        print("[Hardware] Cleaning up BarrierController GPIO resources...")
        if self._entry_timer:
            self._entry_timer.cancel()
        if self._exit_timer:
            self._exit_timer.cancel()

        if self.is_hardware_available:
            try:
                # Move to closed position on shutdown
                closed_entry = getattr(config, "BARRIER_ENTRY_CLOSED_ANGLE", 0)
                closed_exit = getattr(config, "BARRIER_EXIT_CLOSED_ANGLE", 0)
                if self.servo_entry:
                    self.servo_entry.angle = closed_entry
                if self.servo_exit:
                    self.servo_exit.angle = closed_exit
                time.sleep(0.5)

                if self.servo_entry:
                    self.servo_entry.detach()
                    self.servo_entry.close()
                if self.servo_exit:
                    self.servo_exit.detach()
                    self.servo_exit.close()
                if self.btn_entry:
                    self.btn_entry.close()
                if self.btn_exit:
                    self.btn_exit.close()
            except Exception as e:
                print(f"[Hardware Warning] Cleanup error: {e}")

        print("[Hardware] ✓ Barrier GPIO cleanup completed.")
