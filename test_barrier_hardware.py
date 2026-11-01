#!/usr/bin/env python3
"""
Interactive Hardware Test & Calibration Tool for SG90 Servos & Push Buttons.
Designed for Raspberry Pi 5 Smart Parking System.

Pinout Check:
  - Pin 29: GPIO 5  -> Servo Vào (SG90 9G Signal)
  - Pin 31: GPIO 6  -> Servo Ra  (SG90 9G Signal)
  - Pin 16: GPIO 23 -> Nút Nhấn Vào (NO -> Pin 16, COM -> Pin 14 GND)
  - Pin 18: GPIO 24 -> Nút Nhấn Ra  (NO -> Pin 18, COM -> Pin 20 GND)
  - Pin 30/34/39: GND chung
"""

import sys
import time
import warnings

# Suppress software PWM warnings on Raspberry Pi
try:
    from gpiozero.exc import PWMSoftwareFallback
    warnings.filterwarnings("ignore", category=PWMSoftwareFallback)
except ImportError:
    pass

import config
from core.barrier_controller import BarrierController

def print_header():
    print("\n" + "=" * 65)
    print("  KIỂM TRA & CĂN CHỈNH PHẦN CỨNG SERVO SG90 & NÚT NHẤN (Pi 5)")
    print("=" * 65)
    print(f"  • Servo Vào:   GPIO {config.SERVO_ENTRY_PIN} (Chân 29) | Góc Đóng={config.BARRIER_ENTRY_CLOSED_ANGLE}°, Mở={config.BARRIER_ENTRY_OPEN_ANGLE}°")
    print(f"  • Servo Ra:    GPIO {config.SERVO_EXIT_PIN} (Chân 31) | Góc Đóng={config.BARRIER_EXIT_CLOSED_ANGLE}°, Mở={config.BARRIER_EXIT_OPEN_ANGLE}°")
    print(f"  • Nút Nhấn Vào: GPIO {config.BUTTON_ENTRY_PIN} (Chân 16 - GND Chân 14)")
    print(f"  • Nút Nhấn Ra:  GPIO {config.BUTTON_EXIT_PIN} (Chân 18 - GND Chân 20)")
    print("=" * 65)


def test_servo_entry(controller: BarrierController):
    print("\n>>> [1] TEST SERVO VÀO (GPIO 5)...")
    print("  -> Đang mở barrier vào (90°)...")
    controller.open_entry_barrier(auto_close_delay=3.0)
    print("  -> Giữ mở trong 3 giây...")
    time.sleep(3.5)
    print("  -> Đã tự động đóng barrier vào (0°).")
    print("  ✓ Hoàn thành test Servo Vào!")


def test_servo_exit(controller: BarrierController):
    print("\n>>> [2] TEST SERVO RA (GPIO 6)...")
    print("  -> Đang mở barrier ra (90°)...")
    controller.open_exit_barrier(auto_close_delay=3.0)
    print("  -> Giữ mở trong 3 giây...")
    time.sleep(3.5)
    print("  -> Đã tự động đóng barrier ra (0°).")
    print("  ✓ Hoàn thành test Servo Ra!")


def test_buttons_listener(controller: BarrierController):
    print("\n>>> [3] TEST NÚT NHẤN REAL-TIME (GPIO 23 & GPIO 24)...")
    print("  -> Vui lòng ấn thử Nút Vào hoặc Nút Ra trên mạch.")
    print("  -> Nhấn phím 'Enter' hoặc 'Ctrl + C' bất cứ lúc nào để dừng chế độ test nút.")

    def on_press(action: str):
        if action == "checkin":
            print("  [BẤM NÚT] >>> NÚT VÀO (GPIO 23) ĐÃ BẤM! (Trigger Check-in) <<<")
        elif action == "checkout":
            print("  [BẤM NÚT] >>> NÚT RA (GPIO 24) ĐÃ BẤM! (Trigger Check-out) <<<")

    controller.action_callback = on_press

    try:
        input("\n[Đang lắng nghe nút nhấn... Nhấn Enter để kết thúc test]")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        controller.action_callback = None
        print("  ✓ Đã kết thúc chế độ test nút nhấn.")


def test_live_simulation(controller: BarrierController):
    print("\n>>> [4] TEST QUY TRÌNH BARRIER TỰ ĐỘNG VỚI NÚT NHẤN...")
    print("  -> Bấm Nút Vào (GPIO 23) -> Barrier Vào mở 5s rồi tự đóng.")
    print("  -> Bấm Nút Ra (GPIO 24)  -> Barrier Ra mở 5s rồi tự đóng.")
    print("  -> Nhấn Enter để kết thúc chế độ mô phỏng.")

    def on_action(action: str):
        if action == "checkin":
            print("\n  >> Nhận tín hiệu NÚT VÀO -> Mở Barrier Vào 5s...")
            controller.open_entry_barrier(auto_close_delay=5.0)
        elif action == "checkout":
            print("\n  >> Nhận tín hiệu NÚT RA -> Mở Barrier Ra 5s...")
            controller.open_exit_barrier(auto_close_delay=5.0)

    controller.action_callback = on_action

    try:
        input("\n[Hệ thống đang chạy... Nhấn nút vật lý để thử. Nhấn Enter để dừng]")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        controller.action_callback = None
        controller.close_all()
        print("  ✓ Đã đóng các barrier và dừng mô phỏng.")


def calibrate_servo_angles(controller: BarrierController):
    print("\n>>> [5] CĂN CHỈNH GÓC SERVO SG90 (0° đến 180°)...")
    print("  Chế độ này giúp bạn tìm góc Đóng và Mở hoàn hảo nhất cho thanh chắn.")
    
    target = input("  Chọn servo cần căn chỉnh ('1' = Servo Vào, '2' = Servo Ra): ").strip()
    servo = controller.servo_entry if target == "1" else controller.servo_exit
    servo_name = "Servo Vào (GPIO 5)" if target == "1" else "Servo Ra (GPIO 6)"

    if not servo:
        print(f"  [Lỗi] {servo_name} chưa được khởi tạo phần cứng!")
        return

    print(f"\n  Đang điều khiển {servo_name}.")
    print("  Nhập số góc (ví dụ: 0, 45, 90, 135, 180) hoặc 'q' để quay lại menu:")

    while True:
        val = input("  Nhập góc (0 - 180) > ").strip()
        if val.lower() in ("q", "quit", "exit", ""):
            break
        try:
            ang = float(val)
            if 0.0 <= ang <= 180.0:
                print(f"  -> Xoay {servo_name} đến {ang}°...")
                controller._move_servo_smooth(servo, ang, servo_name)
            else:
                print("  [Cảnh báo] Góc phải nằm trong khoảng từ 0 đến 180 độ.")
        except ValueError:
            print("  [Lỗi] Vui lòng nhập số hợp lệ.")


def main():
    print_header()
    print("Khởi tạo bộ điều khiển barrier...")
    controller = BarrierController()

    if not controller.is_hardware_available:
        print("\n[CẢNH BÁO] Không thể kết nối với chân GPIO trên máy này.")
        print("Nếu bạn đang test trên máy tính không phải Pi 5, hệ thống sẽ chạy ở chế độ mô phỏng.")

    try:
        while True:
            print("\n" + "-" * 45)
            print("  MENU CHỌN BÀI TEST:")
            print("  [1] Test Servo Vào (GPIO 5) -> Mở & Tự đóng")
            print("  [2] Test Servo Ra (GPIO 6)  -> Mở & Tự đóng")
            print("  [3] Test Nút Nhấn (GPIO 23 & GPIO 24)")
            print("  [4] Test Quy trình Barrier tự động (Bấm nút -> Mở 5s -> Đóng)")
            print("  [5] Căn chỉnh góc Servo SG90 (Tìm góc 0° / 90° chuẩn)")
            print("  [0] Thoát")
            print("-" * 45)

            choice = input("Lựa chọn của bạn (0-5): ").strip()

            if choice == "1":
                test_servo_entry(controller)
            elif choice == "2":
                test_servo_exit(controller)
            elif choice == "3":
                test_buttons_listener(controller)
            elif choice == "4":
                test_live_simulation(controller)
            elif choice == "5":
                calibrate_servo_angles(controller)
            elif choice in ("0", "q", "exit"):
                print("Đang thoát...")
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng chọn từ 0 đến 5.")

    finally:
        controller.cleanup()
        print("Hoàn tất dọn dẹp GPIO. Tạm biệt!")


if __name__ == "__main__":
    main()
