# 🚗 Hệ Thống Quản Lý Bãi Đỗ Xe Thông Minh (Phiên Bản Raspberry Pi 5)
### Smart Parking Management & Vehicle Localization System - Raspberry Pi 5 Edition

<p align="center">
  <b>Tối ưu hóa chuyên biệt cho phần cứng Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4GHz, Raspberry Pi OS 64-bit Bookworm)</b>
</p>

---

## 📌 Giới thiệu & Điểm Tối Ưu Cho Raspberry Pi 5

Phiên bản này được thiết kế và tối ưu riêng cho **Raspberry Pi 5**, tích hợp trọn vẹn 3 phân hệ cốt lõi:
1. 🔍 **Nhận diện Biển số xe (ALPR)**: Phát hiện vùng biển số (YOLOv5 Nano) + Engine OCR siêu tốc dựa trên **`fast-plate-ocr` (ONNX Runtime)** cho phép trích xuất ký tự biển số tức thì với độ chính xác cao và mức tiêu thụ tài nguyên CPU cực thấp trên Pi 5. Tự động fallback về thuật toán Deskew RAM + YOLOv5 OCR nếu cần.
2. 👤 **Nhận diện Khuôn mặt Tài xế (Face Recognition)**: InsightFace `buffalo_sc` chạy trên nền **ONNX Runtime CPU** tận dụng tối đa 4 nhân Cortex-A76 của Pi 5 để trích xuất 512-d embedding và tính Cosine Similarity tức thì.
3. 🚗 **Phát hiện & Theo dõi xe (Vehicle Tracking)**: YOLOv8 Nano + Bộ theo dõi đối tượng nhẹ (Lightweight Centroid/Velocity Tracker).
4. 🏗️ **Quản lý Cổng Tích Hợp (Smart Gate In/Out)**:
   - **Xe Vào (Check-in)**: Tự động ghi nhận biển số và chụp khuôn mặt tài xế -> Lưu phiên vào SQLite -> Mở barrier.
   - **Xe Ra (Check-out)**: Quét biển số ra -> Đối soát chéo với khuôn mặt tài xế lúc vào (Cosine Similarity). Nếu khớp: Cho phép ra và tính thời gian đỗ xe; nếu không khớp: Cảnh báo gian lận / biển số giả.
5. 🌐 **Web Dashboard Trực Tiếp Trên Raspberry Pi 5**: Host web app (Flask + MJPEG Stream) thay thế hoàn toàn màn hình HDMI. Cho phép thiết bị trong cùng mạng LAN (laptop, điện thoại, tablet) theo dõi live camera, bản đồ 12 ô đỗ, CPU/RAM/Temp/FPS và điều khiển Check-in/out từ xa.

---

## 🌐 Hướng Dẫn Truy Cập Web Dashboard Từ Trình Duyệt

khi hệ thống chạy, Web Dashboard tự động mở tại cổng `5000`:

1. **Tìm địa chỉ IP của Raspberry Pi 5 trong mạng LAN**:
   ```bash
   hostname -I
   # Ví dụ đầu ra: 192.168.1.150
   ```
2. **Mở trình duyệt trên máy tính / điện thoại / tablet cùng mạng LAN**:
   ```
   http://192.168.1.150:5000
   ```
3. **Các tính năng trên Dashboard**:
   - 📷 **Live Cameras**: Luồng MJPEG realtime từ Camera Tracking (Cam 0) và Camera Gate (Cam 1).
   - 🅿️ **Parking Lot Map**: Bản đồ trực quan 4 Khu (A, B, C, D) & 12 ô đỗ (A1..D3) cập nhật màu xanh (trống) / đỏ (đang đỗ) theo thời gian thực.
   - ⚡ **System Telemetry**: Theo dõi chỉ số phần cứng (CPU %, RAM %, Nhiệt độ CPU, AI FPS, trạng thái camera).
   - 🔘 **Remote Controls**: Kích hoạt từ xa nút Check-in (Xe Vào), Check-out (Xe Ra), Quét mắt Tìm Xe.

---


## ⚙️ Yêu cầu Phần cứng Khuyến nghị

- **Bo mạch**: Raspberry Pi 5 (bản 4GB hoặc 8GB RAM).
- **Hệ điều hành**: Raspberry Pi OS 64-bit (Debian 12 Bookworm).
- **Tản nhiệt**: Bắt buộc trang bị **Raspberry Pi Active Cooler** để đảm bảo CPU luôn mát mẻ và duy trì xung nhịp đỉnh 2.4GHz.
- **Phần cứng Barrier & Nút Nhấn**:
  - 2x Động cơ Servo 9G SG90 (180°) cho cổng Vào và cổng Ra.
  - 2x Nút nhấn cơ học (Push button NO/COM).
  - Nguồn 5V ngoài (hoặc nguồn cấp riêng) cho Servo (nối chung mass GND với Pi).

---

## 🔌 Sơ Đồ Đấu Nối Chân Vật Lý (Raspberry Pi 5 Pinout)

Hệ thống sử dụng trực tiếp các chân GPIO trên bo mạch Raspberry Pi 5 theo sơ đồ chuẩn:

| Thiết bị | Dây / Chân linh kiện | Tên GPIO (BCM) | Số Chân Vật Lý Pi 5 | Mô tả chức năng |
|---|---|---|---|---|
| 🚧 **Servo Vào** | Tín hiệu (Dây Cam/Vàng) | **GPIO 5** | **Chân 29** | Mở/Đóng Barrier Xe Vào (0° - 90°) |
| 🚧 **Servo Ra** | Tín hiệu (Dây Cam/Vàng) | **GPIO 6** | **Chân 31** | Mở/Đóng Barrier Xe Ra (0° - 90°) |
| 🔘 **Nút Nhấn Vào** | Chân NO | **GPIO 23** | **Chân 16** | Ấn nút để quét khuôn mặt + biển số Check-in |
| 🔘 **Nút Nhấn Ra** | Chân NO | **GPIO 24** | **Chân 18** | Ấn nút để quét khuôn mặt + đối soát Check-out |
| ⏚ **GND Servo** | Dây Nâu / Đen | GND | **Chân 30 / 34** | Chân mass cho 2 Servo |
| ⏚ **GND Nút Nhấn** | Chân COM của nút | GND | **Chân 14 / 20** | Kéo GPIO xuống GND khi ấn nút |
| ⚡ **GND Chung Nguồn**| Dây Mass nguồn ngoài | GND | **Chân 39** | **Cực kỳ quan trọng**: Nối chung GND giữa Pi và Nguồn 5V của Servo |

> [!TIP]
> **Chống rung rè (Anti-jitter & Power-saving)**: Hệ thống tự động kích hoạt chế độ `detach PWM` sau khi servo quay xong góc mục tiêu (0.8s). Điều này ngăn ngừa hoàn toàn tình trạng servo SG90 bị rè, rung, nóng hoặc sụt áp bo mạch khi giữ vị trí.

---

## 🛠️ Công Cụ Kiểm Tra & Căn Chỉnh Phần Cứng Nhanh

Trước khi chạy toàn bộ hệ thống AI, bạn có thể kiểm tra riêng 2 servo và 2 nút nhấn bằng công cụ tương tác:

```bash
# Kích hoạt venv (nếu có) và chạy công cụ test:
python3 test_barrier_hardware.py
```

Menu hỗ trợ kiểm tra trực quan:
- `[1]` Test Servo Vào (GPIO 5): Xoay mở 90° -> Giữ 3s -> Tự đóng 0°.
- `[2]` Test Servo Ra (GPIO 6): Xoay mở 90° -> Giữ 3s -> Tự đóng 0°.
- `[3]` Test Nút Nhấn (GPIO 23 & 24): Lắng nghe nút nhấn theo thời gian thực.
- `[4]` Test Quy trình tự động: Ấn nút -> Mở barrier 5s -> Tự đóng.
- `[5]` Căn chỉnh góc Servo: Nhập góc tùy ý (0° - 180°) để tìm vị trí thanh chắn đóng/mở chuẩn nhất cho cơ cấu của bạn.

---

## 🚀 Hướng Dẫn Cài Đặt (Toàn Bộ Câu Lệnh Chi Tiết)

### Cách 1: Cài đặt tự động bằng Script 1-Click (Khuyên dùng)

Mở Terminal trên Raspberry Pi 5 và chạy:

```bash
cd smart_parking_rpi5
chmod +x setup_rpi5.sh
./setup_rpi5.sh
```

---

### Cách 2: Cài đặt thủ công từng bước qua Terminal

#### Bước 1: Cập nhật hệ điều hành & cài đặt gói phụ thuộc hệ thống
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    libcap-dev \
    ffmpeg \
    libopenblas-dev \
    v4l-utils \
    libcamera-tools \
    git \
    wget \
    unzip
```

#### Bước 2: Tạo và kích hoạt Môi trường ảo Python (Virtual Environment)
*Lưu ý: Trên Raspberry Pi OS Bookworm, bắt buộc phải dùng virtual environment (chuẩn PEP 668):*

```bash
cd smart_parking_rpi5

# Tạo môi trường ảo kế thừa system-site-packages
python3 -m venv .venv --system-site-packages

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Nâng cấp công cụ pip
pip install --upgrade pip setuptools wheel
```

#### Bước 3: Cài đặt các thư viện Python
```bash
pip install -r requirements.txt
```

#### Bước 4: Tải và kiểm tra mô hình AI
Chạy script kiểm tra và tải trọng số InsightFace `buffalo_sc`:
```bash
python3 models/download_models.py
```

---

## 🖥️ Hướng Dẫn Chạy Chương Trình

> **Quan trọng**: Luôn kích hoạt môi trường ảo trước khi chạy:
> ```bash
> cd smart_parking_rpi5
> source .venv/bin/activate
> ```

### 1. Chạy Ứng Dụng Quản Lý Cổng Bãi Xe Tích Hợp (Khuyên dùng)

Ứng dụng đầy đủ kết hợp cả Biển số xe + Khuôn mặt tài xế + Lưu cơ sở dữ liệu:

```bash
# Chế độ có màn hình GUI hiển thị Dashboard
python3 main_parking_system.py

# Hoặc chế độ Headless (khi chạy từ xa qua SSH không có màn hình)
python3 main_parking_system.py --headless
```

**Bảng phím tắt điều khiển trên giao diện:**
| Phím | Chức năng | Mô tả chi tiết |
| :---: | :--- | :--- |
| `I` hoặc `R` | **Check-in (Xe Vào)** | Quét biển số xe + chụp khuôn mặt tài xế, lưu phiên vào CSDL SQLite và mở barrier vào. |
| `O` hoặc `X` | **Check-out (Xe Ra)** | Quét biển số ra, tự động so khớp khuôn mặt với lúc vào (Cosine Similarity). Nếu khớp: Đóng phiên, tính thời gian gửi và mở barrier ra. |
| `Q` | **Thoát (Quit)** | Dừng camera an toàn và đóng kết nối cơ sở dữ liệu. |

---

### 2. Chạy Demo Độc Lập Từng Phân Hệ

#### A. Phân hệ Nhận diện Biển số xe (ALPR)
```bash
# Chạy nhận diện trực tiếp qua Camera USB / Pi Camera
python3 demo_plate.py

# Hoặc chạy kiểm tra trên 1 bức ảnh cụ thể
python3 demo_plate.py --image test_images/1.jpg
```

#### B. Phân hệ Nhận diện Khuôn mặt Tài xế
```bash
# Chạy giao diện nhận diện khuôn mặt tài xế (Phím R: Vào, C: Check, X: Ra)
python3 demo_face.py
```

#### C. Phân hệ Phát hiện & Theo dõi Xe trong Bãi (Tracking)
```bash
# Chạy tracking phương tiện từ camera hoặc video
python3 demo_tracking.py --source 0
python3 demo_tracking.py --source /path/to/video.mp4
```

---

## 🛠️ Cấu Hình Hệ Thống (`config.py`)

Tất cả các tham số có thể dễ dàng điều chỉnh trong file [config.py](file:///c:/Users/ba_dat/Documents/1.%20Do%20An%20Tot%20Nghiep/smart_parking_rpi5/config.py):

- `CAMERA_SOURCE = 0`: Chỉ số Camera USB (hoặc đường dẫn RTSP stream).
- `USE_PICAMERA2 = True / False`: Đặt `True` nếu dùng camera ribbon chính hãng của Raspberry Pi qua thư viện `picam2`.
- `FACE_MATCH_THRESHOLD = 0.45`: Ngưỡng độ tương đồng Cosine để xác thực đúng khuôn mặt tài xế lúc ra.
- `PLATE_CONF_THRESHOLD = 0.50`: Ngưỡng tin cậy phát hiện biển số xe.
- `ONNX_NUM_THREADS = 4`: Số luồng CPU phân bổ cho ONNX Runtime (tối ưu cho 4 nhân Cortex-A76 của Pi 5).
- `ENABLE_GUI = True`: Đặt `False` nếu bạn muốn chạy hệ thống ở chế độ nền (Background Service / Headless).

---

## 📁 Cấu Trúc Thư Mục `smart_parking_rpi5/`

```
smart_parking_rpi5/
├── README.md                      # Tài liệu hướng dẫn chi tiết (File này)
├── requirements.txt               # Danh sách thư viện Python tối ưu cho Pi 5
├── setup_rpi5.sh                  # Script cài đặt tự động 1-click
├── config.py                      # Cấu hình trung tâm (Camera, Threshold, Threads)
├── database.py                    # Module SQLite lưu trữ session, embedding, thời gian
│
├── core/                          # Các engine AI & Computer Vision cốt lõi
│   ├── __init__.py
│   ├── camera_stream.py           # Luồng camera đa luồng (USB Cam, Picamera2, RTSP)
│   ├── face_engine.py             # Engine InsightFace buffalo_sc (ONNX Runtime CPU)
│   ├── plate_engine.py            # Engine ALPR (YOLOv5 nano plate + Deskew RAM + OCR)
│   ├── vehicle_tracker.py         # Engine phát hiện & tracking xe (YOLOv8 nano)
│   └── utils_rotate.py            # Thuật toán deskew xoay phẳng biển số xe
│
├── models/                        # Thư mục chứa trọng số mô hình
│   ├── LP_detector_nano_61.pt     # Trọng số phát hiện biển số (Nano 3.7MB)
│   ├── LP_ocr_nano_62.pt          # Trọng số nhận diện ký tự (Nano 4.0MB)
│   ├── best.pt                    # Trọng số nhận diện xe (5.3MB)
│   └── download_models.py         # Script tự động tải InsightFace buffalo_sc
│
├── web/                           # Phân hệ Web Dashboard (Flask + HTML5 + CSS + JS)
│   ├── app.py                     # Flask server routes, MJPEG streaming & WebState
│   ├── templates/
│   │   └── index.html             # Giao diện Web Dashboard (Glassmorphism Dark Mode)
│   └── static/
│       ├── css/dashboard.css      # CSS styling & responsive layout
│       └── js/dashboard.js        # JavaScript polling client & interactive controls
│
├── yolov5/                        # Mã nguồn YOLOv5 inference engine cục bộ
├── test_images/                   # Ảnh mẫu để kiểm thử ALPR
├── main_parking_system.py         # Ứng dụng chính: Cổng bãi đỗ xe thông minh (Vào/Ra + Web Server)
├── demo_face.py                   # Script test riêng nhận diện khuôn mặt
├── demo_plate.py                  # Script test riêng nhận diện biển số
└── demo_tracking.py               # Script test riêng theo dõi xe
```


---

## ❓ Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 1. Lỗi không mở được camera (`Khong mo duoc camera /dev/video0`)
- **Kiểm tra danh sách camera khả dụng**:
  ```bash
  v4l2-ctl --list-devices
  ```
- **Cấp quyền truy cập camera cho user**:
  ```bash
  sudo usermod -a -G video $USER
  ```
- Sau đó đổi `CAMERA_SOURCE = 0` hoặc `1` trong `config.py`.

### 2. Lỗi `externally-managed-environment` khi chạy pip
- Đây là cơ chế bảo vệ của Debian 12 Bookworm. Hãy đảm bảo bạn đã kích hoạt môi trường ảo:
  ```bash
  source .venv/bin/activate
  ```

### 3. Khi chạy SSH không có màn hình bị báo lỗi `cv2.error: (-215) size.width>0` hoặc lỗi X11
- Thêm cờ `--headless` khi chạy ứng dụng:
  ```bash
  python3 main_parking_system.py --headless
  ```
- Hoặc trong `config.py`, đặt `ENABLE_GUI = False`.

### 4. Tối ưu nhiệt độ & xung nhịp
- Khuyến nghị kiểm tra nhiệt độ Pi 5 khi chạy liên tục:
  ```bash
  vcgencmd measure_temp
  ```
- Với quạt Active Cooler, nhiệt độ khi suy luận AI liên tục trên Pi 5 sẽ duy trì ổn định ở mức 45°C - 58°C.

---

## 🙏 Tri Ân & Tín Nhiệm Mã Nguồn Mở (Credits & Acknowledgments)

Dự án này sử dụng và chân thành cảm ơn tác giả cũng như các cộng đồng mã nguồn mở sau:

- 📖 **[fast-plate-ocr](https://github.com/pysolotravel/fast-plate-ocr)** bởi **[pysolotravel](https://github.com/pysolotravel)**: Thư viện OCR biển số xe tối ưu trên nền ONNX Runtime cho tốc độ nhận diện tức thì và độ chính xác cực cao.
- 👁️ **[InsightFace](https://github.com/deepinsight/insightface)**: Thư viện phân tích và nhận diện khuôn mặt hàng đầu.
- ⚡ **[YOLOv5](https://github.com/ultralytics/yolov5)** & **[YOLOv8](https://github.com/ultralytics/ultralytics)** bởi Ultralytics: Mô hình phát hiện đối tượng thời gian thực.
- 📦 **[ONNX Runtime](https://onnxruntime.ai/)** bởi Microsoft: Inference engine tối ưu hóa đa luồng trên CPU ARM Cortex-A76.

