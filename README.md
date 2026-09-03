# 🚗 Hệ Thống Quản Lý Bãi Đỗ Xe Thông Minh (Phiên Bản Raspberry Pi 5)
### Smart Parking Management & Vehicle Localization System - Raspberry Pi 5 Edition

<p align="center">
  <b>Tối ưu hóa chuyên biệt cho phần cứng Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4GHz, Raspberry Pi OS 64-bit Bookworm)</b>
</p>

---

## 📌 Giới thiệu & Điểm Tối Ưu Cho Raspberry Pi 5

Phiên bản này được thiết kế và tối ưu riêng cho **Raspberry Pi 5**, tích hợp trọn vẹn 3 phân hệ cốt lõi:
1. 🔍 **Nhận diện Biển số xe (ALPR)**: Phát hiện vùng biển số (YOLOv5 Nano) + Thuật toán Deskew xoay phẳng + Nhận diện ký tự OCR (YOLOv5 Nano OCR). Toàn bộ ảnh crop và xoay được xử lý trực tiếp trên RAM, không ghi liên tục vào thẻ nhớ MicroSD để tránh nghẽn I/O và bảo vệ tuổi thọ thẻ nhớ.
2. 👤 **Nhận diện Khuôn mặt Tài xế (Face Recognition)**: InsightFace `buffalo_sc` chạy trên nền **ONNX Runtime CPU** tận dụng tối đa 4 nhân Cortex-A76 của Pi 5 để trích xuất 512-d embedding và tính Cosine Similarity tức thì.
3. 🚗 **Phát hiện & Theo dõi xe (Vehicle Tracking)**: YOLOv8 Nano + Bộ theo dõi đối tượng nhẹ (Lightweight Centroid/Velocity Tracker).
4. 🏗️ **Quản lý Cổng Tích Hợp (Smart Gate In/Out)**:
   - **Xe Vào (Check-in)**: Tự động ghi nhận biển số và chụp khuôn mặt tài xế -> Lưu phiên vào SQLite -> Mở barrier.
   - **Xe Ra (Check-out)**: Quét biển số ra -> Đối soát chéo với khuôn mặt tài xế lúc vào (Cosine Similarity). Nếu khớp: Cho phép ra và tính thời gian đỗ xe; nếu không khớp: Cảnh báo gian lận / biển số giả.

---

## ⚙️ Yêu cầu Phần cứng Khuyến nghị

- **Bo mạch**: Raspberry Pi 5 (bản 4GB hoặc 8GB RAM).
- **Hệ điều hành**: Raspberry Pi OS 64-bit (Debian 12 Bookworm).
- **Tản nhiệt**: Bắt buộc trang bị **Raspberry Pi Active Cooler** để đảm bảo CPU luôn mát mẻ và duy trì xung nhịp đỉnh 2.4GHz.
- **Nguồn cấp**: Nguồn chuẩn Raspberry Pi 27W USB-C PD (5V/5A).
- **Thẻ nhớ**: Thẻ MicroSD tốc độ cao (A2 / V30, dung lượng từ 32GB trở lên) hoặc ổ SSD qua Raspberry Pi M.2 HAT.
- **Camera**:
  - USB Webcam (chuẩn V4L2) cắm qua cổng USB 3.0.
  - Hoặc Raspberry Pi Camera Module v2/v3 / HQ Camera (qua cáp ribbon CSI/libcamera).

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
    libatlas-base-dev \
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
├── yolov5/                        # Mã nguồn YOLOv5 inference engine cục bộ
├── test_images/                   # Ảnh mẫu để kiểm thử ALPR
├── main_parking_system.py         # Ứng dụng chính: Cổng bãi đỗ xe thông minh (Vào/Ra)
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
