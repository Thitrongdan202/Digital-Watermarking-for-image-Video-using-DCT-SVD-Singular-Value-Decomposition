# Xây dựng hệ thống Digital Watermarking cho ảnh bằng DCT–SVD (Singular Value Decomposition)

Hệ thống thực hiện **nhúng (EMBED)**, **tách (EXTRACT)** và **phát hiện (DETECT)** watermark trên **ảnh tĩnh** dựa trên **DCT + SVD**.  
Không cần ảnh gốc khi extract/detect, **nhưng cần file meta `.npz`** sinh ra ở bước embed.

---

## Tính năng chính

- **Watermark ảnh**: hỗ trợ **đen-trắng** *và* **màu RGB** (bật tùy chọn *Color watermark* khi embed).
- **Watermark dữ liệu TEXT/JSON**: tự chuyển bytes → **ảnh bit** nội bộ trước khi nhúng (pipeline trên kênh Y).
- **Giữ nguyên kích thước watermark gốc khi EXTRACT** (không resize ngược) — áp dụng cho watermark ảnh.
- Tính **PSNR/SSIM** sau khi embed để đánh giá chất lượng.
- **Detect** sự hiện diện của watermark (điểm NC-like) từ **stego + meta**, không cần ảnh gốc.

> **Non-blind theo meta**: tách/phát hiện không cần ảnh gốc, **nhưng bắt buộc** có đúng file `*_meta.npz`.

---

## Hướng dẫn cài đặt

### Yêu cầu
- **Python 3.9 – 3.12** (khuyến nghị **3.12**)
- Hệ điều hành: Windows / macOS / Linux
- Dung lượng trống vài trăm MB để cài thư viện

> Khuyến nghị dùng **PySide6** (ổn định hơn PyQt5 trên Windows).  
> Nếu bạn gặp lỗi DLL với PyQt5, hãy chuyển sang PySide6.

---

### 1) Tải mã nguồn
Đặt các file sau **cùng một thư mục**:
dct_svd_core.py
app_dct_svd.py # Bản dùng PyQt5
app_dct_svd_pyside6.py # Bản dùng PySide6 (khuyến nghị)


---

### 2) Tạo môi trường ảo & cài thư viện

#### Windows (PowerShell / CMD)
```bat
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip wheel
pip install numpy opencv-python PySide6


#### MacOS - Linux (Terminal)
```bat
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel
pip install numpy opencv-python PySide6


