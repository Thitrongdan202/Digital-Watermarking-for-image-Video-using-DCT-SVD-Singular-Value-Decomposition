# Hướng dẫn cài đặt hệ thống Digital Watermarking (DCT–SVD)

Tài liệu này chỉ tập trung vào **hướng dẫn cài đặt & chạy** ứng dụng.  
(Ứng dụng có 3 tab: **EMBED / EXTRACT / DETECT**)

---

## 1) Yêu cầu

- **Python 3.9 – 3.12** (khuyến nghị **3.12**)
- Hệ điều hành: Windows / macOS / Linux
- Dung lượng trống vài trăm MB để cài thư viện

> **Khuyến nghị dùng PySide6** (ổn định hơn PyQt5 trên Windows).  
> Nếu gặp lỗi DLL với PyQt5, hãy chuyển sang PySide6.

---

## 2) Chuẩn bị mã nguồn

Đặt các file sau **cùng một thư mục** (project root):

```
dct_svd_core.py
app_dct_svd.py            # Bản dùng PyQt5
app_dct_svd_pyside6.py    # Bản dùng PySide6 (khuyến nghị)
```

> Nếu  chỉ dùng PySide6, có thể chạy trực tiếp `app_dct_svd_pyside6.py`.

---

## 3) Tạo môi trường ảo & cài thư viện

### Windows (PowerShell hoặc CMD)
```bat
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip wheel
pip install numpy opencv-python PySide6
```

### macOS / Linux (Terminal)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel
pip install numpy opencv-python PySide6
```

> **Tuỳ chọn (nếu muốn dùng PyQt5 thay PySide6):**
> ```bash
> pip install --only-binary=:all: PyQt5==5.15.10 PyQt5-sip==12.13.0
> ```

---

## 4) Chạy ứng dụng

### Khuyến nghị (PySide6)
```bash
python app_dct_svd_pyside6.py
```

### Nếu đã cài PyQt5
```bash
python app_dct_svd.py
```

> Ứng dụng sẽ mở giao diện gồm 3 tab: **EMBED / EXTRACT / DETECT**.

---

## 5) Kiểm tra nhanh (tuỳ chọn)

Kiểm tra OpenCV đã cài và phiên bản:

- **Windows (PowerShell/CMD):**
  ```bat
  python -c "import cv2, numpy as np; print('OpenCV:', cv2.__version__)"
  ```

- **macOS/Linux:**
  ```bash
  python -c 'import cv2, numpy as np; print("OpenCV:", cv2.__version__)'
  ```

---

## 6) Khắc phục lỗi thường gặp

- **ImportError: QtCore / lỗi DLL PyQt5 trên Windows**  
→ Dùng **PySide6** (cài `pip install PySide6`) và chạy:  
  ```bash
  python app_dct_svd_pyside6.py
  ```

- **Màn hình đen / không lên cửa sổ trên Linux**  
→ Cài thêm thư viện GUI hệ thống (ví dụ Ubuntu):  
  ```bash
  sudo apt-get update
  sudo apt-get install -y libgl1 libxkbcommon-x11-0
  ```

- **Cài pip thất bại**  
→ Nâng cấp pip/wheel trước:
  ```bash
  python -m pip install --upgrade pip wheel
  ```

- **Thiếu quyền**  
→ Tránh dùng `sudo pip`; hãy dùng **venv** như hướng dẫn ở trên.

---

## 7) Gợi ý sử dụng 

- **EMBED:** chọn ảnh gốc → chọn loại payload (Image/Text/JSON) → chỉnh **Alpha** → **EMBED WATERMARK**  
- **EXTRACT:** chọn `*_stego.png` (để trống meta cũng được, app sẽ tự dò) → **EXTRACT**  
- **DETECT:** chọn `*_stego.png` → chỉnh ngưỡng **Thresh NC** → **DETECT PRESENCE**

