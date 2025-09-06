# DCT–SVD Watermarking (Images + Password)

> **Nhúng/giải trích ảnh + mật khẩu**. Đã **bỏ Text/JSON**. Có 2 cách chạy:
>
> 1) **Một file duy nhất** (khuyến nghị): `app_dct_svd_single.py` – không phụ thuộc core bên ngoài.
> 2) **Tách core + app** (chỉ dùng khi bạn cần tách module): `app_dct_svd_image_only_nopreview_forcecore.py` + `dct_svd_core_secure.py`

---

## 1) Cài thư viện

```bat
pip install -r requirements.txt
```

### Nếu chưa có `pip`/venv (khuyến nghị dùng venv)
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Mặc định dự án dùng `opencv-python`. Nếu hàm **`cv2.fastNlMeansDenoisingColored`** không có trên máy, hãy **gỡ `opencv-python` và cài `opencv-contrib-python`**:
```bat
pip uninstall -y opencv-python
pip install opencv-contrib-python>=4.8
```

---

## 2) Cách chạy

### Cách A – **Một file duy nhất** (đề xuất)
Không cần để core ở ngoài, không sợ nạp nhầm module.

```bat
python app_dct_svd_single.py
```

**EMBED**
1. Chọn **Host Image**, **Watermark Image**, nhập **Password**.
2. Chọn **Alpha** (khuyến nghị 0.10 → 0.18; logo mảnh có thể 0.15 → 0.22).
3. (Tuỳ chọn) tick **Color watermark (RGB)** nếu muốn nhúng màu.
4. Bấm **EMBED WATERMARK** → sinh `*_stego.png` và `*_stego_meta.npz`.

**EXTRACT**
1. Chọn **Stego**, **Meta (.npz)**, nhập **Password**.
2. Chọn nơi lưu **Out** → **EXTRACT** → xem preview watermark phía dưới.
   - Ảnh sẽ được denoise + enhance (CLAHE + unsharp).

**DETECT**
- Chọn **Stego** + **Meta** → **DETECT** (không cần mật khẩu) để xem **Score**.

---

### Cách B – **Tách core + app**
Chỉ dùng khi bạn muốn tách file:

- Để **cùng thư mục**:
  - `app_dct_svd_image_only_nopreview_forcecore.py`
  - `dct_svd_core_secure.py`
- Chạy:
  ```bat
  python app_dct_svd_image_only_nopreview_forcecore.py
  ```
- App sẽ **ép nạp core theo đúng đường dẫn** cạnh file app (không ăn nhầm core cũ trong máy).

> Nếu dùng bản app khác (`app_dct_svd.py`) thì **bắt buộc** đặt `dct_svd_core_secure.py` hoặc `dct_svd_core.py` (bản mới) **cạnh file app** và xoá thư mục `__pycache__` để tránh cache cũ.

---

## 3) Tính năng kỹ thuật
- DCT + SVD, **mid-band embedding** (lưu `kfrac` trong meta) → tăng **tính bền vững** & **chất lượng**.
- **Bảo vệ bằng mật khẩu**: khoá ngẫu nhiên (nonce) + HMAC kiểm tra meta.
- **EXTRACT**: denoise (Non-local Means) + **enhance** (CLAHE + unsharp).
- PSNR/SSIM sau embed để bạn ước lượng độ méo.
- **PNG** cho stego để tránh nén mất mát.

---

## 4) Mẹo chất lượng
- Ưu tiên **PNG** cho cover/stego.
- **Alpha** gợi ý:
  - Ảnh sáng/tối rõ, watermark đơn sắc: `0.10 – 0.16`.
  - Logo/biên mảnh hoặc muốn dễ nhìn hơn khi extract: `0.15 – 0.22`.
- Nếu watermark nhiều hạt/nhiễu, thử **không tick Color watermark** để nhúng kênh Y (xám).

---

## 5) Lỗi thường gặp & cách xử lý

### “`unexpected keyword argument 'password'`”
- Bạn đang dùng **app mới** nhưng **core cũ**. Sử dụng **Cách A (1 file)** để khỏi lệ thuộc core.
- Hoặc chắc chắn đặt **`dct_svd_core_secure.py`** (hoặc bản mới của `dct_svd_core.py`) cạnh file app và xoá `__pycache__`.

### “`The truth value of an array with more than one element is ambiguous...`”
- Đã vá trong app: đọc màu trước, nếu `None` thì mới đọc xám – không còn lỗi này.

### Preview sau EMBED vướng màn hình
- Dùng bản **no preview** (đã set sẵn) hoặc bản thường nếu muốn xem stego ngay.

---

## 6) Đóng gói EXE (tuỳ chọn)
```bat
pip install pyinstaller
pyinstaller -F -w app_dct_svd_single.py
```
- File `.exe` nằm trong thư mục `dist/`.  
- Nếu dùng PyInstaller, đảm bảo Windows có **VC++ Redistributable** (thường đã có).

---

## 7) Cấu trúc đề xuất
```
project/
 ├─ app_dct_svd_single.py                 # bản 1 file (đề xuất)
 ├─ app_dct_svd_image_only_nopreview_forcecore.py  # bản tách core (ép nạp theo path)
 ├─ dct_svd_core_secure.py                # core mới (nếu tách module)
 ├─ requirements.txt
 └─ README.md
```



