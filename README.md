
# Hệ thống Digital Watermarking cho Ảnh dùng DCT–SVD

**Chủ đề 9: Xây dựng hệ thống Digital Watermarking cho ảnh sử dụng DCT-SVD (Singular Value Decomposition)**  
Ứng dụng giao diện 3 tab (**EMBED / EXTRACT / DETECT**) – nhúng, khôi phục và phát hiện watermark ảnh.

---

## 1) Tổng quan

- **Phương pháp**: DCT đưa ảnh sang miền tần số, SVD phân rã ma trận thành \(U \Sigma V^T\). Ta điều chỉnh **singular values** để giấu watermark ảnh sao cho ít biến dạng (PSNR/SSIM cao) và vẫn có thể tách/kiểm tra lại.
- **Chế độ**: **Non‑blind** – cần file **meta.npz** (lưu \(S_c\), \(U_w\), \(V_w^T\), kích thước, \(\alpha\)) để trích xuất/detect chính xác.
- **Đầu vào/ra**:  
  - EMBED: ảnh gốc (host) + watermark ảnh → **stego.png** + **meta.npz**  
  - EXTRACT: **stego.png** + **meta.npz** → **watermark_khôi_phục.png**  
  - DETECT: **stego.png** + **meta.npz** → điểm **NC** + quyết định có/không.

> Phiên bản này **chỉ cho ẢNH** (image). Video không nằm trong phạm vi của chủ đề DCT–SVD thuần.

---

## 2) Tính năng chính

- Giao diện 3 tab rõ ràng: **EMBED**, **EXTRACT**, **DETECT**.
- Nhúng watermark ảnh (PNG/JPG) vào **kênh Y** (YCrCb) của ảnh host, xuất **stego.png** lossless.
- Tự tính **PSNR/SSIM** sau khi nhúng để kiểm soát chất lượng.
- Khôi phục watermark bằng file **meta.npz** (không cần ảnh gốc nguyên thủy).
- Phát hiện hiện diện watermark dựa trên **Normalized Correlation (NC)** và ngưỡng tùy chỉnh.
- Mã nguồn tách lớp rõ ràng: lõi thuật toán (`dct_svd_core.py`) và app UI (`app_dct_svd.py` / `app_dct_svd_pyside6.py`).

---

## 3) Cấu trúc thư mục (đề xuất)

```
project/
├─ dct_svd_core.py              # Thuật toán DCT–SVD (image-only)
├─ app_dct_svd.py               # Ứng dụng UI dùng PyQt5
├─ app_dct_svd_pyside6.py       # Ứng dụng UI dùng PySide6 (tuỳ chọn)
├─ README.md
└─ data/                        # (tùy chọn) chứa ảnh mẫu
```

---

## 4) Yêu cầu hệ thống

- **Python 3.10 – 3.12** (khuyến nghị 64‑bit)
- Thư viện:
  - `numpy`, `opencv-python`
  - Một trong hai:
    - `PyQt5` **hoặc**
    - `PySide6` (thay thế PyQt5, dễ chạy trên Windows)
- Windows cần **Microsoft Visual C++ Redistributable 2015–2022 (x64)**

---

## 5) Cài đặt & chạy

> Khuyến nghị dùng **virtual environment** riêng.

### Cách A – PyQt5
```bat
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy opencv-python
pip install --only-binary=:all: PyQt5==5.15.10 PyQt5-sip==12.13.0

python app_dct_svd.py
```

Nếu lỗi `ImportError: DLL load failed while importing QtCore`:
- Chạy trong **venv sạch** như trên.
- Cài **MSVC Redistributable x64**.
- Tránh xung đột Qt cũ trong PATH (Anaconda, QGIS…).

### Cách B – PySide6 (khuyến nghị nếu gặp lỗi PyQt5)
```bat
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy opencv-python PySide6

python app_dct_svd_pyside6.py
```

---

## 6) Sử dụng ứng dụng

### Tab **EMBED**
1. **Host image**: chọn ảnh gốc (PNG/JPG/BMP/TIFF).
2. **Watermark image**: chọn ảnh watermark (PNG/JPG). App sẽ **resize watermark** khớp kích thước host trong nội bộ (miền DCT).
3. **Alpha** (độ mạnh nhúng): mặc định **0.05**.  
   - Nhỏ quá → watermark yếu, khó khôi phục; lớn quá → ảnh lộ (PSNR giảm).
4. **EMBED WATERMARK** → xuất:
   - `*_stego.png` (ảnh có watermark, lưu lossless)
   - `*_meta.npz` (thông số cần thiết để extract/detect)
   - Hiển thị **PSNR** và **SSIM**.

### Tab **EXTRACT**
- Chọn **Stego image**. Meta sẽ **tự động dò** `<stego>_meta.npz` (nếu không có, bấm Browse).  
- **EXTRACT WATERMARK** → xuất `*_wm.png` và hiển thị trong khung preview.

### Tab **DETECT**
- Chọn **Stego image** + **Meta**.  
- Chỉnh **Thresh NC** (mặc định 0.60).  
- **DETECT PRESENCE** → hiển thị **NC** và kết luận ✅/❌.

---

## 7) Thuật toán (tóm tắt kỹ thuật)

1. **Nhúng**
   - Chuyển ảnh BGR → YCrCb, lấy **kênh Y** (xám).
   - DCT 2D: \( C = \mathrm{DCT}(Y) \).
   - SVD cover: \( C = U_c \, \Sigma_c \, V_c^T \).
   - Watermark xám \(W\) → \( W_m = \mathrm{DCT}(W) \), rồi \( W_m = U_w \, \Sigma_w \, V_w^T \).
   - Nhúng singular values: \( \Sigma' = \Sigma_c + \alpha \, \Sigma_w \).
   - \( C' = U_c \, \Sigma' \, V_c^T \), \( Y' = \mathrm{IDCT}(C') \) → ghép lại thành stego BGR.
   - Lưu `meta.npz` chứa: \(\Sigma_c, U_w, V_w^T, \alpha, \text{shape}\).

2. **Khôi phục**
   - Từ stego: \( C'_c = \mathrm{DCT}(Y_{\text{stego}}) \), SVD: \( U', \Sigma'_c, V'^T \).
   - Ước lượng \( \Sigma_w \approx (\Sigma'_c - \Sigma_c) / \alpha \).
   - Tái tạo watermark miền DCT: \( W_m \approx U_w \, \Sigma_w \, V_w^T \) → **IDCT** → watermark.

3. **Phát hiện (NC)**  
   - Dựa vào biểu diễn phổ của watermark ước lượng. Tính **Normalized Correlation (NC)** và so ngưỡng → kết luận.

---

## 8) Gợi ý tham số & đánh giá

- **Alpha**: 0.03–0.08 là vùng “đẹp”; 0.05 mặc định.
- **Chỉ số**:
  - **PSNR** ≥ 38–40 dB → ảnh hầu như không đổi với mắt thường.
  - **SSIM** ≥ 0.95 → cấu trúc giữ tốt.
  - **NC** (detect) ≥ 0.6–0.7 thường coi là có watermark (tùy dữ liệu).

---

## 9) Kịch bản demo nhanh (đề xuất)

1. EMBED với `alpha=0.05` → báo **PSNR/SSIM**.  
2. Mở **stego.png** bằng phần mềm bất kỳ (xem mắt thường).  
3. EXTRACT → hiển thị watermark khôi phục.  
4. DETECT → báo **NC** ≥ ngưỡng.  
5. (Tuỳ chọn) Giảm **alpha** để so sánh chất lượng và khả năng khôi phục.

---

## 10) Câu hỏi thường gặp

- **Vì sao cần `meta.npz`?**  
  Là hệ thống **non‑blind**. `meta.npz` lưu \( \Sigma_c, U_w, V_w^T, \alpha \) để phục hồi watermark chính xác.
- **Có hỗ trợ video/text/json?**  
  Chủ đề 9 yêu cầu **DCT–SVD ảnh** → bản này chỉ xử lý ảnh. (Có thể mở rộng sau.)
- **Watermark bị mờ?**  
  Tăng \( \alpha \) nhẹ; chọn watermark có **tương phản** tốt; đảm bảo host không quá nhỏ/mờ.

---

## 11) Tham khảo & ghi chú

- Mã nguồn lõi: `dct_svd_core.py`
- App PyQt5: `app_dct_svd.py`
- App PySide6: `app_dct_svd_pyside6.py` (khuyên dùng nếu gặp lỗi DLL Qt trên Windows)


