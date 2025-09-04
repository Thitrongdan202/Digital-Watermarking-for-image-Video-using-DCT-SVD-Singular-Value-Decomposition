# app_dct_svd.py
# Same 3-tab UI (EMBED / EXTRACT / DETECT), but strictly DCT–SVD for IMAGES.
# - Only watermark IMAGE is supported (PNG/JPG).
# - Video not supported .

import os, sys, json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QPlainTextEdit, QRadioButton,
    QMessageBox, QSpinBox, QTabWidget, QMainWindow
)

import cv2
import numpy as np
from dct_svd_core import embed as embed_dctsvd, extract as extract_dctsvd, detect as detect_dctsvd

APP_TITLE = "DCT–SVD Watermarking (image only)"

# ---------- Small helpers ----------
def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def first_frame_of_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return bgr

# -------------- EMBED TAB --------------
class EmbedTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Host media
        grp_host = QGroupBox("Host Media")
        lh = QHBoxLayout(grp_host)
        self.ed_host = QLineEdit()
        btn_browse_host = QPushButton("Browse")
        btn_browse_host.clicked.connect(self.on_browse_host)
        lh.addWidget(btn_browse_host, 0)
        lh.addWidget(self.ed_host, 1)

        # Preview host
        self.lb_preview = QLabel("")
        self.lb_preview.setFixedHeight(320)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444; background:#111; color:#bbb;}")
        main.addWidget(grp_host)
        main.addWidget(self.lb_preview)

        # Payload (keep UI structure, only Image used)
        grp_payload = QGroupBox("Payload")
        pv = QVBoxLayout(grp_payload)

        row = QHBoxLayout()
        self.rb_text = QRadioButton("Text")
        self.rb_json = QRadioButton("JSON")
        self.rb_img  = QRadioButton("Image")
        self.rb_img.setChecked(True)
        row.addWidget(self.rb_text); row.addWidget(self.rb_json); row.addWidget(self.rb_img)
        pv.addLayout(row)

        # text/json editor (kept but disabled)
        self.ed_payload = QPlainTextEdit()
        self.ed_payload.setPlaceholderText("DCT–SVD (ảnh) chỉ hỗ trợ watermark IMAGE.")
        self.ed_payload.setEnabled(False)
        pv.addWidget(self.ed_payload)

        # payload image path
        row2 = QHBoxLayout()
        self.ed_payload_img = QLineEdit()
        self.ed_payload_img.setPlaceholderText("Đường dẫn watermark ảnh (PNG/JPG)…")
        btn_browse_payload_img = QPushButton("Browse")
        btn_browse_payload_img.clicked.connect(self.on_browse_payload_img)
        row2.addWidget(btn_browse_payload_img, 0)
        row2.addWidget(self.ed_payload_img, 1)
        pv.addLayout(row2)

        # Settings
        grp_set = QGroupBox("Settings")
        ls = QHBoxLayout(grp_set)
        self.lb_strength = QLabel("Alpha: 0.05")
        self.sl_strength = QSlider(Qt.Horizontal)
        self.sl_strength.setMinimum(1)   # 0.01
        self.sl_strength.setMaximum(20)  # 0.20
        self.sl_strength.setValue(5)     # 0.05
        self.sl_strength.valueChanged.connect(self._on_strength_change)
        ls.addWidget(self.lb_strength)
        ls.addWidget(self.sl_strength)

        self.lb_interval = QLabel("Frame interval (video):")
        self.sp_interval = QSpinBox(); self.sp_interval.setEnabled(False) # not used
        ls.addWidget(self.lb_interval)
        ls.addWidget(self.sp_interval)

        # Output
        grp_out = QGroupBox("Output")
        lo = QHBoxLayout(grp_out)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Đường dẫn file xuất (để trống: *_stego.png và *_meta.npz)")
        btn_browse_out = QPushButton("Save As")
        btn_browse_out.clicked.connect(self.on_browse_out)
        lo.addWidget(btn_browse_out, 0)
        lo.addWidget(self.ed_out, 1)

        main.addWidget(grp_payload)
        main.addWidget(grp_set)
        main.addWidget(grp_out)

        # Action
        btn = QPushButton("EMBED WATERMARK")
        btn.clicked.connect(self.on_embed)
        main.addWidget(btn)

    def on_browse_host(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn host image", "", "Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ed_host.setText(path)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is not None:
                self.lb_preview.setPixmap(cv2_to_qpixmap(bgr).scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_browse_payload_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn watermark image", "", "Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ed_payload_img.setText(path)

    def on_browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu stego", "", "PNG (*.png)")
        if path:
            self.ed_out.setText(path)

    def _on_strength_change(self, v):
        self.lb_strength.setText(f"Alpha: {v/100:.2f}")

    def _alpha(self) -> float:
        return self.sl_strength.value()/100.0

    def on_embed(self):
        host = self.ed_host.text().strip()
        if not os.path.isfile(host) or not is_image(host):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn host IMAGE hợp lệ.")
            return
        wm = self.ed_payload_img.text().strip()
        if not os.path.isfile(wm) or not is_image(wm):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn watermark IMAGE (PNG/JPG).")
            return

        out = self.ed_out.text().strip()
        if not out:
            out = os.path.splitext(host)[0] + "_stego.png"
        meta = os.path.splitext(out)[0] + "_meta.npz"
        alpha = self._alpha()

        try:
            out_path, meta_path, ps, ss = embed_dctsvd(host, wm, out, meta, alpha=alpha)
            QMessageBox.information(self, "Thành công",
                f"Đã nhúng watermark!\nStego: {out_path}\nMeta: {meta_path}\nPSNR: {ps:.2f} dB\nSSIM: {ss:.4f}")
            bgr = cv2.imread(out_path, cv2.IMREAD_COLOR)
            if bgr is not None:
                self.lb_preview.setPixmap(cv2_to_qpixmap(bgr).scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Embed thất bại", str(e))

# -------------- EXTRACT TAB --------------
class ExtractTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        grp_host = QGroupBox("Watermarked Image")
        lh = QHBoxLayout(grp_host)
        self.ed_stego = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.on_browse)
        lh.addWidget(btn_browse, 0)
        lh.addWidget(self.ed_stego, 1)
        main.addWidget(grp_host)

        # meta path (auto, but allow manual select on missing)
        grp_meta = QGroupBox("Meta (.npz)")
        lm = QHBoxLayout(grp_meta)
        self.ed_meta = QLineEdit()
        self.ed_meta.setPlaceholderText("Tự động: <stego>_meta.npz (có thể chọn tay nếu không tìm thấy)")
        btn_meta = QPushButton("Browse")
        btn_meta.clicked.connect(self.on_browse_meta)
        lm.addWidget(btn_meta, 0); lm.addWidget(self.ed_meta, 1)
        main.addWidget(grp_meta)

        self.lb_preview = QLabel("Preview / Results")
        self.lb_preview.setFixedHeight(320)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444; background:#111; color:#bbb;}")
        main.addWidget(self.lb_preview)

        btn = QPushButton("EXTRACT WATERMARK")
        btn.clicked.connect(self.on_extract)
        main.addWidget(btn)

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn stego image", "", "Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path: self.ed_stego.setText(path)

    def on_browse_meta(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")
        if path: self.ed_meta.setText(path)

    def on_extract(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn stego image.")
            return
        meta = self.ed_meta.text().strip()
        if not meta:
            guess = os.path.splitext(stego)[0] + "_meta.npz"
            if os.path.isfile(guess):
                meta = guess
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")
                if not path: return
                meta = path
                self.ed_meta.setText(path)
        out_wm = os.path.splitext(stego)[0] + "_wm.png"
        try:
            wm_path = extract_dctsvd(stego, meta, out_wm)
            pix = QPixmap(wm_path)
            self.lb_preview.setPixmap(pix.scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            QMessageBox.information(self, "Xong", f"Khôi phục watermark: {wm_path}")
        except Exception as e:
            QMessageBox.critical(self, "Extract thất bại", str(e))

# -------------- DETECT TAB --------------
class DetectTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        grp_host = QGroupBox("Watermarked Image")
        lh = QHBoxLayout(grp_host)
        self.ed_stego = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.on_browse)
        lh.addWidget(btn_browse, 0)
        lh.addWidget(self.ed_stego, 1)
        main.addWidget(grp_host)

        grp_meta = QGroupBox("Meta (.npz)")
        lm = QHBoxLayout(grp_meta)
        self.ed_meta = QLineEdit()
        self.ed_meta.setPlaceholderText("Tự động: <stego>_meta.npz (có thể chọn tay nếu không tìm thấy)")
        btn_meta = QPushButton("Browse")
        btn_meta.clicked.connect(self.on_browse_meta)
        lm.addWidget(btn_meta, 0); lm.addWidget(self.ed_meta, 1)
        main.addWidget(grp_meta)

        self.lb_preview = QLabel("Preview / Results")
        self.lb_preview.setFixedHeight(220)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444; background:#111; color:#bbb;}")
        main.addWidget(self.lb_preview)

        self.lb_strength = QLabel("Thresh NC: 0.60")
        self.sl_strength = QSlider(Qt.Horizontal)
        self.sl_strength.setMinimum(10)  # 0.10
        self.sl_strength.setMaximum(90)  # 0.90
        self.sl_strength.setValue(60)    # 0.60
        self.sl_strength.valueChanged.connect(self._on_thresh_change)
        main.addWidget(self.lb_strength)
        main.addWidget(self.sl_strength)

        btn = QPushButton("DETECT PRESENCE")
        btn.clicked.connect(self.on_detect)
        main.addWidget(btn)

    def _on_thresh_change(self, v):
        self.lb_strength.setText(f"Thresh NC: {v/100:.2f}")

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn stego image", "", "Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path: self.ed_stego.setText(path)

    def on_browse_meta(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")
        if path: self.ed_meta.setText(path)

    def on_detect(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn stego image.")
            return
        meta = self.ed_meta.text().strip()
        if not meta:
            guess = os.path.splitext(stego)[0] + "_meta.npz"
            meta = guess if os.path.isfile(guess) else ""
        if not meta:
            path, _ = QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")
            if not path: return
            meta = path; self.ed_meta.setText(path)
        thresh = self.sl_strength.value()/100.0
        try:
            ok, score = detect_dctsvd(stego, meta, thresh=thresh)
            msg = f"NC = {score:.4f}\n" + ("✅ Có watermark" if ok else "❌ Không thấy watermark")
            self.lb_preview.setText(msg)
            QMessageBox.information(self, "Kết quả detect", msg)
        except Exception as e:
            QMessageBox.critical(self, "Detect thất bại", str(e))

# -------------- Main Window --------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(980, 680)

        tabs = QTabWidget()
        tabs.addTab(EmbedTab(), "EMBED")
        tabs.addTab(ExtractTab(), "EXTRACT")
        tabs.addTab(DetectTab(), "DETECT")
        self.setCentralWidget(tabs)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
