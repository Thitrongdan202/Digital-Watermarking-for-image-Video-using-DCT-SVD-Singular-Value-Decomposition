# detect_tab.py
# Tab DETECT: phát hiện watermark từ stego + meta
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets

# --- nạp dct_svd_core.py một cách chắc chắn ---
import os, sys, importlib, importlib.util

def _load_core_module():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(THIS_DIR, "dct_svd_core.py"),
        os.path.join(os.getcwd(), "dct_svd_core.py"),
        os.path.join(os.path.dirname(THIS_DIR), "dct_svd_core.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("dct_svd_core", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return importlib.import_module("dct_svd_core")

_core = _load_core_module()
core_detect = _core.detect

import numpy as np

class DetectTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)

        # Stego
        gb_stego = QtWidgets.QGroupBox("Watermarked Image")
        hl1 = QtWidgets.QHBoxLayout(gb_stego)
        self.ed_stego = QtWidgets.QLineEdit()
        btn_s = QtWidgets.QPushButton("Browse")
        btn_s.clicked.connect(self.pick_stego)
        hl1.addWidget(btn_s, 0); hl1.addWidget(self.ed_stego, 1)
        v.addWidget(gb_stego)

        # Meta
        gb_meta = QtWidgets.QGroupBox("Meta (.npz)")
        hl2 = QtWidgets.QHBoxLayout(gb_meta)
        self.ed_meta = QtWidgets.QLineEdit()
        self.ed_meta.setPlaceholderText("Tự dò <stego>_meta.npz nếu để trống")
        btn_m = QtWidgets.QPushButton("Browse")
        btn_m.clicked.connect(self.pick_meta)
        hl2.addWidget(btn_m, 0); hl2.addWidget(self.ed_meta, 1)
        v.addWidget(gb_meta)

        # Settings
        gb = QtWidgets.QGroupBox("Detect Settings")
        hl = QtWidgets.QHBoxLayout(gb)
        self.sp_thresh = QtWidgets.QDoubleSpinBox()
        self.sp_thresh.setRange(-10.0, 10.0)
        self.sp_thresh.setDecimals(3)
        self.sp_thresh.setValue(0.60)
        hl.addWidget(QtWidgets.QLabel("Thresh NC:"))
        hl.addWidget(self.sp_thresh)
        v.addWidget(gb)

        # Result
        self.lb_res = QtWidgets.QLabel("Preview / Results")
        self.lb_res.setMinimumHeight(220)
        self.lb_res.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_res.setStyleSheet("background:#111;color:#bbb;border:1px solid #333;")
        v.addWidget(self.lb_res)

        self.btn = QtWidgets.QPushButton("DETECT PRESENCE")
        self.btn.clicked.connect(self.on_detect)
        v.addWidget(self.btn)
        v.addStretch(1)

    def pick_stego(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn ảnh đã nhúng", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ed_stego.setText(path)
            base,_ = os.path.splitext(path)
            cand = base.replace("_stego","") + "_meta.npz"
            if os.path.isfile(cand):
                self.ed_meta.setText(cand)

    def pick_meta(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn file meta (.npz)", "", "NPZ (*.npz)")
        if path:
            self.ed_meta.setText(path)

    def on_detect(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh stego hợp lệ.")
            return
        meta = self.ed_meta.text().strip()
        if not os.path.isfile(meta):
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không tìm thấy meta .npz.")
            return

        try:
            ok, score = core_detect(stego, meta, self.sp_thresh.value())
            msg = f"NC = {score:.4f}\n\n" + ("✅ Có watermark" if ok else "❌ Không thấy watermark")
            self.lb_res.setText(msg)
            QtWidgets.QMessageBox.information(self, "Kết quả", msg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Detect thất bại", str(e))
