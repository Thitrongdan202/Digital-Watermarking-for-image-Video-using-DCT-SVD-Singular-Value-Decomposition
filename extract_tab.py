# extract_tab.py
# Tab EXTRACT: khôi phục watermark (ảnh / text / json)
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
core_extract = _core.extract

import json
import numpy as np

class ExtractTab(QtWidgets.QWidget):
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

        # Preview
        self.lb_prev = QtWidgets.QLabel("Preview / Results")
        self.lb_prev.setMinimumHeight(220)
        self.lb_prev.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_prev.setStyleSheet("background:#111;color:#bbb;border:1px solid #333;")
        v.addWidget(self.lb_prev)

        self.btn_x = QtWidgets.QPushButton("EXTRACT")
        self.btn_x.clicked.connect(self.on_extract)
        v.addWidget(self.btn_x)
        v.addStretch(1)

    def pick_stego(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn ảnh đã nhúng", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ed_stego.setText(path)

    def pick_meta(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn file meta (.npz)", "", "NPZ (*.npz)")
        if path:
            self.ed_meta.setText(path)

    def _auto_meta(self, stego):
        base,_ = os.path.splitext(stego)
        cand = base.replace("_stego","") + "_meta.npz"
        return cand if os.path.isfile(cand) else ""

    def on_extract(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh stego hợp lệ.")
            return
        meta = self.ed_meta.text().strip() or self._auto_meta(stego)
        if not os.path.isfile(meta):
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Không tìm thấy meta .npz. Chọn đúng file meta.")
            return

        base,_ = os.path.splitext(stego)
        out_prefix = base + "_recovered"

        try:
            data = np.load(meta, allow_pickle=False)
            payload_type = str(data.get('payload_type','image'))

            if payload_type == 'image':
                out = core_extract(stego, meta, out_prefix)
                pix = QtGui.QPixmap(out)
                if pix.isNull():
                    self.lb_prev.setText("Không xem trước được ảnh.")
                else:
                    self.lb_prev.setPixmap(pix.scaled(self.lb_prev.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            else:
                out = core_extract(stego, meta, out_prefix)
                try:
                    with open(out, 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                    self.lb_prev.setText(txt[:3000] + ("\n...\n" if len(txt) > 3000 else ""))
                except Exception:
                    self.lb_prev.setText("Đã tách TEXT/JSON nhưng không hiển thị được.")
            QtWidgets.QMessageBox.information(self, "Xong", f"Khôi phục: {out}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Extract thất bại", str(e))
