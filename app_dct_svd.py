import os, sys, json
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QPlainTextEdit, QRadioButton,
    QMessageBox, QSpinBox, QTabWidget, QMainWindow, QCheckBox
)
from dct_svd_core import embed as embed_core, extract as extract_core, detect as detect_core

APP_TITLE = "DCT–SVD Watermarking (Images, Text, JSON)"

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def cv2_to_qpixmap(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ------------ EMBED TAB -------------
class EmbedTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        grp_host = QGroupBox("Host Image")
        lh = QHBoxLayout(grp_host)
        self.ed_host = QLineEdit()
        btn = QPushButton("Browse")
        btn.clicked.connect(self.on_browse_host)
        lh.addWidget(btn, 0); lh.addWidget(self.ed_host, 1)
        main.addWidget(grp_host)

        self.lb_preview = QLabel("")
        self.lb_preview.setFixedHeight(320)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444;background:#111;color:#bbb;}")
        main.addWidget(self.lb_preview)

        grp_payload = QGroupBox("Payload")
        pv = QVBoxLayout(grp_payload)
        row = QHBoxLayout()
        self.rb_text = QRadioButton("Text"); self.rb_json = QRadioButton("JSON"); self.rb_img = QRadioButton("Image")
        self.rb_img.setChecked(True)
        row.addWidget(self.rb_text); row.addWidget(self.rb_json); row.addWidget(self.rb_img)
        pv.addLayout(row)

        # editor for text/json
        self.ed_payload = QPlainTextEdit()
        self.ed_payload.setPlaceholderText("Nhập TEXT hoặc JSON tại đây (hoặc dùng Browse JSON/TXT).")
        pv.addWidget(self.ed_payload)
        rowj = QHBoxLayout()
        self.ed_payload_json = QLineEdit(); self.ed_payload_json.setPlaceholderText("Đường dẫn file JSON/TXT…")
        btn_browse_json = QPushButton("Browse JSON/TXT"); btn_browse_json.clicked.connect(self.on_browse_payload_json)
        rowj.addWidget(btn_browse_json, 0); rowj.addWidget(self.ed_payload_json, 1)
        pv.addLayout(rowj)

        # image path
        rowi = QHBoxLayout()
        self.ed_payload_img = QLineEdit(); self.ed_payload_img.setPlaceholderText("Đường dẫn watermark IMAGE (PNG/JPG)…")
        btn_browse_img = QPushButton("Browse Image"); btn_browse_img.clicked.connect(self.on_browse_payload_img)
        rowi.addWidget(btn_browse_img, 0); rowi.addWidget(self.ed_payload_img, 1)
        pv.addLayout(rowi)

        # Settings
        grp_set = QGroupBox("Settings")
        ls = QHBoxLayout(grp_set)
        self.lb_alpha = QLabel("Alpha: 0.05")
        self.sl_alpha = QSlider(Qt.Horizontal); self.sl_alpha.setMinimum(1); self.sl_alpha.setMaximum(20); self.sl_alpha.setValue(5)
        self.sl_alpha.valueChanged.connect(lambda v: self.lb_alpha.setText(f"Alpha: {v/100:.2f}"))
        self.cb_color = QCheckBox("Color watermark (RGB)")
        ls.addWidget(self.lb_alpha); ls.addWidget(self.sl_alpha); ls.addWidget(self.cb_color)

        # Output
        grp_out = QGroupBox("Output")
        lo = QHBoxLayout(grp_out)
        self.ed_out = QLineEdit(); self.ed_out.setPlaceholderText("Đường dẫn file xuất (trống: *_stego.png + *_meta.npz)")
        btn_out = QPushButton("Save As"); btn_out.clicked.connect(self.on_browse_out)
        lo.addWidget(btn_out, 0); lo.addWidget(self.ed_out, 1)

        main.addWidget(grp_payload); main.addWidget(grp_set); main.addWidget(grp_out)
        btn_go = QPushButton("EMBED WATERMARK"); btn_go.clicked.connect(self.on_embed)
        main.addWidget(btn_go)

    # handlers
    def on_browse_host(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn host image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p:
            self.ed_host.setText(p)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                self.lb_preview.setPixmap(cv2_to_qpixmap(img).scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_browse_payload_img(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn watermark image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p: self.ed_payload_img.setText(p)

    def on_browse_payload_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn JSON/TXT", "", "JSON/TXT (*.json *.txt);;All (*.*)")
        if p:
            self.ed_payload_json.setText(p)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    self.ed_payload.setPlainText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không đọc được file: {e}")

    def on_browse_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "Lưu stego", "", "PNG (*.png)")
        if p: self.ed_out.setText(p)

    def on_embed(self):
        host = self.ed_host.text().strip()
        if not (host and is_image(host)):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn host IMAGE hợp lệ."); return
        alpha = self.sl_alpha.value()/100.0
        out = self.ed_out.text().strip()
        if not out: out = os.path.splitext(host)[0] + "_stego.png"
        meta = os.path.splitext(out)[0] + "_meta.npz"

        try:
            if self.rb_img.isChecked():
                wm = self.ed_payload_img.text().strip()
                if not (wm and is_image(wm)):
                    QMessageBox.warning(self, "Lỗi", "Vui lòng chọn watermark IMAGE."); return
                color = self.cb_color.isChecked()
                out_path, meta_path, ps, ss = embed_core(host, wm, out, meta, alpha=alpha, color=color, payload_type='image')
            elif self.rb_text.isChecked():
                txt = self.ed_payload.toPlainText().strip()
                if not txt and self.ed_payload_json.text().strip():
                    try:
                        with open(self.ed_payload_json.text().strip(), 'r', encoding='utf-8', errors='ignore') as f:
                            txt = f.read()
                    except Exception as e:
                        QMessageBox.warning(self, "Lỗi", f"Không đọc được file: {e}"); return
                if not txt:
                    QMessageBox.warning(self, "Lỗi", "Vui lòng nhập TEXT hoặc chọn file .txt"); return
                out_path, meta_path, ps, ss = embed_core(host, "", out, meta, alpha=alpha, color=False, payload_type='text', text_data=txt)
            else:  # JSON
                txt = self.ed_payload.toPlainText().strip()
                if not txt and self.ed_payload_json.text().strip():
                    try:
                        with open(self.ed_payload_json.text().strip(), 'r', encoding='utf-8', errors='ignore') as f:
                            txt = f.read()
                    except Exception as e:
                        QMessageBox.warning(self, "Lỗi", f"Không đọc được file: {e}"); return
                if not txt:
                    QMessageBox.warning(self, "Lỗi", "Vui lòng nhập JSON hoặc chọn file .json"); return
                try: json.loads(txt)
                except Exception as e:
                    QMessageBox.warning(self, "Lỗi", f"JSON không hợp lệ: {e}"); return
                out_path, meta_path, ps, ss = embed_core(host, "", out, meta, alpha=alpha, color=False, payload_type='json', text_data=txt)

            QMessageBox.information(self, "Thành công", f"Đã nhúng watermark!\nStego: {out_path}\nMeta: {meta_path}\nPSNR: {ps:.2f} dB\nSSIM: {ss:.4f}")
            img = cv2.imread(out_path, cv2.IMREAD_COLOR)
            if img is not None:
                self.lb_preview.setPixmap(cv2_to_qpixmap(img).scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Embed thất bại", str(e))

# ------------ EXTRACT TAB -------------
class ExtractTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        grp_host = QGroupBox("Watermarked Image")
        lh = QHBoxLayout(grp_host)
        self.ed_stego = QLineEdit()
        btn = QPushButton("Browse"); btn.clicked.connect(self.on_browse)
        lh.addWidget(btn, 0); lh.addWidget(self.ed_stego, 1)
        main.addWidget(grp_host)

        grp_meta = QGroupBox("Meta (.npz)")
        lm = QHBoxLayout(grp_meta)
        self.ed_meta = QLineEdit(); self.ed_meta.setPlaceholderText("Tự dò <stego>_meta.npz nếu để trống")
        btnm = QPushButton("Browse"); btnm.clicked.connect(self.on_browse_meta)
        lm.addWidget(btnm, 0); lm.addWidget(self.ed_meta, 1)
        main.addWidget(grp_meta)

        self.lb_preview = QLabel("Preview / Results"); self.lb_preview.setFixedHeight(320); self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444;background:#111;color:#bbb;}")
        main.addWidget(self.lb_preview)

        btn_go = QPushButton("EXTRACT"); btn_go.clicked.connect(self.on_extract)
        main.addWidget(btn_go)

    def on_browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn stego image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p: self.ed_stego.setText(p)

    def on_browse_meta(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")
        if p: self.ed_meta.setText(p)

    def on_extract(self):
        stego = self.ed_stego.text().strip()
        if not stego: QMessageBox.warning(self,"Lỗi","Chọn stego image."); return
        meta = self.ed_meta.text().strip()
        if not meta:
            guess = os.path.splitext(stego)[0] + "_meta.npz"
            if os.path.isfile(guess): meta = guess
        out_base = os.path.splitext(stego)[0] + "_recovered"
        try:
            out_path = extract_core(stego, meta, out_base)
            ext = os.path.splitext(out_path)[1].lower()
            if ext in (".png",".jpg",".jpeg"):
                pix = QPixmap(out_path); self.lb_preview.setPixmap(pix.scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                with open(out_path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                if len(txt) > 4000: txt = txt[:4000] + "\n...\n(truncated)"
                self.lb_preview.setText(txt)
            QMessageBox.information(self, "Xong", f"Khôi phục: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Extract thất bại", str(e))

# ------------ DETECT TAB -------------
class DetectTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        grp_host = QGroupBox("Watermarked Image")
        lh = QHBoxLayout(grp_host)
        self.ed_stego = QLineEdit(); btn = QPushButton("Browse"); btn.clicked.connect(self.on_browse)
        lh.addWidget(btn,0); lh.addWidget(self.ed_stego,1); main.addWidget(grp_host)

        grp_meta = QGroupBox("Meta (.npz)")
        lm = QHBoxLayout(grp_meta)
        self.ed_meta = QLineEdit(); self.ed_meta.setPlaceholderText("Tự dò <stego>_meta.npz nếu để trống")
        btnm = QPushButton("Browse"); btnm.clicked.connect(self.on_browse_meta)
        lm.addWidget(btnm,0); lm.addWidget(self.ed_meta,1); main.addWidget(grp_meta)

        self.lb_preview = QLabel("Preview / Results"); self.lb_preview.setFixedHeight(220); self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444;background:#111;color:#bbb;}")
        main.addWidget(self.lb_preview)

        self.lb_thresh = QLabel("Thresh NC: 0.60")
        self.sl_thresh = QSlider(Qt.Horizontal); self.sl_thresh.setMinimum(10); self.sl_thresh.setMaximum(90); self.sl_thresh.setValue(60)
        self.sl_thresh.valueChanged.connect(lambda v: self.lb_thresh.setText(f"Thresh NC: {v/100:.2f}"))
        main.addWidget(self.lb_thresh); main.addWidget(self.sl_thresh)

        btn_go = QPushButton("DETECT PRESENCE"); btn_go.clicked.connect(self.on_detect)
        main.addWidget(btn_go)

    def on_browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn stego image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p: self.ed_stego.setText(p)

    def on_browse_meta(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")
        if p: self.ed_meta.setText(p)

    def on_detect(self):
        stego = self.ed_stego.text().strip()
        if not stego: QMessageBox.warning(self, "Lỗi", "Chọn stego image."); return
        meta = self.ed_meta.text().strip()
        if not meta:
            guess = os.path.splitext(stego)[0] + "_meta.npz"
            if os.path.isfile(guess): meta = guess
        thresh = self.sl_thresh.value()/100.0
        try:
            ok, score = detect_core(stego, meta, thresh=thresh)
            msg = f"NC = {score:.4f}\n" + ("✅ Có watermark" if ok else "❌ Không thấy watermark")
            self.lb_preview.setText(msg); QMessageBox.information(self, "Kết quả detect", msg)
        except Exception as e:
            QMessageBox.critical(self, "Detect thất bại", str(e))

# ----------- Main -----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.resize(1000, 700)
        tabs = QTabWidget()
        tabs.addTab(EmbedTab(), "EMBED")
        tabs.addTab(ExtractTab(), "EXTRACT")
        tabs.addTab(DetectTab(), "DETECT")
        self.setCentralWidget(tabs)

def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
