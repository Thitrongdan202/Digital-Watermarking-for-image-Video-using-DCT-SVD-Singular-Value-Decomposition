
# Minimal PySide6 app: image-only watermarking with password
import sys, os, cv2
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QTabWidget, QCheckBox, QSlider)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from dct_svd_core_secure import embed as embed_core, extract as extract_core, detect as detect_core

def cv2_to_qpixmap(img_bgr):
    if img_bgr is None: return QPixmap()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCT-SVD Watermark (Image-Only, Password)")
        tabs = QTabWidget(self)

        # --------- EMBED TAB ---------
        embed_tab = QWidget(); tabs.addTab(embed_tab, "Embed")
        ev = QVBoxLayout(embed_tab)

        self.ed_cover = QLineEdit(); btn_cover = QPushButton("Chọn ảnh cover...")
        self.ed_wm    = QLineEdit(); btn_wm    = QPushButton("Chọn ảnh watermark...")
        self.ed_out   = QLineEdit(); btn_out   = QPushButton("Chọn nơi lưu stego...")
        self.ed_meta  = QLineEdit(); btn_meta  = QPushButton("Chọn nơi lưu meta (.npz)...")
        self.ed_pwd   = QLineEdit(); self.ed_pwd.setEchoMode(QLineEdit.Password)
        self.cb_color = QCheckBox("Nhúng theo màu (3 kênh)")
        self.s_alpha  = QSlider(Qt.Horizontal); self.s_alpha.setRange(1, 30); self.s_alpha.setValue(10)

        for lab, ed, btn in [
            ("Cover:", self.ed_cover, btn_cover),
            ("Watermark:", self.ed_wm, btn_wm),
            ("Stego out:", self.ed_out, btn_out),
            ("Meta out:", self.ed_meta, btn_meta),
            ("Mật khẩu:", self.ed_pwd, None),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(lab)); row.addWidget(ed)
            if btn: row.addWidget(btn)
            ev.addLayout(row)
        ev.addWidget(self.cb_color)
        row = QHBoxLayout()
        row.addWidget(QLabel("Alpha (x0.01):")); row.addWidget(self.s_alpha)
        ev.addLayout(row)
        self.lbl_psnr = QLabel("PSNR: -"); self.lbl_ssim = QLabel("SSIM: -"); ev.addWidget(self.lbl_psnr); ev.addWidget(self.lbl_ssim)
        self.lbl_preview = QLabel(); ev.addWidget(self.lbl_preview)
        btn_run = QPushButton("NHÚNG"); ev.addWidget(btn_run)

        # file pickers
        btn_cover.clicked.connect(lambda: self.ed_cover.setText(QFileDialog.getOpenFileName(self, "Chọn cover", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]))
        btn_wm.clicked.connect(lambda: self.ed_wm.setText(QFileDialog.getOpenFileName(self, "Chọn watermark", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]))
        btn_out.clicked.connect(lambda: self.ed_out.setText(QFileDialog.getSaveFileName(self, "Chọn stego out", "", "PNG (*.png)")[0]))
        btn_meta.clicked.connect(lambda: self.ed_meta.setText(QFileDialog.getSaveFileName(self, "Chọn meta", "", "NPZ (*.npz)")[0]))

        def do_embed():
            try:
                alpha = self.s_alpha.value()/100.0
                out_path, meta_path, ps, ss = embed_core(
                    self.ed_cover.text().strip(),
                    self.ed_wm.text().strip(),
                    self.ed_out.text().strip() or "stego.png",
                    self.ed_meta.text().strip() or "stego_meta.npz",
                    alpha=alpha,
                    color=self.cb_color.isChecked(),
                    password=self.ed_pwd.text().strip(),
                )
                self.lbl_psnr.setText(f"PSNR: {ps:.2f} dB"); self.lbl_ssim.setText(f"SSIM: {ss:.4f}")
                img = cv2.imread(out_path, cv2.IMREAD_COLOR); self.lbl_preview.setPixmap(cv2_to_qpixmap(img).scaledToWidth(420, Qt.SmoothTransformation))
            except Exception as e:
                self.lbl_psnr.setText("LỖI: " + str(e))
        btn_run.clicked.connect(do_embed)

        # --------- EXTRACT TAB ---------
        xtab = QWidget(); tabs.addTab(xtab, "Extract")
        xv = QVBoxLayout(xtab)
        self.ed_stego = QLineEdit(); btn_stego = QPushButton("Chọn stego...")
        self.ed_meta2 = QLineEdit(); btn_meta2 = QPushButton("Chọn meta (.npz)...")
        self.ed_out2  = QLineEdit(); btn_out2  = QPushButton("Chọn nơi lưu watermark...")
        self.ed_pwd2  = QLineEdit(); self.ed_pwd2.setEchoMode(QLineEdit.Password)
        for lab, ed, btn in [
            ("Stego:", self.ed_stego, btn_stego),
            ("Meta:", self.ed_meta2, btn_meta2),
            ("Out:",  self.ed_out2,  btn_out2),
            ("Mật khẩu:", self.ed_pwd2, None),
        ]:
            row = QHBoxLayout(); row.addWidget(QLabel(lab)); row.addWidget(ed);
            if btn: row.addWidget(btn)
            xv.addLayout(row)
        self.lbl_prev2 = QLabel(); xv.addWidget(self.lbl_prev2)
        btn_x = QPushButton("GIẢI TRÍCH (Nhập mật khẩu)"); xv.addWidget(btn_x)
        btn_stego.clicked.connect(lambda: self.ed_stego.setText(QFileDialog.getOpenFileName(self, "Chọn stego", "", "PNG (*.png);;Images (*.jpg *.jpeg *.bmp)")[0]))
        btn_meta2.clicked.connect(lambda: self.ed_meta2.setText(QFileDialog.getOpenFileName(self, "Chọn meta", "", "NPZ (*.npz)")[0]))
        btn_out2.clicked.connect(lambda: self.ed_out2.setText(QFileDialog.getSaveFileName(self, "Chọn watermark out", "", "PNG (*.png)")[0]))
        def do_extract():
            try:
                outp = extract_core(
                    self.ed_stego.text().strip(),
                    self.ed_meta2.text().strip(),
                    self.ed_out2.text().strip() or "wm.png",
                    password=self.ed_pwd2.text().strip(),
                    normalize=True,
                )
                img = cv2.imread(outp, cv2.IMREAD_COLOR) or cv2.imread(outp, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    self.lbl_prev2.setPixmap(cv2_to_qpixmap(img).scaledToWidth(420, Qt.SmoothTransformation))
            except Exception as e:
                self.lbl_prev2.setText("LỖI: " + str(e))
        btn_x.clicked.connect(do_extract)

        # layout
        main = QVBoxLayout(self); main.addWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App(); w.resize(720, 540); w.show()
    sys.exit(app.exec())
