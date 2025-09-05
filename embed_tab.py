# embed_tab.py
# Tab EMBED: nhúng watermark ảnh / text / json
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # fallback PyQt5
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
    # cuối cùng: thử import theo module name (đòi hỏi file nằm trên sys.path)
    return importlib.import_module("dct_svd_core")

_core = _load_core_module()
core_embed = _core.embed

class EmbedTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # --------- UI ----------
    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)

        # Host
        gb_host = QtWidgets.QGroupBox("Host Image")
        hl = QtWidgets.QHBoxLayout(gb_host)
        self.ed_host = QtWidgets.QLineEdit()
        btn_host = QtWidgets.QPushButton("Browse")
        btn_host.clicked.connect(self.pick_host)
        hl.addWidget(btn_host, 0)
        hl.addWidget(self.ed_host, 1)
        v.addWidget(gb_host)

        # Preview
        self.lb_prev = QtWidgets.QLabel()
        self.lb_prev.setMinimumHeight(220)
        self.lb_prev.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_prev.setStyleSheet("background:#111;color:#bbb;border:1px solid #333;")
        v.addWidget(self.lb_prev)

        # Payload box
        gb_payload = QtWidgets.QGroupBox("Payload")
        gl = QtWidgets.QGridLayout(gb_payload)

        self.rb_text = QtWidgets.QRadioButton("Text")
        self.rb_json = QtWidgets.QRadioButton("JSON")
        self.rb_img  = QtWidgets.QRadioButton("Image")
        self.rb_text.setChecked(True)

        self.txt_payload = QtWidgets.QPlainTextEdit()
        self.txt_payload.setPlaceholderText("Nhập TEXT/JSON tại đây (hoặc dùng Browse để nạp file)…")

        self.ed_wm_img = QtWidgets.QLineEdit()
        btn_wm = QtWidgets.QPushButton("Browse Image/TXT/JSON…")
        btn_wm.clicked.connect(self.pick_payload_file)

        self.cb_color = QtWidgets.QCheckBox("Color watermark (RGB)")
        self.cb_color.setChecked(False)

        gl.addWidget(self.rb_text, 0,0); gl.addWidget(self.rb_json, 0,1); gl.addWidget(self.rb_img, 0,2)
        gl.addWidget(self.txt_payload, 1,0,1,3)
        gl.addWidget(btn_wm,        2,0); gl.addWidget(self.ed_wm_img, 2,1,1,2)
        gl.addWidget(self.cb_color, 3,0,1,3)
        v.addWidget(gb_payload)

        # Settings
        gb_set = QtWidgets.QGroupBox("Settings")
        hl2 = QtWidgets.QHBoxLayout(gb_set)
        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_alpha.setRange(1, 100); self.sld_alpha.setValue(25)
        self.lb_alpha = QtWidgets.QLabel("Alpha: 0.05")
        self.sld_alpha.valueChanged.connect(self._update_alpha_label)
        hl2.addWidget(self.lb_alpha, 0); hl2.addWidget(self.sld_alpha, 1)
        v.addWidget(gb_set)

        # Output
        gb_out = QtWidgets.QGroupBox("Output")
        hl3 = QtWidgets.QHBoxLayout(gb_out)
        self.ed_out = QtWidgets.QLineEdit()
        btn_out = QtWidgets.QPushButton("Save As")
        btn_out.clicked.connect(self.pick_out)
        hl3.addWidget(btn_out, 0); hl3.addWidget(self.ed_out, 1)
        v.addWidget(gb_out)

        # Button
        self.btn_embed = QtWidgets.QPushButton("EMBED WATERMARK")
        self.btn_embed.clicked.connect(self.on_embed)
        v.addWidget(self.btn_embed)
        v.addStretch(1)

    def _update_alpha_label(self):
        a = round(self.sld_alpha.value()/500.0, 3)  # 0.002–0.200
        self.lb_alpha.setText(f"Alpha: {a:.3f}")

    # --------- helpers ----------
    def pick_host(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn ảnh host", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.ed_host.setText(path)
            self._set_preview(path)

    def pick_payload_file(self):
        if self.rb_img.isChecked():
            filt = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        else:
            filt = "Text/JSON (*.txt *.json);;All (*)"
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn payload", "", filt)
        if path:
            self.ed_wm_img.setText(path)
            if not self.rb_img.isChecked():
                try:
                    with open(path,'r',encoding='utf-8',errors='ignore') as f:
                        self.txt_payload.setPlainText(f.read())
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Lỗi", str(e))

    def pick_out(self):
        path,_ = QtWidgets.QFileDialog.getSaveFileName(self, "Tên file xuất (PNG)", "", "PNG (*.png)")
        if path:
            self.ed_out.setText(path)

    def _set_preview(self, path):
        pix = QtGui.QPixmap(path)
        if not pix.isNull():
            self.lb_prev.setPixmap(pix.scaled(self.lb_prev.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.lb_prev.setText("Không xem trước được.")

    # --------- action ----------
    def on_embed(self):
        host = self.ed_host.text().strip()
        if not os.path.isfile(host):
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh host hợp lệ.")
            return

        alpha = round(self.sld_alpha.value()/500.0, 3)
        out = self.ed_out.text().strip()
        if not out:
            base, _ = os.path.splitext(host)
            out = base + "_stego.png"
        base,_ = os.path.splitext(out)
        meta = base.replace("_stego","") + "_meta.npz"

        try:
            if self.rb_img.isChecked():
                wm_path = self.ed_wm_img.text().strip()
                if not os.path.isfile(wm_path):
                    QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn ảnh watermark.")
                    return
                color = self.cb_color.isChecked()
                out_path, meta_path, ps, ss = core_embed(host, wm_path, out, meta, alpha=alpha, color=color, payload_type='image')
            else:
                pdata = self.txt_payload.toPlainText()
                ptype = 'json' if self.rb_json.isChecked() else 'text'
                wm_src = self.ed_wm_img.text().strip() if os.path.isfile(self.ed_wm_img.text().strip()) else ""
                out_path, meta_path, ps, ss = core_embed(host, wm_src, out, meta, alpha=alpha, color=False, payload_type=ptype, text_data=pdata)

            self._set_preview(out_path)
            QtWidgets.QMessageBox.information(self, "Xong", f"Đã nhúng!\n\nStego: {out_path}\nMeta:  {meta_path}\nPSNR: {ps:.2f} dB\nSSIM: {ss:.4f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Embed thất bại", str(e))
