# app.py
import os, sys, json, cv2
import numpy as np

# --- GUI: ưu tiên PyQt5, fallback PySide6 nếu cần ---
try:
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QFileDialog, QMessageBox,
        QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
        QRadioButton, QPlainTextEdit, QSpinBox, QSlider, QButtonGroup
    )
except Exception:
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QPixmap, QImage
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QFileDialog, QMessageBox,
        QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
        QRadioButton, QPlainTextEdit, QSpinBox, QSlider, QButtonGroup
    )

# --- Backend mới: imwatermark ---
from payload_dwt_dct_svd import (
    embed_into_image, extract_from_image,
    embed_into_video, extract_from_video
)

APP_TITLE = "DCT-SVD Watermarking (imwatermark payload)"

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
VID_EXTS = ('.avi', '.mp4', '.mov', '.mkv', '.webm', '.wmv', '.m4v')


def is_image(path: str) -> bool:
    return path and path.lower().endswith(IMG_EXTS)


def is_video(path: str) -> bool:
    return path and path.lower().endswith(VID_EXTS)


def cv2_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    if bgr is None: return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def first_frame_of_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------- EMBED TAB ----------------
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
        self.lb_preview = QLabel("Preview sẽ hiển thị ở đây")
        self.lb_preview.setFixedHeight(320)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #777;}")

        # Payload type
        grp_payload = QGroupBox("Payload")
        pv = QVBoxLayout(grp_payload)

        # radio buttons
        row = QHBoxLayout()
        self.rb_text = QRadioButton("Text")
        self.rb_json = QRadioButton("JSON")
        self.rb_img = QRadioButton("Image")
        self.rb_text.setChecked(True)
        row.addWidget(self.rb_text);
        row.addWidget(self.rb_json);
        row.addWidget(self.rb_img)
        pv.addLayout(row)

        # text/json editor
        self.ed_payload = QPlainTextEdit()
        self.ed_payload.setPlaceholderText("Nhập text / JSON tại đây…")
        pv.addWidget(self.ed_payload)

        # JSON file path
        rowj = QHBoxLayout()
        self.ed_payload_json = QLineEdit()
        self.ed_payload_json.setPlaceholderText("Đường dẫn file JSON…")
        btn_browse_payload_json = QPushButton("Browse JSON")
        btn_browse_payload_json.clicked.connect(self.on_browse_payload_json)
        rowj.addWidget(btn_browse_payload_json, 0)
        rowj.addWidget(self.ed_payload_json, 1)
        pv.addLayout(rowj)

        # payload image path
        row2 = QHBoxLayout()
        self.ed_payload_img = QLineEdit()
        self.ed_payload_img.setPlaceholderText("Đường dẫn payload ảnh (PNG/JPG)…")
        btn_browse_payload_img = QPushButton("Browse")
        btn_browse_payload_img.clicked.connect(self.on_browse_payload_img)
        row2.addWidget(btn_browse_payload_img, 0)
        row2.addWidget(self.ed_payload_img, 1)
        pv.addLayout(row2)

        # Strength + Frame interval
        grp_set = QGroupBox("Settings")
        ls = QHBoxLayout(grp_set)
        self.lb_strength = QLabel("Strength: 0.12")
        self.sl_strength = QSlider(Qt.Horizontal)
        self.sl_strength.setMinimum(5)  # 0.05
        self.sl_strength.setMaximum(30)  # 0.30
        self.sl_strength.setValue(12)  # 0.12
        self.sl_strength.valueChanged.connect(self._on_strength_change)
        ls.addWidget(self.lb_strength)
        ls.addWidget(self.sl_strength)

        self.lb_interval = QLabel("Frame interval (video):")
        self.sp_interval = QSpinBox()
        self.sp_interval.setMinimum(1)
        self.sp_interval.setMaximum(60)
        self.sp_interval.setValue(1)
        ls.addWidget(self.lb_interval)
        ls.addWidget(self.sp_interval)

        # Output
        grp_out = QGroupBox("Output")
        lo = QHBoxLayout(grp_out)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Đường dẫn file xuất (để trống: tự động *_stego.png / *_stego.avi)")
        btn_browse_out = QPushButton("Save As")
        btn_browse_out.clicked.connect(self.on_browse_out)
        lo.addWidget(btn_browse_out, 0)
        lo.addWidget(self.ed_out, 1)

        # Button embed
        btn_embed = QPushButton("EMBED WATERMARK")
        btn_embed.clicked.connect(self.on_embed)

        main.addWidget(grp_host)
        main.addWidget(self.lb_preview)
        main.addWidget(grp_payload)
        main.addWidget(grp_set)
        main.addWidget(grp_out)
        main.addWidget(btn_embed)

        # default visibility
        self._sync_payload_inputs()

        # when radio changed, sync widgets
        self.rb_text.toggled.connect(self._sync_payload_inputs)
        self.rb_json.toggled.connect(self._sync_payload_inputs)
        self.rb_img.toggled.connect(self._sync_payload_inputs)

    # UI helpers
    def _on_strength_change(self, v):
        self.lb_strength.setText(f"Strength: {v / 100:.2f}")

    def _sync_payload_inputs(self):
        is_img = self.rb_img.isChecked()
        self.ed_payload.setEnabled(not is_img)
        self.ed_payload_img.setEnabled(is_img)

    def on_browse_payload_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn JSON", "", "JSON (*.json);;Text (*.txt);;All (*.*)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    self.ed_payload.setPlainText(f.read())
                self.ed_payload_json.setText(path)
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"Không đọc được JSON: {e}")

    def on_browse_host(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn host media", "",
                                              "Media (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.avi *.mp4 *.mov *.mkv *.webm *.wmv *.m4v)")
        if not path: return
        self.ed_host.setText(path)
        # preview
        if is_image(path):
            pix = QPixmap(path)
            self.lb_preview.setPixmap(pix.scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif is_video(path):
            f = first_frame_of_video(path)
            self.lb_preview.setPixmap(
                cv2_to_qpixmap(f).scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lb_preview.setText("Không hỗ trợ định dạng này.")

    def on_browse_payload_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh payload", "", "Image (*.png *.jpg *.jpeg)")
        if path:
            self.ed_payload_img.setText(path)

    def on_browse_out(self):
        # chỉ chọn thư mục/filename – nhưng để đơn giản, cho chọn file tuỳ ý
        path, _ = QFileDialog.getSaveFileName(self, "Chọn đường dẫn xuất", "", "All (*.*)")
        if path:
            self.ed_out.setText(path)

    def _strength(self) -> float:
        return self.sl_strength.value() / 100.0

    def on_embed(self):
        host = self.ed_host.text().strip()
        if not os.path.isfile(host):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn host media hợp lệ.")
            return

        # xác định kind & payload
        if self.rb_text.isChecked():
            kind = 'text'
            payload = self.ed_payload.toPlainText()
        elif self.rb_json.isChecked():
            kind = 'json'
            txt = self.ed_payload.toPlainText().strip()
            if not txt and hasattr(self, 'ed_payload_json') and self.ed_payload_json.text().strip():
                try:
                    with open(self.ed_payload_json.text().strip(), 'r', encoding='utf-8', errors='ignore') as f:
                        txt = f.read()
                        self.ed_payload.setPlainText(txt)
                except Exception as e:
                    QMessageBox.warning(self, "Lỗi", f"Không đọc được JSON: {e}")
                    return
            # chấp nhận text JSON hoặc bạn tự nhập text (sẽ tự parse)
            try:
                json.loads(txt)  # validate
            except Exception as e:
                QMessageBox.warning(self, "Lỗi", f"JSON không hợp lệ:\n{e}")
                return
            payload = txt
        else:
            kind = 'image'
            pimg = self.ed_payload_img.text().strip()
            if not os.path.isfile(pimg):
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn payload ảnh hợp lệ.")
                return
            payload = pimg

        out = self.ed_out.text().strip()
        strength = self._strength()
        interval = self.sp_interval.value()

        try:
            if is_image(host):
                out_path = out or (os.path.splitext(host)[0] + "_stego.png")
                out_path = embed_into_image(host, out_path, kind=kind, payload=payload, strength=strength)
            elif is_video(host):
                out_path = out or (os.path.splitext(host)[0] + "_stego.avi")
                out_path = embed_into_video(host, out_path, kind=kind, payload=payload,
                                            strength=strength, frame_interval=interval)
            else:
                QMessageBox.warning(self, "Lỗi", "Định dạng host không hỗ trợ.")
                return

            QMessageBox.information(self, "Thành công",
                                    f"Đã nhúng watermark!\nFile xuất:\n{out_path}")
            # preview stego
            if is_image(out_path):
                pix = QPixmap(out_path)
                self.lb_preview.setPixmap(
                    pix.scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif is_video(out_path):
                f = first_frame_of_video(out_path)
                self.lb_preview.setPixmap(
                    cv2_to_qpixmap(f).scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            QMessageBox.critical(self, "Embed thất bại", str(e))


# ---------------- DETECT TAB ----------------

class ExtractTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Stego media
        grp_host = QGroupBox("Watermarked Media")
        lh = QHBoxLayout(grp_host)
        self.ed_stego = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.on_browse)
        lh.addWidget(btn_browse, 0)
        lh.addWidget(self.ed_stego, 1)
        main.addWidget(grp_host)

        # Settings
        grp_set = QGroupBox("Extract Settings")
        ls = QHBoxLayout(grp_set)

        self.lb_strength = QLabel("Strength (≈ lúc embed): 0.12")
        self.sl_strength = QSlider(Qt.Horizontal)
        self.sl_strength.setMinimum(5)
        self.sl_strength.setMaximum(30)
        self.sl_strength.setValue(12)
        self.sl_strength.valueChanged.connect(self._on_strength_change)
        ls.addWidget(self.lb_strength)
        ls.addWidget(self.sl_strength)

        self.lb_interval = QLabel("Frame interval:")
        self.sp_interval = QSpinBox()
        self.sp_interval.setMinimum(1)
        self.sp_interval.setMaximum(60)
        self.sp_interval.setValue(1)
        ls.addWidget(self.lb_interval)
        ls.addWidget(self.sp_interval)

        # Preview / results
        self.lb_preview = QLabel("Preview / Results")
        self.lb_preview.setFixedHeight(320)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #444; background:#111; color:#bbb;}")
        main.addWidget(grp_set)
        main.addWidget(self.lb_preview)

        # Extract button
        btn_extract = QPushButton("EXTRACT PAYLOAD")
        btn_extract.clicked.connect(self.on_extract)
        main.addWidget(btn_extract)

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn stego media", "",
                                              "Media (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.avi *.mp4 *.mov *.mkv *.webm *.wmv *.m4v)")
        if path:
            self.ed_stego.setText(path)

    def _on_strength_change(self, v):
        self.lb_strength.setText(f"Strength (≈ lúc embed): {v / 100:.2f}")

    def _strength(self) -> float:
        return self.sl_strength.value() / 100.0

    def on_extract(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn stego media hợp lệ.")
            return
        strength = self._strength()
        interval = self.sp_interval.value()
        save_dir = os.path.splitext(stego)[0] + "_extracted"
        try:
            if is_image(stego):
                payload_file = extract_from_image(stego, save_dir, strength=strength)
            elif is_video(stego):
                payload_file = extract_from_video(stego, save_dir, strength=strength, frame_interval=interval)
            else:
                QMessageBox.warning(self, "Lỗi", "Định dạng stego không hỗ trợ.")
                return
            # show result
            ext = os.path.splitext(payload_file)[1].lower()
            if ext in ('.txt', '.json'):
                with open(payload_file, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                if len(txt) > 3000:
                    txt = txt[:3000] + "\n...\n(truncated)"
                self.lb_preview.setText(txt)
            elif ext in ('.png', '.jpg', '.jpeg'):
                pix = QPixmap(payload_file)
                self.lb_preview.setPixmap(
                    pix.scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.lb_preview.setText(f"Đã lưu payload:\n{payload_file}")
            QMessageBox.information(self, "Extract xong", f"Đã khôi phục payload:\n{payload_file}")
        except Exception as e:
            QMessageBox.critical(self, "Extract thất bại", str(e))


class DetectTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Stego media
        grp_host = QGroupBox("Watermarked Media")
        lh = QHBoxLayout(grp_host)
        self.ed_stego = QLineEdit()
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.on_browse)
        lh.addWidget(btn_browse, 0)
        lh.addWidget(self.ed_stego, 1)

        # Settings
        grp_set = QGroupBox("Detect Settings")
        ls = QHBoxLayout(grp_set)
        self.lb_strength = QLabel("Strength (≈ lúc embed): 0.12")
        self.sl_strength = QSlider(Qt.Horizontal)
        self.sl_strength.setMinimum(5)
        self.sl_strength.setMaximum(30)
        self.sl_strength.setValue(12)
        self.sl_strength.valueChanged.connect(self._on_strength_change)
        ls.addWidget(self.lb_strength)
        ls.addWidget(self.sl_strength)

        self.lb_interval = QLabel("Frame interval:")
        self.sp_interval = QSpinBox()
        self.sp_interval.setMinimum(1)
        self.sp_interval.setMaximum(60)
        self.sp_interval.setValue(1)
        ls.addWidget(self.lb_interval)
        ls.addWidget(self.sp_interval)

        # Preview / results
        self.lb_preview = QLabel("Preview / Results")
        self.lb_preview.setFixedHeight(320)
        self.lb_preview.setAlignment(Qt.AlignCenter)
        self.lb_preview.setStyleSheet("QLabel{border:1px solid #777;}")

        # Detect button only
        btn_detect = QPushButton("DETECT PRESENCE")
        btn_detect.clicked.connect(self.on_detect)

        main.addWidget(grp_host)
        main.addWidget(grp_set)
        main.addWidget(self.lb_preview)
        main.addWidget(btn_detect)

    def _on_strength_change(self, v):
        self.lb_strength.setText(f"Strength (≈ lúc embed): {v / 100:.2f}")

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn stego media", "",
                                              "Media (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.avi *.mp4 *.mov *.mkv *.webm *.wmv *.m4v)")
        if path:
            self.ed_stego.setText(path)

    def _strength(self) -> float:
        return self.sl_strength.value() / 100.0

    def on_detect(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn stego media hợp lệ.")
            return
        strength = self._strength()
        interval = self.sp_interval.value()
        try:
            if is_image(stego):
                present = detect_in_image(stego, strength=strength)
            elif is_video(stego):
                present = detect_in_video(stego, strength=strength, frame_interval=interval)
            else:
                QMessageBox.warning(self, "Lỗi", "Định dạng stego không hỗ trợ.")
                return
            msg = "✅ Có watermark (tìm thấy payload header)." if present else "❌ Không thấy watermark (không tìm thấy header)."
            self.lb_preview.setText(msg)
            QMessageBox.information(self, "Kết quả detect", msg)
        except Exception as e:
            QMessageBox.critical(self, "Detect thất bại", str(e))

    def on_extract(self):
        stego = self.ed_stego.text().strip()
        if not os.path.isfile(stego):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn stego media hợp lệ.")
            return
        strength = self._strength()
        interval = self.sp_interval.value()
        save_dir = os.path.splitext(stego)[0] + "_extracted"
        try:
            if is_image(stego):
                payload_file = extract_from_image(stego, save_dir, strength=strength)
            elif is_video(stego):
                payload_file = extract_from_video(stego, save_dir, strength=strength, frame_interval=interval)
            else:
                QMessageBox.warning(self, "Lỗi", "Định dạng stego không hỗ trợ.")
                return
            # show result
            ext = os.path.splitext(payload_file)[1].lower()
            if ext in ('.txt', '.json'):
                with open(payload_file, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                if len(txt) > 3000:
                    txt = txt[:3000] + "\n...\n(truncated)"
                self.lb_preview.setText(txt)
            elif ext in ('.png', '.jpg', '.jpeg'):
                pix = QPixmap(payload_file)
                self.lb_preview.setPixmap(
                    pix.scaled(self.lb_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.lb_preview.setText(f"Đã lưu payload:\n{payload_file}")
            QMessageBox.information(self, "Extract xong", f"Đã khôi phục payload:\n{payload_file}")
        except Exception as e:
            QMessageBox.critical(self, "Extract thất bại", str(e))


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