# app_dct_svd.py
# Main app: gom 3 tab từ 3 file riêng
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    backend = "PySide6"
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    backend = "PyQt5"

# --- đảm bảo import nội bộ hoạt động dù chạy ở đâu ---
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import sys as _sys
from embed_tab import EmbedTab
from extract_tab import ExtractTab
from detect_tab import DetectTab

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DCT–SVD Watermarking (Images, Text, JSON) – {backend}")
        self.resize(900, 700)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(EmbedTab(),  "EMBED")
        tabs.addTab(ExtractTab(), "EXTRACT")
        tabs.addTab(DetectTab(),  "DETECT")
        self.setCentralWidget(tabs)

def main():
    app = QtWidgets.QApplication(_sys.argv)
    w = MainWindow()
    w.show()
    # chạy được cả PySide6 lẫn PyQt5
    if hasattr(app, "exec"):
        _sys.exit(app.exec())
    else:
        _sys.exit(app.exec_())

if __name__ == "__main__":
    main()
