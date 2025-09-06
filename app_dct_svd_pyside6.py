# PySide6 variant of the same app
import os, sys, json
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
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
