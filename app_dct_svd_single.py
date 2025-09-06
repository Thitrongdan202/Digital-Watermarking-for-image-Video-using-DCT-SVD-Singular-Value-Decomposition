
# app_dct_svd_single.py — One-file app (Images + Password, no EMBED preview)
# Requires: pip install opencv-python PySide6
import sys, os, cv2, numpy as np, hashlib, hmac, inspect
from typing import Optional
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QFileDialog, QTabWidget, QCheckBox, QSlider, QGroupBox,
                               QDoubleSpinBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

# ----------------------- SECURE CORE (embedded) -----------------------
K_FRAC_DEFAULT = 0.6  # embed top 60% singular values

def _read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Không mở được ảnh: {path}')
    return bgr

def _to_Y(bgr: np.ndarray):
    YCrCb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(YCrCb)
    return Y.astype(np.float32), YCrCb

def _from_Y(Yw: np.ndarray, YCrCb_ref: np.ndarray) -> np.ndarray:
    Yw = np.clip(Yw, 0, 255).astype(np.uint8)
    _, Cr, Cb = cv2.split(YCrCb_ref)
    out = cv2.merge([Yw, Cr, Cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

def dct2(x: np.ndarray) -> np.ndarray:
    return cv2.dct(x.astype(np.float32))

def idct2(X: np.ndarray) -> np.ndarray:
    return cv2.idct(X.astype(np.float32))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12: return 99.0
    return 20.0 * np.log10(255.0 / max(np.sqrt(mse), 1e-12))

def ssim(img1, img2) -> float:
    if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float32); img2 = img2.astype(np.float32)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    k = (11, 11); s = 1.5
    mu1 = cv2.GaussianBlur(img1, k, s); mu2 = cv2.GaussianBlur(img2, k, s)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1*img1, k, s) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, k, s) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1*img2, k, s) - mu1_mu2
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    return float(np.mean(num / den))

def _derive_key(password: str, nonce: bytes) -> bytes:
    return hashlib.sha256(password.encode('utf-8') + nonce).digest()

def _rng_from_key(key: bytes) -> np.random.Generator:
    seed = int.from_bytes(key[:8], 'big', signed=False)
    return np.random.default_rng(seed)

def _permute(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    H, W = img.shape[:2]
    idx = np.arange(H*W)
    rng.shuffle(idx)
    flat = img.reshape(-1)
    scrambled = flat[idx].reshape(H, W)
    return scrambled.astype(np.float32), idx

def _unpermute(img_scrambled: np.ndarray, idx: np.ndarray) -> np.ndarray:
    H, W = img_scrambled.shape[:2]
    flat = img_scrambled.reshape(-1)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(idx.size)
    restored = flat[inv].reshape(H, W)
    return restored

def _hmac_check(key: bytes, parts: list[bytes]) -> bytes:
    h = hmac.new(key, b'', hashlib.sha256)
    for p in parts:
        h.update(p)
    return h.digest()

def _enhance_gray(img_u8: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        e = clahe.apply(img_u8)
    except Exception:
        e = img_u8
    blur = cv2.GaussianBlur(e, (0,0), 1.0)
    sharp = cv2.addWeighted(e, 1.25, blur, -0.25, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _enhance_color(img_bgr_u8: np.ndarray) -> np.ndarray:
    try:
        ycc = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        ycc = cv2.merge([y, cr, cb])
        e = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    except Exception:
        e = img_bgr_u8
    blur = cv2.GaussianBlur(e, (0,0), 1.0)
    sharp = cv2.addWeighted(e, 1.15, blur, -0.15, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def embed(cover_path: str, wm_source: str, out_path: str, meta_path: str,
          alpha: float = 0.1, color: bool = False, password: Optional[str] = None,
          kfrac: float = K_FRAC_DEFAULT):
    if not password:
        raise ValueError('Vui lòng nhập mật khẩu để nhúng.')
    cover = _read_image(cover_path); H, W = cover.shape[:2]
    wm = _read_image(wm_source); wm = cv2.resize(wm, (W, H), interpolation=cv2.INTER_AREA)
    nonce = os.urandom(8); key = _derive_key(password, nonce); rng = _rng_from_key(key)

    if color:
        b, g, r = cv2.split(cover.astype(np.float32))
        wb, wg, wr = cv2.split(wm.astype(np.float32))
        idx = np.arange(H*W); rng.shuffle(idx)
        def _shuffle(x): return x.reshape(-1)[idx].reshape(H, W).astype(np.float32)
        wb_s, wg_s, wr_s = _shuffle(wb), _shuffle(wg), _shuffle(wr)
        Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
        Ub, Sb, Vbt = np.linalg.svd(Cb, full_matrices=False)
        Ug, Sg, Vgt = np.linalg.svd(Cg, full_matrices=False)
        Ur, Sr, Vrt = np.linalg.svd(Cr, full_matrices=False)
        CWb, CWg, CWr = dct2(wb_s), dct2(wg_s), dct2(wr_s)
        UWb, SWb, VWbt = np.linalg.svd(CWb, full_matrices=False)
        UWg, SWg, VWgt = np.linalg.svd(CWg, full_matrices=False)
        UWr, SWr, VWrt = np.linalg.svd(CWr, full_matrices=False)
        Lb = min(len(Sb), len(SWb)); Lg = min(len(Sg), len(SWg)); Lr = min(len(Sr), len(SWr))
        Kb = max(8, int(kfrac * Lb)); Kg = max(8, int(kfrac * Lg)); Kr = max(8, int(kfrac * Lr))
        Sb_, Sg_, Sr_ = Sb.copy(), Sg.copy(), Sr.copy()
        Sb_[:Kb] = Sb[:Kb] + alpha*SWb[:Kb]
        Sg_[:Kg] = Sg[:Kg] + alpha*SWg[:Kg]
        Sr_[:Kr] = Sr[:Kr] + alpha*SWr[:Kr]
        Cbw = (Ub @ np.diag(Sb_) @ Vbt).astype(np.float32)
        Cgw = (Ug @ np.diag(Sg_) @ Vgt).astype(np.float32)
        Crw = (Ur @ np.diag(Sr_) @ Vrt).astype(np.float32)
        bw, gw, rw = idct2(Cbw), idct2(Cgw), idct2(Crw)
        stego = cv2.merge([np.clip(bw,0,255).astype(np.uint8),
                           np.clip(gw,0,255).astype(np.uint8),
                           np.clip(rw,0,255).astype(np.uint8)])
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_stego.png'
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok: raise IOError('Ghi stego thất bại.')
        digest = _hmac_check(key, [
            Sb.tobytes(), Sg.tobytes(), Sr.tobytes(),
            UWb.tobytes(), UWg.tobytes(), UWr.tobytes(),
            VWbt.tobytes(), VWgt.tobytes(), VWrt.tobytes()
        ])
        np.savez_compressed(meta_path,
            mode='color', payload_type='image',
            Sb=Sb, Sg=Sg, Sr=Sr,
            UWb=UWb, VWbt=VWbt, SWb=SWb,
            UWg=UWg, VWgt=VWgt, SWg=SWg,
            UWr=UWr, VWrt=VWrt, SWr=SWr,
            shape=(H,W), alpha=float(alpha), kfrac=float(kfrac),
            nonce=np.frombuffer(nonce, dtype=np.uint8),
            digest=np.frombuffer(digest, dtype=np.uint8)
        )
        return out_path, meta_path, psnr(cover, stego), ssim(cover, stego)
    else:
        Y, YCrCb = _to_Y(cover)
        wy = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY).astype(np.float32)
        wy_s, idx = _permute(wy, rng)
        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        Wm = dct2(wy_s); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw)); K = max(8, int(kfrac * L))
        S_ = Sc.copy(); S_[:K] = Sc[:K] + alpha*Sw[:K]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_stego.png'
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok: raise IOError('Ghi stego thất bại.')
        digest = _hmac_check(key, [Sc.tobytes(), Uw.tobytes(), Vwt.tobytes()])
        np.savez_compressed(meta_path,
            mode='gray', payload_type='image',
            Sc=Sc, Uw=Uw, Vwt=Vwt, Sw=Sw,
            shape=(H, W), alpha=float(alpha), kfrac=float(kfrac),
            nonce=np.frombuffer(nonce, dtype=np.uint8),
            digest=np.frombuffer(digest, dtype=np.uint8)
        )
        return out_path, meta_path, psnr(cover, stego), ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), Yw)

def extract(stego_path: str, meta_path: str, out_path: str, password: str, normalize: bool = True) -> str:
    if not password:
        raise ValueError('Vui lòng nhập mật khẩu để giải trích.')
    data = np.load(meta_path, allow_pickle=False)
    mode  = str(data['mode']); alpha = float(data['alpha'])
    H, W  = map(int, data['shape'])
    nonce  = bytes(bytearray(data['nonce'].astype(np.uint8).tolist()))
    digest = bytes(bytearray(data['digest'].astype(np.uint8).tolist()))
    key    = _derive_key(password, nonce)
    st = _read_image(stego_path)

    if mode == 'gray':
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        expected = _hmac_check(key, [Sc.tobytes(), Uw.tobytes(), Vwt.tobytes()])
        if not hmac.compare_digest(expected, digest):
            raise ValueError('Sai mật khẩu hoặc meta không khớp.')
        L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
        kfrac = float(data.get('kfrac', K_FRAC_DEFAULT)); K = max(8, int(kfrac * L))
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        Sw_hat[K:] = 0
        Wm_hat = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)
        Wm_full = np.zeros((H, W), np.float32)
        hh = min(Wm_hat.shape[0], H); ww = min(Wm_hat.shape[1], W)
        Wm_full[:hh, :ww] = Wm_hat[:hh, :ww]
        wy_s = idct2(Wm_full)
        rng = _rng_from_key(key); idx = np.arange(H*W); rng.shuffle(idx)
        wy = _unpermute(wy_s, idx)
        if normalize: wy = cv2.normalize(wy, None, 0, 255, cv2.NORM_MINMAX)
        wy = np.clip(wy, 0, 255).astype(np.uint8)
        try: wy = cv2.fastNlMeansDenoising(wy, None, 7, 7, 21)
        except Exception: pass
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_wm.png'
        wy = _enhance_gray(wy)
        ok = cv2.imwrite(out_path, wy)
        if not ok: raise IOError('Ghi watermark thất bại.')
        return out_path

    b,g,r = cv2.split(st.astype(np.float32))
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    _, S_cwb, _ = np.linalg.svd(Cb, full_matrices=False)
    _, S_cwg, _ = np.linalg.svd(Cg, full_matrices=False)
    _, S_cwr, _ = np.linalg.svd(Cr, full_matrices=False)
    Sb, Sg, Sr = data['Sb'], data['Sg'], data['Sr']
    UWb, VWbt = data['UWb'], data['VWbt']
    UWg, VWgt = data['UWg'], data['VWgt']
    UWr, VWrt = data['UWr'], data['VWrt']
    expected = _hmac_check(key, [
        Sb.tobytes(), Sg.tobytes(), Sr.tobytes(),
        UWb.tobytes(), UWg.tobytes(), UWr.tobytes(),
        VWbt.tobytes(), VWgt.tobytes(), VWrt.tobytes()
    ])
    if not hmac.compare_digest(expected, digest):
        raise ValueError('Sai mật khẩu hoặc meta không khớp.')
    Lb = min(len(Sb), len(S_cwb), UWb.shape[0], VWbt.shape[0])
    Lg = min(len(Sg), len(S_cwg), UWg.shape[0], VWgt.shape[0])
    Lr = min(len(Sr), len(S_cwr), UWr.shape[0], VWrt.shape[0])
    kfrac = float(data.get('kfrac', K_FRAC_DEFAULT))
    Kb = max(8, int(kfrac * Lb)); Kg = max(8, int(kfrac * Lg)); Kr = max(8, int(kfrac * Lr))
    SWb_hat = (S_cwb[:Lb] - Sb[:Lb]) / max(alpha, 1e-8)
    SWg_hat = (S_cwg[:Lg] - Sg[:Lg]) / max(alpha, 1e-8)
    SWr_hat = (S_cwr[:Lr] - Sr[:Lr]) / max(alpha, 1e-8)
    SWb_hat[Kb:] = 0; SWg_hat[Kg:] = 0; SWr_hat[Kr:] = 0
    WB_hat = (UWb[:Lb,:Lb] @ np.diag(SWb_hat) @ VWbt[:Lb,:Lb]).astype(np.float32)
    WG_hat = (UWg[:Lg,:Lg] @ np.diag(SWg_hat) @ VWgt[:Lg,:Lg]).astype(np.float32)
    WR_hat = (UWr[:Lr,:Lr] @ np.diag(SWr_hat) @ VWrt[:Lr,:Lr]).astype(np.float32)
    WB_full = np.zeros((H,W), np.float32); WG_full = np.zeros((H,W), np.float32); WR_full = np.zeros((H,W), np.float32)
    WB_full[:WB_hat.shape[0], :WB_hat.shape[1]] = WB_hat
    WG_full[:WG_hat.shape[0], :WG_hat.shape[1]] = WG_hat
    WR_full[:WR_hat.shape[0], :WR_hat.shape[1]] = WR_hat
    wb_s = idct2(WB_full); wg_s = idct2(WG_full); wr_s = idct2(WR_full)
    rng = _rng_from_key(key); idx = np.arange(H*W); rng.shuffle(idx)
    def _un(x): return _unpermute(x, idx)
    wb, wg, wr = _un(wb_s), _un(wg_s), _un(wr_s)
    if normalize:
        wb = cv2.normalize(wb, None, 0, 255, cv2.NORM_MINMAX)
        wg = cv2.normalize(wg, None, 0, 255, cv2.NORM_MINMAX)
        wr = cv2.normalize(wr, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.merge([np.clip(wb,0,255).astype(np.uint8),
                     np.clip(wg,0,255).astype(np.uint8),
                     np.clip(wr,0,255).astype(np.uint8)])
    try: out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    except Exception: pass
    out = _enhance_color(out)
    if not out_path.lower().endswith('.png'):
        out_path = os.path.splitext(out_path)[0] + '_wm.png'
    ok = cv2.imwrite(out_path, out)
    if not ok: raise IOError('Ghi watermark thất bại.')
    return out_path

def _nc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    if a.size == 0 or b.size == 0: return 0.0
    a = a - np.mean(a); b = b - np.mean(b)
    den = np.linalg.norm(a)*np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / den)

def detect(stego_path: str, meta_path: str, thresh: float = 0.6):
    data = np.load(meta_path, allow_pickle=False)
    mode  = str(data['mode']); alpha = float(data['alpha'])
    st = _read_image(stego_path)
    if mode == 'gray':
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Sw = data['Sw']
        L = min(len(Sc), len(S_cw), len(Sw))
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        score = _nc(Sw[:L], Sw_hat)
        return bool(score >= thresh), float(score)
    b,g,r = cv2.split(st.astype(np.float32))
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    _, S_cwb, _ = np.linalg.svd(Cb, full_matrices=False)
    _, S_cwg, _ = np.linalg.svd(Cg, full_matrices=False)
    _, S_cwr, _ = np.linalg.svd(Cr, full_matrices=False)
    Sb, Sg, Sr = data['Sb'], data['Sg'], data['Sr']
    SWb, SWg, SWr = data['SWb'], data['SWg'], data['SWr']
    Lb = min(len(Sb), len(S_cwb), len(SWb))
    Lg = min(len(Sg), len(S_cwg), len(SWg))
    Lr = min(len(Sr), len(S_cwr), len(SWr))
    SWb_hat = (S_cwb[:Lb] - Sb[:Lb]) / max(alpha, 1e-8)
    SWg_hat = (S_cwg[:Lg] - Sg[:Lg]) / max(alpha, 1e-8)
    SWr_hat = (S_cwr[:Lr] - Sr[:Lr]) / max(alpha, 1e-8)
    nc_b = _nc(SWb[:Lb], SWb_hat); nc_g = _nc(SWg[:Lg], SWg_hat); nc_r = _nc(SWr[:Lr], SWr_hat)
    score = (nc_b + nc_g + nc_r) / 3.0
    return bool(score >= thresh), float(score)

# ----------------------- GUI (no preview after EMBED) -----------------------
def is_image(p): return p and os.path.splitext(p)[1].lower() in (".png",".jpg",".jpeg",".bmp",".tiff",".tif",".webp")

def cv2_to_qpixmap(img_bgr):
    if img_bgr is None: return QPixmap()
    if img_bgr.ndim == 2: rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else: rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888))

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCT–SVD Watermarking (Images + Password) — single file")
        tabs = QTabWidget(self)

        # EMBED
        embed_tab = QWidget(); tabs.addTab(embed_tab, "EMBED")
        ev = QVBoxLayout(embed_tab)
        g1 = QGroupBox("Host Image"); v1 = QVBoxLayout(g1)
        self.ed_cover = QLineEdit(); b1 = QPushButton("Browse")
        r = QHBoxLayout(); r.addWidget(b1); r.addWidget(self.ed_cover); v1.addLayout(r)
        self.lbl_host = QLabel(); self.lbl_host.setAlignment(Qt.AlignCenter); self.lbl_host.setStyleSheet("background:#111; min-height:200px;")
        v1.addWidget(self.lbl_host); ev.addWidget(g1)

        g2 = QGroupBox("Watermark  •  Password"); v2 = QVBoxLayout(g2)
        self.ed_wm = QLineEdit(); b2 = QPushButton("Browse Image")
        r2 = QHBoxLayout(); r2.addWidget(b2); r2.addWidget(self.ed_wm); v2.addLayout(r2)
        r3 = QHBoxLayout(); r3.addWidget(QLabel("Password:")); self.ed_pwd = QLineEdit(); self.ed_pwd.setEchoMode(QLineEdit.Password); r3.addWidget(self.ed_pwd); v2.addLayout(r3)
        ev.addWidget(g2)

        g3 = QGroupBox("Settings"); h3 = QHBoxLayout(g3)
        self.sl = QSlider(Qt.Horizontal); self.sl.setRange(1,30); self.sl.setValue(12)
        self.sp = QDoubleSpinBox(); self.sp.setRange(0.01,0.30); self.sp.setDecimals(2); self.sp.setSingleStep(0.01); self.sp.setValue(0.12)
        self.cb = QCheckBox("Color watermark (RGB)"); self.lbla = QLabel("α = 0.12")
        h3.addWidget(QLabel("Alpha")); h3.addWidget(self.sl); h3.addWidget(self.sp); h3.addWidget(self.lbla); h3.addWidget(self.cb); ev.addWidget(g3)

        g4 = QGroupBox("Output"); v4 = QVBoxLayout(g4)
        self.ed_out = QLineEdit(); b3 = QPushButton("Save As")
        rr2 = QHBoxLayout(); rr2.addWidget(b3); rr2.addWidget(self.ed_out); v4.addLayout(rr2)
        self.lbl_info = QLabel("-"); v4.addWidget(self.lbl_info); ev.addWidget(g4)

        btnE = QPushButton("EMBED WATERMARK"); ev.addWidget(btnE)

        # EXTRACT
        xtab = QWidget(); tabs.addTab(xtab, "EXTRACT")
        xv = QVBoxLayout(xtab)
        self.ed_stego = QLineEdit(); bs1 = QPushButton("Browse stego")
        rx1 = QHBoxLayout(); rx1.addWidget(bs1); rx1.addWidget(self.ed_stego); xv.addLayout(rx1)
        self.ed_meta = QLineEdit(); bs2 = QPushButton("Browse meta (.npz)")
        rx2 = QHBoxLayout(); rx2.addWidget(bs2); rx2.addWidget(self.ed_meta); xv.addLayout(rx2)
        rx3 = QHBoxLayout(); rx3.addWidget(QLabel("Password:")); self.ed_pwd2 = QLineEdit(); self.ed_pwd2.setEchoMode(QLineEdit.Password); rx3.addWidget(self.ed_pwd2); xv.addLayout(rx3)
        self.ed_out2 = QLineEdit(); bs3 = QPushButton("Save As")
        rx4 = QHBoxLayout(); rx4.addWidget(bs3); rx4.addWidget(self.ed_out2); xv.addLayout(rx4)
        self.lbl_prev2 = QLabel(); self.lbl_prev2.setAlignment(Qt.AlignCenter); self.lbl_prev2.setStyleSheet("background:#111; min-height:200px;")
        xv.addWidget(self.lbl_prev2)
        btnX = QPushButton("EXTRACT (Password required)"); xv.addWidget(btnX)

        # DETECT
        dtab = QWidget(); tabs.addTab(dtab, "DETECT")
        dv = QVBoxLayout(dtab)
        self.ed_stego3 = QLineEdit(); bd1 = QPushButton("Browse stego")
        self.ed_meta3  = QLineEdit(); bd2 = QPushButton("Browse meta (.npz)")
        q1 = QHBoxLayout(); q1.addWidget(bd1); q1.addWidget(self.ed_stego3); dv.addLayout(q1)
        q2 = QHBoxLayout(); q2.addWidget(bd2); q2.addWidget(self.ed_meta3); dv.addLayout(q2)
        self.lbl_det = QLabel("Score: -"); dv.addWidget(self.lbl_det)
        btnD = QPushButton("DETECT"); dv.addWidget(btnD)

        layout = QVBoxLayout(self); layout.addWidget(tabs)

        # hooks
        b1.clicked.connect(self._pick_cover)
        b2.clicked.connect(lambda: self._pick(self.ed_wm))
        b3.clicked.connect(lambda: self._save(self.ed_out, "PNG (*.png)"))
        bs1.clicked.connect(lambda: self._pick(self.ed_stego))
        bs2.clicked.connect(lambda: self._pick(self.ed_meta, "NPZ (*.npz)"))
        bs3.clicked.connect(lambda: self._save(self.ed_out2, "PNG (*.png)"))
        bd1.clicked.connect(lambda: self._pick(self.ed_stego3))
        bd2.clicked.connect(lambda: self._pick(self.ed_meta3, "NPZ (*.npz)"))
        btnE.clicked.connect(self._do_embed); btnX.clicked.connect(self._do_extract); btnD.clicked.connect(self._do_detect)
        self.sl.valueChanged.connect(self._sync_from_slider); self.sp.valueChanged.connect(self._sync_from_spin)
        self._sync_from_slider(self.sl.value())

    def _pick(self, line, filt="Images (*.png *.jpg *.jpeg *.bmp)"):
        p = QFileDialog.getOpenFileName(self, "Choose file", "", filt)[0]
        if p: line.setText(p)

    def _save(self, line, filt):
        p = QFileDialog.getSaveFileName(self, "Save as", "", filt)[0]
        if p: line.setText(p)

    def _pick_cover(self):
        p = QFileDialog.getOpenFileName(self, "Choose host image", "", "Images (*.png *.jpg *.jpeg *.bmp)")[0]
        if not p: return
        self.ed_cover.setText(p)
        base, _ = os.path.splitext(p)
        self.ed_out.setText(base + "_stego.png")
        self.ed_stego.setText(base + "_stego.png")
        self.ed_meta.setText(base + "_stego_meta.npz")
        self.ed_out2.setText(base + "_wm.png")
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            pm = cv2_to_qpixmap(img).scaled(self.lbl_host.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_host.setPixmap(pm)

    def _sync_from_slider(self, v):
        a = max(1, min(30, v)) / 100.0
        if abs(self.sp.value() - a) > 1e-6:
            self.sp.blockSignals(True); self.sp.setValue(a); self.sp.blockSignals(False)
        self.lbla.setText(f"α = {a:.2f}")

    def _sync_from_spin(self, a):
        v = int(round(float(a)*100))
        if self.sl.value() != v:
            self.sl.blockSignals(True); self.sl.setValue(v); self.sl.blockSignals(False)
        self.lbla.setText(f"α = {float(a):.2f}")

    def _do_embed(self):
        try:
            cover = self.ed_cover.text().strip()
            wm    = self.ed_wm.text().strip()
            outp  = self.ed_out.text().strip() or "stego.png"
            if not is_image(cover): raise ValueError("Chọn ảnh cover hợp lệ")
            if not is_image(wm):    raise ValueError("Chọn ảnh watermark hợp lệ")
            meta  = os.path.splitext(outp)[0] + "_meta.npz"
            alpha = float(self.sp.value())
            out_path, meta_path, ps, ss = embed(cover, wm, outp, meta, alpha=alpha, color=self.cb.isChecked(), password=self.ed_pwd.text().strip())
            self.lbl_info.setText(f"Saved: {out_path}\nMeta: {meta_path}\nPSNR: {ps:.2f}  SSIM: {ss:.4f}")
        except Exception as e:
            self.lbl_info.setText("LỖI: " + str(e))

    def _do_extract(self):
        try:
            stego = self.ed_stego.text().strip(); meta = self.ed_meta.text().strip()
            outp  = self.ed_out2.text().strip() or "wm.png"
            outp = extract(stego, meta, outp, password=self.ed_pwd2.text().strip(), normalize=True)
            img = cv2.imread(outp, cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imread(outp, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img is not None:
                pm = cv2_to_qpixmap(img).scaled(self.lbl_prev2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_prev2.setPixmap(pm)
        except Exception as e:
            self.lbl_prev2.setText("LỖI: " + str(e))

    def _do_detect(self):
        try:
            ok, score = detect(self.ed_stego3.text().strip(), self.ed_meta3.text().strip(), thresh=0.6)
            self.lbl_det.setText(f"Score: {score:.4f} → {'Watermarked' if ok else 'Not found'}")
        except Exception as e:
            self.lbl_det.setText("LỖI: " + str(e))

def main():
    app = QApplication(sys.argv)
    w = App(); w.resize(900, 760); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
