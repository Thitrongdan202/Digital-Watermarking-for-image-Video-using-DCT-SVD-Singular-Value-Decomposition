
# dct_svd_core.py
# DCT–SVD watermarking for IMAGES (non-blind): requires meta.npz to extract/detect.
# - Embed:    cover + watermark -> stego.png + meta.npz
# - Extract:  stego.png + meta.npz -> recovered watermark.png
# - Detect:   stego.png + meta.npz -> NC score & boolean

import os
import cv2
import numpy as np
from typing import Tuple

def _read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không mở được ảnh: {path}")
    return bgr

def _to_Y(bgr: np.ndarray) -> np.ndarray:
    YCrCb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return YCrCb[:,:,0].astype(np.float32), YCrCb

def _from_Y(Y: np.ndarray, YCrCb_ref: np.ndarray) -> np.ndarray:
    out = YCrCb_ref.copy()
    H, W = out.shape[:2]
    out = out.astype(np.float32)
    out[:H,:W,0] = np.clip(Y[:H,:W], 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

def dct2(x: np.ndarray) -> np.ndarray:
    return cv2.dct(x.astype(np.float32))

def idct2(X: np.ndarray) -> np.ndarray:
    return cv2.idct(X.astype(np.float32))

def _resize_to(wm: np.ndarray, shape: Tuple[int,int]) -> np.ndarray:
    return cv2.resize(wm, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32)-b.astype(np.float32))**2)
    if mse <= 1e-12: return 99.0
    return 10.0 * np.log10(255.0**2 / mse)

def ssim(img1, img2):
    # Simplified SSIM (8x8 window, luminance only)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    img1 = img1.astype(np.float32); img2 = img2.astype(np.float32)
    mu1 = cv2.GaussianBlur(img1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11,11), 1.5)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1*img1, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, (11,11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1*img2, (11,11), 1.5) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(np.mean(ssim_map))

def _normcorr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    if np.all(a==0) or np.all(b==0): return 0.0
    a = (a - a.mean())/(a.std()+1e-6)
    b = (b - b.mean())/(b.std()+1e-6)
    return float(np.dot(a, b) / (len(a)))

def embed(cover_path: str, wm_path: str, out_path: str, meta_path: str, alpha: float=0.05) -> Tuple[str,str,float,float]:
    # Read cover & watermark (grayscale)
    cover_bgr = _read_image(cover_path)
    Y, YCrCb = _to_Y(cover_bgr)
    H, W = Y.shape
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    if wm is None: raise ValueError(f"Không mở được watermark: {wm_path}")
    wm = _resize_to(wm, (H, W)).astype(np.float32)

    # DCT + SVD
    C = dct2(Y)
    Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)

    Wm = dct2(wm)
    Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)

    # Embed on singular values
    L = min(len(Sc), len(Sw))
    S_ = Sc.copy()
    S_[:L] = Sc[:L] + alpha * Sw[:L]

    Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
    Yw = idct2(Cw)
    st_bgr = _from_Y(Yw, YCrCb)

    # Save stego + meta
    if not out_path.lower().endswith(".png"):
        out_path = os.path.splitext(out_path)[0] + "_stego.png"
    ok = cv2.imwrite(out_path, st_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok: raise IOError("Ghi stego thất bại.")

    np.savez_compressed(meta_path, Sc=Sc, Uw=Uw, Vwt=Vwt, shape=(H,W), alpha=alpha)
    # Quality
    ps = psnr(_from_Y(Y, YCrCb), st_bgr)
    ss = ssim(Y, Yw)
    return out_path, meta_path, ps, ss

def extract(stego_path: str, meta_path: str, out_wm_path: str) -> str:
    data = np.load(meta_path, allow_pickle=False)
    Sc = data["Sc"]; Uw = data["Uw"]; Vwt = data["Vwt"]
    alpha = float(data["alpha"])
    # Read stego
    st_bgr = _read_image(stego_path)
    Y, _ = _to_Y(st_bgr)

    # DCT + SVD
    Cw = dct2(Y)
    Ucw, S_cw, Vcwt = np.linalg.svd(Cw, full_matrices=False)

    # Recover watermark singular values
    L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
    Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)

    # Reconstruct watermark in DCT domain and invert
    Wm_hat = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)

    # Pad/crop to original shape using zeros if needed
    H = int(data["shape"][0]); W = int(data["shape"][1])
    Wm_full = np.zeros((H,W), np.float32)
    h = min(Wm_hat.shape[0], H); w = min(Wm_hat.shape[1], W)
    Wm_full[:h,:w] = Wm_hat[:h,:w]

    wm_rec = idct2(Wm_full)
    wm_rec = np.clip(wm_rec, 0, 255).astype(np.uint8)

    if not out_wm_path.lower().endswith(".png"):
        out_wm_path = os.path.splitext(out_wm_path)[0] + "_wm.png"
    ok = cv2.imwrite(out_wm_path, wm_rec)
    if not ok: raise IOError("Ghi watermark thất bại.")
    return out_wm_path

def detect(stego_path: str, meta_path: str, thresh: float=0.6) -> Tuple[bool, float]:
    # Try to reconstruct watermark and compute normalized correlation with its DCT-spectral estimate
    data = np.load(meta_path, allow_pickle=False)
    Sc = data["Sc"]; Uw = data["Uw"]; Vwt = data["Vwt"]; alpha = float(data["alpha"])
    st_bgr = _read_image(stego_path)
    Y, _ = _to_Y(st_bgr)
    Cw = dct2(Y); Ucw, S_cw, Vcwt = np.linalg.svd(Cw, full_matrices=False)
    L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
    Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)

    # Build an internal watermark "template" in spectral space for correlation
    T = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)
    score = _normcorr(T, T.mean() * np.ones_like(T))  # simple energy-normalized check
    present = bool(score >= thresh)
    return present, float(score)
