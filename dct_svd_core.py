
# dct_svd_core_clean.py
# Clean DCT–SVD watermarking core for:
# - IMAGE watermark (gray/color)
# - TEXT / JSON payloads (via bit-image)
#
# Functions:
#   embed(cover_path, wm_source, out_path, meta_path, alpha=0.1, color=False, payload_type='image', text_data=None)
#   extract(stego_path, meta_path, out_path, normalize=True) -> str
#   detect(stego_path, meta_path, thresh=0.6) -> (bool, float)

import os, json
import cv2
import numpy as np
from typing import Tuple, Optional

# ---------- basic utils ----------

def _read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không mở được ảnh: {path}")
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
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12: return 99.0
    PIXEL_MAX = 255.0
    return 20.0 * np.log10(PIXEL_MAX / np.sqrt(mse))

def ssim(img1, img2) -> float:
    # simple SSIM on gray using gaussian blur windows
    if img1.ndim == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    kernel = (11,11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(img1, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel, sigma)
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.GaussianBlur(img1*img1, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1*img1, kernel, sigma) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)+1e-12)
    return float(np.mean(ssim_map))

# ---------- bit image for text/json ----------

_HEADER_BITS = 32  # 4-byte big-endian length

def _bytes_to_bitimg(data: bytes, H: int, W: int) -> np.ndarray:
    n = len(data)
    header = n.to_bytes(4, 'big')
    bits = []
    for byte in header + data:
        for i in range(8)[::-1]:  # MSB first
            bits.append((byte >> i) & 1)
    total = H*W
    bits = (bits + [0]*max(0, total - len(bits)))[:total]
    img = np.array(bits, dtype=np.uint8).reshape(H, W) * 255
    return img.astype(np.float32)

def _bitimg_to_bytes(img: np.ndarray) -> bytes:
    # img expected in {0,255} or any thresholded 0..255
    flat = img.flatten()
    bits = (flat > 127).astype(np.uint8)
    # read length
    h = 0
    for i in range(32):
        h = (h << 1) | int(bits[i])
    L = int(h)
    out = bytearray()
    cur = 0; cnt = 0
    for i in range(32, 32 + L*8):
        cur = (cur << 1) | int(bits[i])
        cnt += 1
        if cnt == 8:
            out.append(cur)
            cur = 0; cnt = 0
    return bytes(out)

# ---------- core API ----------

def embed(cover_path: str,
          wm_source: str,
          out_path: str,
          meta_path: str,
          alpha: float = 0.1,
          color: bool = False,
          payload_type: str = 'image',
          text_data: Optional[str] = None):
    cover = _read_image(cover_path)
    H, W = cover.shape[:2]

    if payload_type in ('text','json'):
        if text_data is None:
            if not os.path.isfile(wm_source):
                raise ValueError("Vui lòng nhập nội dung hoặc chọn file .txt/.json để nhúng.")
            with open(wm_source, 'r', encoding='utf-8', errors='ignore') as f:
                text_data = f.read()
        if payload_type == 'json':
            # validate & pack compact
            obj = json.loads(text_data)
            text_data = json.dumps(obj, ensure_ascii=False, separators=(',',':'))
        data = text_data.encode('utf-8')
        wm_img = _bytes_to_bitimg(data, H, W)

        # GRAY pipeline in Y
        Y, YCrCb = _to_Y(cover)
        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        Wm = dct2(wm_img); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw))
        S_ = Sc.copy(); S_[:L] = Sc[:L] + alpha * Sw[:L]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)

        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_stego.png'
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok: raise IOError("Ghi stego thất bại.")
        np.savez_compressed(meta_path, mode='gray', payload_type=payload_type,
                            Sc=Sc, Uw=Uw, Vwt=Vwt, Sw=Sw, shape=(H,W), alpha=alpha)
        ps = psnr(cover, stego); ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY))
        return out_path, meta_path, ps, ss

    # IMAGE watermark path
    wm = _read_image(wm_source)
    if color:
        # color pipeline (3 channels separately)
        b, g, r = cv2.split(cover.astype(np.float32))
        wb, wg, wr = cv2.split(cv2.resize(wm, (W,H), interpolation=cv2.INTER_AREA).astype(np.float32))

        Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
        Ub, Sb, Vbt = np.linalg.svd(Cb, full_matrices=False)
        Ug, Sg, Vgt = np.linalg.svd(Cg, full_matrices=False)
        Ur, Sr, Vrt = np.linalg.svd(Cr, full_matrices=False)

        CWb, CWg, CWr = dct2(wb), dct2(wg), dct2(wr)
        UWb, SWb, VWbt = np.linalg.svd(CWb, full_matrices=False)
        UWg, SWg, VWgt = np.linalg.svd(CWg, full_matrices=False)
        UWr, SWr, VWrt = np.linalg.svd(CWr, full_matrices=False)

        Lb = min(len(Sb), len(SWb)); Lg = min(len(Sg), len(SWg)); Lr = min(len(Sr), len(SWr))
        Sb_, Sg_, Sr_ = Sb.copy(), Sg.copy(), Sr.copy()
        Sb_[:Lb] = Sb[:Lb] + alpha*SWb[:Lb]
        Sg_[:Lg] = Sg[:Lg] + alpha*SWg[:Lg]
        Sr_[:Lr] = Sr[:Lr] + alpha*SWr[:Lr]

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
        if not ok: raise IOError("Ghi stego thất bại.")
        np.savez_compressed(meta_path, mode='color', payload_type='image',
                            Sb=Sb, Sg=Sg, Sr=Sr,
                            UWb=UWb, VWbt=VWbt, SWb=SWb,
                            UWg=UWg, VWgt=VWgt, SWg=SWg,
                            UWr=UWr, VWrt=VWrt, SWr=SWr,
                            shape=(H,W), alpha=alpha)
        ps = psnr(cover, stego); ss = ssim(cover, stego)
        return out_path, meta_path, ps, ss
    else:
        # gray pipeline in Y
        Y, YCrCb = _to_Y(cover)
        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        wy = cv2.cvtColor(cv2.resize(wm, (W,H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        Wm = dct2(wy); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw))
        S_ = Sc.copy(); S_[:L] = Sc[:L] + alpha*Sw[:L]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_stego.png'
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok: raise IOError("Ghi stego thất bại.")
        np.savez_compressed(meta_path, mode='gray', payload_type='image',
                            Sc=Sc, Uw=Uw, Vwt=Vwt, Sw=Sw, shape=(H,W), alpha=alpha)
        ps = psnr(cover, stego); ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), Yw)
        return out_path, meta_path, ps, ss

def extract(stego_path: str, meta_path: str, out_path: str, normalize: bool=True) -> str:
    data = np.load(meta_path, allow_pickle=False)
    mode = str(data.get('mode','gray'))
    payload_type = str(data.get('payload_type','image'))
    alpha = float(data['alpha'])
    st = _read_image(stego_path)

    if payload_type in ('text','json'):
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        Wm_hat = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)
        H, W = map(int, data['shape'])
        Wm_full = np.zeros((H,W), np.float32)
        h = min(Wm_hat.shape[0], H); w = min(Wm_hat.shape[1], W)
        Wm_full[:h, :w] = Wm_hat[:h, :w]
        bits = idct2(Wm_full)
        # strengthen contrast and binarize
        bits = cv2.normalize(bits, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bits = cv2.threshold(bits, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        by = _bitimg_to_bytes(bits)
        text = by.decode('utf-8', errors='ignore')
        if payload_type == 'json':
            # canonicalize if valid
            try:
                obj = json.loads(text)
                text = json.dumps(obj, ensure_ascii=False, separators=(',',':'))
                if not out_path.lower().endswith('.json'):
                    out_path = os.path.splitext(out_path)[0] + '_data.json'
            except Exception:
                if not out_path.lower().endswith('.txt'):
                    out_path = os.path.splitext(out_path)[0] + '_text.txt'
        else:
            if not out_path.lower().endswith('.txt'):
                out_path = os.path.splitext(out_path)[0] + '_text.txt'
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text)
        return out_path

    if mode == 'gray':
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        Wm_hat = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)
        H, W = map(int, data['shape'])
        Wm_full = np.zeros((H,W), np.float32)
        hh = min(Wm_hat.shape[0], H); ww = min(Wm_hat.shape[1], W)
        Wm_full[:hh, :ww] = Wm_hat[:hh, :ww]
        wm = idct2(Wm_full)
        if normalize:
            wm = cv2.normalize(wm, None, 0, 255, cv2.NORM_MINMAX)
        wm = np.clip(wm, 0, 255).astype(np.uint8)
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_wm.png'
        ok = cv2.imwrite(out_path, wm)
        if not ok: raise IOError('Ghi watermark thất bại.')
        return out_path

    # mode == 'color'
    b,g,r = cv2.split(st.astype(np.float32))
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    _, S_cwb, _ = np.linalg.svd(Cb, full_matrices=False)
    _, S_cwg, _ = np.linalg.svd(Cg, full_matrices=False)
    _, S_cwr, _ = np.linalg.svd(Cr, full_matrices=False)
    Sb, Sg, Sr = data['Sb'], data['Sg'], data['Sr']
    UWb, VWbt = data['UWb'], data['VWbt']
    UWg, VWgt = data['UWg'], data['VWgt']
    UWr, VWrt = data['UWr'], data['VWrt']

    Lb = min(len(Sb), len(S_cwb), UWb.shape[0], VWbt.shape[0])
    Lg = min(len(Sg), len(S_cwg), UWg.shape[0], VWgt.shape[0])
    Lr = min(len(Sr), len(S_cwr), UWr.shape[0], VWrt.shape[0])

    SWb_hat = (S_cwb[:Lb] - Sb[:Lb]) / max(alpha, 1e-8)
    SWg_hat = (S_cwg[:Lg] - Sg[:Lg]) / max(alpha, 1e-8)
    SWr_hat = (S_cwr[:Lr] - Sr[:Lr]) / max(alpha, 1e-8)

    WB_hat = (UWb[:Lb,:Lb] @ np.diag(SWb_hat) @ VWbt[:Lb,:Lb]).astype(np.float32)
    WG_hat = (UWg[:Lg,:Lg] @ np.diag(SWg_hat) @ VWgt[:Lg,:Lg]).astype(np.float32)
    WR_hat = (UWr[:Lr,:Lr] @ np.diag(SWr_hat) @ VWrt[:Lr,:Lr]).astype(np.float32)

    Hs, Ws = map(int, data['shape'])
    WB_full = np.zeros((Hs,Ws), np.float32); WG_full = np.zeros((Hs,Ws), np.float32); WR_full = np.zeros((Hs,Ws), np.float32)
    WB_full[:WB_hat.shape[0], :WB_hat.shape[1]] = WB_hat
    WG_full[:WG_hat.shape[0], :WG_hat.shape[1]] = WG_hat
    WR_full[:WR_hat.shape[0], :WR_hat.shape[1]] = WR_hat

    wb = idct2(WB_full); wg = idct2(WG_full); wr = idct2(WR_full)
    if normalize:
        wb = cv2.normalize(wb, None, 0, 255, cv2.NORM_MINMAX)
        wg = cv2.normalize(wg, None, 0, 255, cv2.NORM_MINMAX)
        wr = cv2.normalize(wr, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.merge([np.clip(wb,0,255).astype(np.uint8),
                     np.clip(wg,0,255).astype(np.uint8),
                     np.clip(wr,0,255).astype(np.uint8)])
    if not out_path.lower().endswith('.png'):
        out_path = os.path.splitext(out_path)[0] + '_wm.png'
    ok = cv2.imwrite(out_path, out)
    if not ok: raise IOError('Ghi watermark thất bại.')
    return out_path

def _nc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    if a.size == 0 or b.size == 0: return 0.0
    a = a - np.mean(a); b = b - np.mean(b)
    den = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / den)

def detect(stego_path: str, meta_path: str, thresh: float = 0.6):
    data = np.load(meta_path, allow_pickle=False)
    mode = str(data.get('mode','gray'))
    payload_type = str(data.get('payload_type','image'))
    alpha = float(data['alpha'])
    st = _read_image(stego_path)

    if mode == 'gray':
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']
        L = min(len(Sc), len(S_cw))
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        if 'Sw' in data:
            Sw_orig = data['Sw'][:L]
            score = _nc(Sw_orig, Sw_hat)
        else:
            score = float(np.mean(np.abs(Sw_hat) > 0))
        return bool(score >= thresh), float(score)

    # color
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

    nc_b = _nc(SWb[:Lb], SWb_hat)
    nc_g = _nc(SWg[:Lg], SWg_hat)
    nc_r = _nc(SWr[:Lr], SWr_hat)
    score = (nc_b + nc_g + nc_r) / 3.0
    return bool(score >= thresh), float(score)
