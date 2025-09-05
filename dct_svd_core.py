# dct_svd_core.py
# DCT–SVD watermarking for IMAGES (non-blind):
# - Supports gray/color IMAGE watermark
# - Supports TEXT/JSON payloads by mapping bytes -> bit-image internally (gray pipeline)
#
# Outputs:
#   - stego.png (lossless)
#   - meta.npz  (stores SVD data + mode + payload_type + alpha + shape)

import os, json
import cv2
import numpy as np
from typing import Tuple, Optional

def _read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không mở được ảnh: {path}")
    return bgr

def _to_Y(bgr: np.ndarray):
    YCrCb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return YCrCb[:,:,0].astype(np.float32), YCrCb

def _from_Y(Y: np.ndarray, YCrCb_ref: np.ndarray) -> np.ndarray:
    out = YCrCb_ref.copy().astype(np.float32)
    H, W = out.shape[:2]
    out[:H,:W,0] = np.clip(Y[:H,:W], 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

def dct2(x: np.ndarray) -> np.ndarray:
    return cv2.dct(x.astype(np.float32))

def idct2(X: np.ndarray) -> np.ndarray:
    return cv2.idct(X.astype(np.float32))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32)-b.astype(np.float32))**2)
    if mse <= 1e-12: return 99.0
    return 10.0 * np.log10(255.0**2 / mse)

def ssim(img1, img2):
    # Simplified SSIM (luminance only)
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

# ---------------- bytes <-> bit-image (for text/json) ----------------
def _bytes_to_bitimg(data: bytes, H: int, W: int) -> np.ndarray:
    # Prepend 4-byte little-endian length
    L = len(data)
    header = L.to_bytes(4, 'little', signed=False)
    bits = np.unpackbits(np.frombuffer(header+data, dtype=np.uint8))
    total = H*W
    if bits.size > total:
        raise ValueError(f"Payload quá dài ({bits.size} bits) > dung lượng bit ảnh ({total} bits). Dùng ảnh host lớn hơn hoặc rút gọn dữ liệu.")
    arr = np.zeros(total, dtype=np.uint8)
    arr[:bits.size] = bits
    img = arr.reshape(H, W) * 255  # 0/255
    return img.astype(np.float32)

def _bitimg_to_bytes(img: np.ndarray) -> bytes:
    bits = (img.ravel() > 127).astype(np.uint8)
    if bits.size < 32:
        return b""
    header_bytes = np.packbits(bits[:32]).tobytes()
    L = int.from_bytes(header_bytes, "little", signed=False)
    need = 32 + L*8
    need = min(need, bits.size)
    payload_bits = bits[32:need]
    # pad to byte
    if payload_bits.size % 8 != 0:
        payload_bits = np.pad(payload_bits, (0, 8 - (payload_bits.size % 8)))
    by = np.packbits(payload_bits).tobytes()
    return by[:L]

# ---------------- Core DCT-SVD ----------------
def embed(cover_path: str,
          wm_source: str,
          out_path: str,
          meta_path: str,
          alpha: float = 0.05,
          color: bool = False,
          payload_type: str = 'image',
          text_data: Optional[str] = None) -> Tuple[str,str,float,float]:
    """
    payload_type: 'image' | 'text' | 'json'
    - if 'image': wm_source is path to image; color flag controls RGB mode.
    - if 'text' or 'json': text_data used (if None, read from wm_source as text file)
    """
    cover = _read_image(cover_path)
    H, W = cover.shape[:2]

    if payload_type in ('text','json'):
        if text_data is None:
            if not os.path.isfile(wm_source):
                raise ValueError("Vui lòng nhập nội dung hoặc chọn file .txt/.json để nhúng.")
            with open(wm_source, 'r', encoding='utf-8', errors='ignore') as f:
                text_data = f.read()
        if payload_type == 'json':
            try:
                obj = json.loads(text_data)
                text_data = json.dumps(obj, ensure_ascii=False, separators=(',',':'))
            except Exception as e:
                raise ValueError(f"JSON không hợp lệ: {e}")
        data = text_data.encode('utf-8')
        wm_img = _bytes_to_bitimg(data, H, W)  # grayscale bit-image same size as host

        # Use GRAY pipeline on Y
        Y, YCrCb = _to_Y(cover)
        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        Wm = dct2(wm_img); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw))
        S_ = Sc.copy(); S_[:L] = Sc[:L] + alpha * Sw[:L]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)

        if not out_path.lower().endswith(".png"): out_path = os.path.splitext(out_path)[0] + "_stego.png"
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok: raise IOError("Ghi stego thất bại.")
        np.savez_compressed(meta_path, mode='gray', payload_type=payload_type,
                            Sc=Sc, Uw=Uw, Vwt=Vwt, shape=(H,W), alpha=alpha)
        ps = psnr(cover, stego); ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), Yw)
        return out_path, meta_path, ps, ss

    # IMAGE payload
    wm = cv2.imread(wm_source, cv2.IMREAD_COLOR)
    if wm is None: raise ValueError(f"Không mở được watermark: {wm_source}")
    wm = cv2.resize(wm, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)

    if not color:
        Y, YCrCb = _to_Y(cover)
        wm_gray = cv2.cvtColor(wm.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        Wm = dct2(wm_gray); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw)); S_ = Sc.copy(); S_[:L] = Sc[:L] + alpha*Sw[:L]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)
        if not out_path.lower().endswith(".png"): out_path = os.path.splitext(out_path)[0] + "_stego.png"
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok: raise IOError("Ghi stego thất bại.")
        np.savez_compressed(meta_path, mode='gray', payload_type='image',
                            Sc=Sc, Uw=Uw, Vwt=Vwt, shape=(H,W), alpha=alpha)
        ps = psnr(cover, stego); ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), Yw)
        return out_path, meta_path, ps, ss

    # COLOR image watermark (per-channel)
    b,g,r = cv2.split(cover.astype(np.float32))
    wb,wg,wr = cv2.split(wm)
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    Ub, Sb, Vbt = np.linalg.svd(Cb, full_matrices=False)
    Ug, Sg, Vgt = np.linalg.svd(Cg, full_matrices=False)
    Ur, Sr, Vrt = np.linalg.svd(Cr, full_matrices=False)

    WB = dct2(wb); UWb, SWb, VWbt = np.linalg.svd(WB, full_matrices=False)
    WG = dct2(wg); UWg, SWg, VWgt = np.linalg.svd(WG, full_matrices=False)
    WR = dct2(wr); UWr, SWr, VWrt = np.linalg.svd(WR, full_matrices=False)

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
    if not out_path.lower().endswith(".png"): out_path = os.path.splitext(out_path)[0] + "_stego.png"
    ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok: raise IOError("Ghi stego thất bại.")
    np.savez_compressed(meta_path, mode='color', payload_type='image',
                        Sb=Sb, Sg=Sg, Sr=Sr,
                        UWb=UWb, VWbt=VWbt,
                        UWg=UWg, VWgt=VWgt,
                        UWr=UWr, VWrt=VWrt,
                        shape=(H,W), alpha=alpha)
    ps = psnr(cover, stego); ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY))
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
        Wm_full[:h,:w] = Wm_hat[:h,:w]
        bits_img = idct2(Wm_full)
        if normalize:
            bits_img = cv2.normalize(bits_img, None, 0, 255, cv2.NORM_MINMAX)
        bits_img = np.clip(bits_img, 0, 255).astype(np.uint8)
        by = _bitimg_to_bytes(bits_img)
        if payload_type == 'json':
            try:
                obj = json.loads(by.decode('utf-8', errors='ignore'))
                text = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                text = by.decode('utf-8', errors='ignore')
            if not out_path.lower().endswith('.json'):
                out_path = os.path.splitext(out_path)[0] + '_data.json'
        else:
            text = by.decode('utf-8', errors='ignore')
            if not out_path.lower().endswith('.txt'):
                out_path = os.path.splitext(out_path)[0] + '_text.txt'
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text)
        return out_path

    if mode == 'gray':
        Y, _ = _to_Y(st); Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        Wm_hat = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)
        H, W = map(int, data['shape'])
        Wm_full = np.zeros((H,W), np.float32)
        Wm_full[:Wm_hat.shape[0], :Wm_hat.shape[1]] = Wm_hat
        wm = idct2(Wm_full)
        if normalize: wm = cv2.normalize(wm, None, 0, 255, cv2.NORM_MINMAX)
        wm = np.clip(wm, 0, 255).astype(np.uint8)
        if not out_path.lower().endswith('.png'): out_path = os.path.splitext(out_path)[0] + '_wm.png'
        ok = cv2.imwrite(out_path, wm);
        if not ok: raise IOError('Ghi watermark thất bại.')
        return out_path

    # color
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
    WB_full = np.zeros((Hs,Ws), np.float32); WG_full = WB_full.copy(); WR_full = WB_full.copy()
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
    if not out_path.lower().endswith('.png'): out_path = os.path.splitext(out_path)[0] + '_wm.png'
    ok = cv2.imwrite(out_path, out)
    if not ok: raise IOError('Ghi watermark thất bại.')
    return out_path

def detect(stego_path: str, meta_path: str, thresh: float=0.6) -> Tuple[bool, float]:
    data = np.load(meta_path, allow_pickle=False)
    mode = str(data.get('mode','gray'))
    alpha = float(data['alpha'])
    st = _read_image(stego_path).astype(np.float32)

    def _score(T: np.ndarray) -> float:
        T = T.astype(np.float32)
        if T.size == 0: return 0.0
        flat = T.ravel()
        if np.std(flat) < 1e-6: return 0.0
        return float(np.mean((flat-flat.mean())/(flat.std()+1e-6)))

    if mode == 'gray':
        Y, _ = _to_Y(st.astype(np.uint8))
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        L = min(len(Sc), len(S_cw), Uw.shape[0], Vwt.shape[0])
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        T = (Uw[:L,:L] @ np.diag(Sw_hat) @ Vwt[:L,:L]).astype(np.float32)
        s = _score(T); return bool(s>=thresh), float(s)

    # color
    b,g,r = cv2.split(st)
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

    TB = (UWb[:Lb,:Lb] @ np.diag(SWb_hat) @ VWbt[:Lb,:Lb]).astype(np.float32)
    TG = (UWg[:Lg,:Lg] @ np.diag(SWg_hat) @ VWgt[:Lg,:Lg]).astype(np.float32)
    TR = (UWr[:Lr,:Lr] @ np.diag(SWr_hat) @ VWrt[:Lr,:Lr]).astype(np.float32)
    s = (_score(TB)+_score(TG)+_score(TR))/3.0
    return bool(s>=thresh), float(s)
