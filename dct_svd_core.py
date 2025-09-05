# dct_svd_core.py
# DCT–SVD watermarking for IMAGES + TEXT/JSON (non-blind w.r.t. meta)
# - Ảnh watermark: hỗ trợ grayscale hoặc COLOR (RGB), *giữ nguyên kích thước gốc khi extract*
# - TEXT/JSON: chuyển bytes -> ảnh bit (kích thước = host), robust extract (percentile + Otsu + đảo bit fallback)
# - Detect: kiểm tra hiện diện watermark từ stego + meta (không cần ảnh gốc)

import os, json
import cv2
import numpy as np
from typing import Tuple, Optional


# ---------------------------- utils ----------------------------
def _read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không mở được ảnh: {path}")
    return bgr

def _to_Y(bgr: np.ndarray):
    YCrCb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return YCrCb[:, :, 0].astype(np.float32), YCrCb

def _from_Y(Y: np.ndarray, YCrCb_ref: np.ndarray) -> np.ndarray:
    out = YCrCb_ref.copy().astype(np.float32)
    H, W = out.shape[:2]
    out[:H, :W, 0] = np.clip(Y[:H, :W], 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

def dct2(x: np.ndarray) -> np.ndarray:
    return cv2.dct(x.astype(np.float32))

def idct2(X: np.ndarray) -> np.ndarray:
    return cv2.idct(X.astype(np.float32))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(255.0 ** 2 / mse)

def ssim(img1, img2) -> float:
    # SSIM đơn giản (chỉ luminance)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1 = img1.astype(np.float32); img2 = img2.astype(np.float32)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )
    return float(np.mean(ssim_map))


# -------------- bytes <-> bit image (for TEXT/JSON) --------------
def _bytes_to_bitimg(data: bytes, H: int, W: int) -> np.ndarray:
    # Prefix 4-byte little-endian length
    L = len(data)
    header = L.to_bytes(4, 'little', signed=False)
    bits = np.unpackbits(np.frombuffer(header + data, dtype=np.uint8))
    total = H * W
    if bits.size > total:
        raise ValueError(
            f"Payload quá dài ({bits.size} bits) > dung lượng ảnh ({total} bits). "
            f"Dùng host lớn hơn hoặc rút gọn dữ liệu."
        )
    arr = np.zeros(total, dtype=np.uint8)
    arr[:bits.size] = bits
    img = arr.reshape(H, W) * 255  # 0/255 image
    return img.astype(np.float32)

def _bitimg_to_bytes(img: np.ndarray) -> bytes:
    bits = (img.ravel() > 127).astype(np.uint8)
    if bits.size < 32:
        return b""
    header_bytes = np.packbits(bits[:32]).tobytes()
    L = int.from_bytes(header_bytes, "little", signed=False)
    need = 32 + L * 8
    need = min(need, bits.size)
    payload_bits = bits[32:need]
    if payload_bits.size % 8 != 0:
        payload_bits = np.pad(payload_bits, (0, 8 - (payload_bits.size % 8)))
    by = np.packbits(payload_bits).tobytes()
    return by[:L]


# -------------- robust binarize for TEXT/JSON extract --------------
def _robust_bits_from_idct(Wm_hat: np.ndarray) -> np.ndarray:
    """Sinh ảnh nhị phân 0/255 từ ảnh float bằng percentile scaling + Otsu."""
    img = Wm_hat.astype(np.float32)
    lo, hi = np.percentile(img, (2, 98))
    if hi - lo < 1e-6:
        lo = float(img.min()); hi = float(img.max() + 1e-6)
    scaled = (np.clip(img, lo, hi) - lo) / (hi - lo)
    scaled8 = (scaled * 255.0).astype(np.uint8)
    scaled8 = cv2.GaussianBlur(scaled8, (3, 3), 0)
    _, binary = cv2.threshold(scaled8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ----------------------------- EMBED -----------------------------
def embed(
    cover_path: str,
    wm_source: str,
    out_path: str,
    meta_path: str,
    alpha: float = 0.05,
    color: bool = False,
    payload_type: str = 'image',
    text_data: Optional[str] = None
) -> Tuple[str, str, float, float]:
    """
    payload_type: 'image' | 'text' | 'json'
    - 'image': wm_source là đường dẫn watermark ảnh; color=True để nhúng màu (B,G,R).
    - 'text'/'json': nếu text_data=None thì đọc từ wm_source (file .txt/.json).
    """
    cover = _read_image(cover_path)
    H, W = cover.shape[:2]

    # ----- TEXT / JSON -> bit-image (size = host) -----
    if payload_type in ('text', 'json'):
        if text_data is None:
            if not os.path.isfile(wm_source):
                raise ValueError("Nhập nội dung hoặc chọn file .txt/.json để nhúng.")
            with open(wm_source, 'r', encoding='utf-8', errors='ignore') as f:
                text_data = f.read()
        if payload_type == 'json':
            try:
                obj = json.loads(text_data)
                text_data = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
            except Exception as e:
                raise ValueError(f"JSON không hợp lệ: {e}")

        data = text_data.encode('utf-8')
        wm_img = _bytes_to_bitimg(data, H, W)  # grayscale bit image

        Y, YCrCb = _to_Y(cover)
        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        Wm = dct2(wm_img); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw))
        S_ = Sc.copy(); S_[:L] = Sc[:L] + alpha * Sw[:L]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)

        if not out_path.lower().endswith(".png"):
            out_path = os.path.splitext(out_path)[0] + "_stego.png"
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            raise IOError("Ghi stego thất bại.")
        np.savez_compressed(
            meta_path,
            mode='gray',
            payload_type=payload_type,
            Sc=Sc, Uw=Uw, Vwt=Vwt,
            shape=(H, W),
            alpha=alpha
        )
        ps = psnr(cover, stego)
        ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), Yw)
        return out_path, meta_path, ps, ss

    # ----- IMAGE watermark -----
    wm = cv2.imread(wm_source, cv2.IMREAD_COLOR)
    if wm is None:
        raise ValueError(f"Không mở được watermark: {wm_source}")
    hW, wW = wm.shape[:2]

    # Grayscale (embed trên kênh Y, giữ nguyên kích thước watermark)
    if not color:
        Y, YCrCb = _to_Y(cover)
        wm_gray = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY).astype(np.float32)

        C = dct2(Y); Uc, Sc, Vct = np.linalg.svd(C, full_matrices=False)
        Wm = dct2(wm_gray); Uw, Sw, Vwt = np.linalg.svd(Wm, full_matrices=False)
        L = min(len(Sc), len(Sw))
        S_ = Sc.copy(); S_[:L] = Sc[:L] + alpha * Sw[:L]
        Cw = (Uc @ np.diag(S_) @ Vct).astype(np.float32)
        Yw = idct2(Cw); stego = _from_Y(Yw, YCrCb)

        if not out_path.lower().endswith(".png"):
            out_path = os.path.splitext(out_path)[0] + "_stego.png"
        ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            raise IOError("Ghi stego thất bại.")
        np.savez_compressed(
            meta_path,
            mode='gray',
            payload_type='image',
            Sc=Sc, Uw=Uw, Vwt=Vwt,
            payload_shape=(hW, wW),
            alpha=alpha
        )
        ps = psnr(cover, stego)
        ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), Yw)
        return out_path, meta_path, ps, ss

    # Color (per-channel B,G,R), giữ nguyên kích thước watermark
    b, g, r = cv2.split(cover.astype(np.float32))
    wb, wg, wr = cv2.split(wm.astype(np.float32))
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    Ub, Sb, Vbt = np.linalg.svd(Cb, full_matrices=False)
    Ug, Sg, Vgt = np.linalg.svd(Cg, full_matrices=False)
    Ur, Sr, Vrt = np.linalg.svd(Cr, full_matrices=False)

    WB = dct2(wb); UWb, SWb, VWbt = np.linalg.svd(WB, full_matrices=False)
    WG = dct2(wg); UWg, SWg, VWgt = np.linalg.svd(WG, full_matrices=False)
    WR = dct2(wr); UWr, SWr, VWrt = np.linalg.svd(WR, full_matrices=False)

    Lb = min(len(Sb), len(SWb)); Lg = min(len(Sg), len(SWg)); Lr = min(len(Sr), len(SWr))
    Sb_, Sg_, Sr_ = Sb.copy(), Sg.copy(), Sr.copy()
    Sb_[:Lb] = Sb[:Lb] + alpha * SWb[:Lb]
    Sg_[:Lg] = Sg[:Lg] + alpha * SWg[:Lg]
    Sr_[:Lr] = Sr[:Lr] + alpha * SWr[:Lr]

    Cbw = (Ub @ np.diag(Sb_) @ Vbt).astype(np.float32)
    Cgw = (Ug @ np.diag(Sg_) @ Vgt).astype(np.float32)
    Crw = (Ur @ np.diag(Sr_) @ Vrt).astype(np.float32)

    bw, gw, rw = idct2(Cbw), idct2(Cgw), idct2(Crw)
    stego = cv2.merge([
        np.clip(bw, 0, 255).astype(np.uint8),
        np.clip(gw, 0, 255).astype(np.uint8),
        np.clip(rw, 0, 255).astype(np.uint8)
    ])

    if not out_path.lower().endswith(".png"):
        out_path = os.path.splitext(out_path)[0] + "_stego.png"
    ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError("Ghi stego thất bại.")
    np.savez_compressed(
        meta_path,
        mode='color',
        payload_type='image',
        Sb=Sb, Sg=Sg, Sr=Sr,
        UWb=UWb, VWbt=VWbt,
        UWg=UWg, VWgt=VWgt,
        UWr=UWr, VWrt=VWrt,
        payload_shape=(hW, wW),
        alpha=alpha
    )
    ps = psnr(cover, stego)
    ss = ssim(cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY), cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY))
    return out_path, meta_path, ps, ss


# ---------------------------- EXTRACT ----------------------------
def extract(stego_path: str, meta_path: str, out_path: str, normalize: bool = True) -> str:
    data = np.load(meta_path, allow_pickle=False)
    mode = str(data.get('mode', 'gray'))
    payload_type = str(data.get('payload_type', 'image'))
    alpha = float(data['alpha'])
    st = _read_image(stego_path)

    # ---------- TEXT / JSON (robust) ----------
    if payload_type in ('text', 'json'):
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        r = Uw.shape[1]
        L = min(len(Sc), len(S_cw), r, Vwt.shape[0])
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        Sw_full = np.zeros(r, np.float32); Sw_full[:L] = Sw_hat[:L]
        Wm_hat = (Uw[:, :r] @ np.diag(Sw_full) @ Vwt[:r, :]).astype(np.float32)

        # Binarize thông minh + fallback đảo bit
        binary = _robust_bits_from_idct(Wm_hat)
        by = _bitimg_to_bytes(binary)

        if payload_type == 'json':
            text = None
            try:
                obj = json.loads(by.decode('utf-8', errors='ignore'))
                text = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                inv_binary = (255 - binary).astype(np.uint8)
                by2 = _bitimg_to_bytes(inv_binary)
                try:
                    obj = json.loads(by2.decode('utf-8', errors='ignore'))
                    text = json.dumps(obj, ensure_ascii=False, indent=2)
                except Exception:
                    text = (by or by2).decode('utf-8', errors='ignore')
            if not out_path.lower().endswith('.json'):
                out_path = os.path.splitext(out_path)[0] + '_data.json'
        else:
            primary = by.decode('utf-8', errors='ignore')
            printable = sum(32 <= ord(c) <= 126 or ord(c) >= 160 for c in primary)
            if len(primary) == 0 or printable / max(1, len(primary)) < 0.6:
                inv_binary = (255 - binary).astype(np.uint8)
                by2 = _bitimg_to_bytes(inv_binary)
                alt = by2.decode('utf-8', errors='ignore')
                printable2 = sum(32 <= ord(c) <= 126 or ord(c) >= 160 for c in alt)
                if printable2 / max(1, len(alt)) > printable / max(1, len(primary)):
                    primary = alt
            text = primary
            if not out_path.lower().endswith('.txt'):
                out_path = os.path.splitext(out_path)[0] + '_text.txt'

        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text)
        return out_path

    # ---------- IMAGE watermark (grayscale on Y) ----------
    if mode == 'gray':
        Y, _ = _to_Y(st)
        Cw = dct2(Y); _, S_cw, _ = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        hW, wW = map(int, data['payload_shape'])
        r = Uw.shape[1]  # = min(hW, wW)
        L = min(len(Sc), len(S_cw), r, Vwt.shape[0])
        Sw_hat = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        Sw_full = np.zeros(r, np.float32); Sw_full[:L] = Sw_hat[:L]
        Wm_hat = (Uw[:, :r] @ np.diag(Sw_full) @ Vwt[:r, :]).astype(np.float32)  # (hW x wW)
        wm = idct2(Wm_hat)
        if normalize:
            wm = cv2.normalize(wm, None, 0, 255, cv2.NORM_MINMAX)
        wm = np.clip(wm, 0, 255).astype(np.uint8)
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '_wm.png'
        ok = cv2.imwrite(out_path, wm)
        if not ok:
            raise IOError('Ghi watermark thất bại.')
        return out_path

    # ---------- IMAGE watermark (color per-channel) ----------
    b, g, r = cv2.split(st.astype(np.float32))
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    _, S_cwb, _ = np.linalg.svd(Cb, full_matrices=False)
    _, S_cwg, _ = np.linalg.svd(Cg, full_matrices=False)
    _, S_cwr, _ = np.linalg.svd(Cr, full_matrices=False)

    UWb, VWbt = data['UWb'], data['VWbt']
    UWg, VWgt = data['UWg'], data['VWgt']
    UWr, VWrt = data['UWr'], data['VWrt']
    hW, wW = map(int, data['payload_shape'])

    rb = UWb.shape[1]; rg = UWg.shape[1]; rr = UWr.shape[1]
    Sb = data['Sb']; Sg = data['Sg']; Sr = data['Sr']

    Lb = min(len(Sb), len(S_cwb), rb, VWbt.shape[0])
    Lg = min(len(Sg), len(S_cwg), rg, VWgt.shape[0])
    Lr = min(len(Sr), len(S_cwr), rr, VWrt.shape[0])

    SWb_hat = (S_cwb[:Lb] - Sb[:Lb]) / max(alpha, 1e-8)
    SWg_hat = (S_cwg[:Lg] - Sg[:Lg]) / max(alpha, 1e-8)
    SWr_hat = (S_cwr[:Lr] - Sr[:Lr]) / max(alpha, 1e-8)

    SWb_full = np.zeros(rb, np.float32); SWb_full[:Lb] = SWb_hat[:Lb]
    SWg_full = np.zeros(rg, np.float32); SWg_full[:Lg] = SWg_hat[:Lg]
    SWr_full = np.zeros(rr, np.float32); SWr_full[:Lr] = SWr_hat[:Lr]

    WB_hat = (UWb[:, :rb] @ np.diag(SWb_full) @ VWbt[:rb, :]).astype(np.float32)
    WG_hat = (UWg[:, :rg] @ np.diag(SWg_full) @ VWgt[:rg, :]).astype(np.float32)
    WR_hat = (UWr[:, :rr] @ np.diag(SWr_full) @ VWrt[:rr, :]).astype(np.float32)

    wb = idct2(WB_hat); wg = idct2(WG_hat); wr = idct2(WR_hat)
    if normalize:
        wb = cv2.normalize(wb, None, 0, 255, cv2.NORM_MINMAX)
        wg = cv2.normalize(wg, None, 0, 255, cv2.NORM_MINMAX)
        wr = cv2.normalize(wr, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.merge([
        np.clip(wb, 0, 255).astype(np.uint8),
        np.clip(wg, 0, 255).astype(np.uint8),
        np.clip(wr, 0, 255).astype(np.uint8)
    ])
    if not out_path.lower().endswith('.png'):
        out_path = os.path.splitext(out_path)[0] + '_wm.png'
    ok = cv2.imwrite(out_path, out)
    if not ok:
        raise IOError('Ghi watermark thất bại.')
    return out_path


# ----------------------------- DETECT -----------------------------
def detect(stego_path: str, meta_path: str, thresh: float = 0.6) -> Tuple[bool, float]:
    """Trả về (có_watermark, score). Score càng cao càng chắc có watermark."""
    data = np.load(meta_path, allow_pickle=False)
    mode = str(data.get('mode', 'gray'))
    alpha = float(data['alpha'])
    st = _read_image(stego_path).astype(np.float32)

    def _score_partial(U, dS, Vt, L):
        if L <= 0:
            return 0.0
        U_L = U[:, :L]; Vt_L = Vt[:L, :]
        T = (U_L @ np.diag(dS[:L]) @ Vt_L).astype(np.float32)
        flat = T.ravel()
        if flat.size == 0 or np.std(flat) < 1e-6:
            return 0.0
        # Chuẩn hoá z-score và lấy mean (NC-like)
        return float(np.mean((flat - flat.mean()) / (flat.std() + 1e-6)))

    if mode == 'gray':
        Y, _ = _to_Y(st.astype(np.uint8))
        Cw = dct2(Y)
        Ucw, S_cw, Vcwt = np.linalg.svd(Cw, full_matrices=False)
        Sc = data['Sc']; Uw = data['Uw']; Vwt = data['Vwt']
        L = min(len(Sc), len(S_cw), Uw.shape[1], Vwt.shape[0])
        dS = (S_cw[:L] - Sc[:L]) / max(alpha, 1e-8)
        s = _score_partial(Uw, dS, Vwt, L)
        return bool(s >= thresh), float(s)

    # color
    b, g, r = cv2.split(st)
    Cb, Cg, Cr = dct2(b), dct2(g), dct2(r)
    Ucb, S_cwb, Vcbt = np.linalg.svd(Cb, full_matrices=False)
    Ucg, S_cwg, Vcgt = np.linalg.svd(Cg, full_matrices=False)
    Ucr, S_cwr, Vcrt = np.linalg.svd(Cr, full_matrices=False)

    Sb, Sg, Sr = data['Sb'], data['Sg'], data['Sr']
    UWb, VWbt = data['UWb'], data['VWbt']
    UWg, VWgt = data['UWg'], data['VWgt']
    UWr, VWrt = data['UWr'], data['VWrt']

    Lb = min(len(Sb), len(S_cwb), UWb.shape[1], VWbt.shape[0])
    Lg = min(len(Sg), len(S_cwg), UWg.shape[1], VWgt.shape[0])
    Lr = min(len(Sr), len(S_cwr), UWr.shape[1], VWrt.shape[0])

    sB = _score_partial(UWb, (S_cwb[:Lb] - Sb[:Lb]) / max(alpha, 1e-8), VWbt, Lb)
    sG = _score_partial(UWg, (S_cwg[:Lg] - Sg[:Lg]) / max(alpha, 1e-8), VWgt, Lg)
    sR = _score_partial(UWr, (S_cwr[:Lr] - Sr[:Lr]) / max(alpha, 1e-8), VWrt, Lr)
    s = (sB + sG + sR) / 3.0
    return bool(s >= thresh), float(s)
