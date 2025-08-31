# dct_svd.py  –  FULL REWRITE
import numpy as np
from scipy.fftpack import dct, idct
from numpy.linalg import svd
from PIL import Image
from pathlib import Path


# ------------------------------------------------------------------
# Helper: 2-D (I)DCT
# ------------------------------------------------------------------
def _dct2(a: np.ndarray) -> np.ndarray:
    """2-D orthonormal DCT (type-II)."""
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def _idct2(a: np.ndarray) -> np.ndarray:
    """2-D inverse orthonormal DCT (type-II)."""
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def embed_watermark(
    host_path: str,
    watermark_path: str,
    output_image_path: str,
    metadata_path: str,
    alpha: float = 0.05,
) -> None:
    """
    Embed watermark into host image using DCT-SVD.

    Parameters
    ----------
    host_path : str
        Grayscale/RGB image containing the content.
    watermark_path : str
        Image used as watermark (resized to host dimensions).
    output_image_path : str
        Where to save the watermarked image.
    metadata_path : str
        Path to .npz file storing singular values & matrices for extraction.
    alpha : float, optional
        Embedding strength; 0.02–0.10 thường cho PSNR 38–44 dB.
    """
    # 1) Đọc và chuyển ảnh sang grayscale double
    host = Image.open(host_path).convert("L")
    watermark = Image.open(watermark_path).convert("L").resize(host.size)

    host_arr = np.asarray(host, dtype=np.float64)
    wm_arr = np.asarray(watermark, dtype=np.float64)

    # 2) DCT
    host_dct = _dct2(host_arr)
    wm_dct = _dct2(wm_arr)

    # 3) SVD rút gọn (k = min(m, n))
    U, S, Vt = svd(host_dct, full_matrices=False)
    Uw, Sw, Vtw = svd(wm_dct, full_matrices=False)

    # 4) Nhúng watermark vào singular values
    S_marked = S + alpha * Sw
    watermarked_dct = (U * S_marked) @ Vt  # broadcast nhân nhanh

    # 5) IDCT & lưu ảnh
    watermarked = _idct2(watermarked_dct)
    Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8)).save(output_image_path)

    # 6) Lưu metadata để trích
    np.savez(
        metadata_path,
        S=S,
        Uw=Uw,
        Vtw=Vtw,
        alpha=alpha,
        shape=wm_arr.shape,
    )


def extract_watermark(
    watermarked_path: str,
    metadata_path: str,
    output_path: str,
) -> None:
    """
    Reconstruct watermark from watermarked image using saved metadata.

    Parameters
    ----------
    watermarked_path : str
        Path to image suspected of containing watermark.
    metadata_path : str
        .npz file created during embedding.
    output_path : str
        Where to save the extracted watermark.
    """
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta = np.load(metadata_path, allow_pickle=True)
    S_orig = meta["S"]
    Uw = meta["Uw"]
    Vtw = meta["Vtw"]
    alpha = float(meta["alpha"])
    shape = tuple(int(x) for x in meta["shape"])

    watermarked = Image.open(watermarked_path).convert("L")
    watermarked_arr = np.asarray(watermarked, dtype=np.float64)
    wm_dct = _dct2(watermarked_arr)

    # SVD rút gọn
    _, S_wm, _ = svd(wm_dct, full_matrices=False)
    Sw_est = (S_wm - S_orig) / alpha

    wm_dct_est = (Uw * Sw_est) @ Vtw
    wm_est = _idct2(wm_dct_est)[: shape[0], : shape[1]]

    Image.fromarray(np.clip(wm_est, 0, 255).astype(np.uint8)).save(output_path)


def singular_values(image_path: str) -> np.ndarray:
    """Compute singular values of an image’s DCT – tiện cho AI detector."""
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    _, s, _ = svd(_dct2(arr), full_matrices=False)
    return s
