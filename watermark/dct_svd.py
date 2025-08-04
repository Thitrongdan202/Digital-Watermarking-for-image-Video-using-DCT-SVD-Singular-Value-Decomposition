import numpy as np
from scipy.fftpack import dct, idct
from numpy.linalg import svd
from PIL import Image


def _dct2(a: np.ndarray) -> np.ndarray:
    """Compute 2D DCT (type-II) with orthogonal normalization."""
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def _idct2(a: np.ndarray) -> np.ndarray:
    """Compute 2D inverse DCT (type-II) with orthogonal normalization."""
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def embed_watermark(host_path: str, watermark_path: str, output_image_path: str,
                     metadata_path: str, alpha: float = 0.1) -> None:
    """Embed watermark into host image using DCT-SVD and save metadata."""
    host = Image.open(host_path).convert('L')
    watermark = Image.open(watermark_path).convert('L').resize(host.size)

    host_arr = np.array(host, dtype=np.float64)
    wm_arr = np.array(watermark, dtype=np.float64)

    host_dct = _dct2(host_arr)
    wm_dct = _dct2(wm_arr)

    U, S, Vt = svd(host_dct)
    Uw, Sw, Vtw = svd(wm_dct)

    S_marked = S + alpha * Sw

    watermarked_dct = U @ np.diag(S_marked) @ Vt
    watermarked = _idct2(watermarked_dct)

    Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8)).save(output_image_path)

    np.savez(metadata_path, S=S, Uw=Uw, Vtw=Vtw, alpha=alpha, shape=wm_arr.shape)


def extract_watermark(watermarked_path: str, metadata_path: str, output_path: str) -> None:
    """Extract watermark from watermarked image using saved metadata."""
    meta = np.load(metadata_path, allow_pickle=True)
    S_orig = meta['S']
    Uw = meta['Uw']
    Vtw = meta['Vtw']
    alpha = float(meta['alpha'])
    shape = tuple(int(x) for x in meta['shape'])

    watermarked = Image.open(watermarked_path).convert('L')
    watermarked_arr = np.array(watermarked, dtype=np.float64)
    watermarked_dct = _dct2(watermarked_arr)

    _, S_wm, _ = svd(watermarked_dct)
    Sw_est = (S_wm - S_orig) / alpha

    wm_dct_est = Uw @ np.diag(Sw_est) @ Vtw
    wm_est = _idct2(wm_dct_est)
    wm_est = wm_est[:shape[0], :shape[1]]

    Image.fromarray(np.clip(wm_est, 0, 255).astype(np.uint8)).save(output_path)


def singular_values(image_path: str) -> np.ndarray:
    """Utility to compute singular values of an image's DCT representation."""
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    dct_arr = _dct2(arr)
    _, s, _ = svd(dct_arr)
    return s
