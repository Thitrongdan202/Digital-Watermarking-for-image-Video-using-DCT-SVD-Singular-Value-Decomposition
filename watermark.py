import cv2
import numpy as np

def embed_watermark(host_path: str, watermark_path: str, alpha: float = 0.1, output_path: str = 'watermarked.png') -> None:
    """Embed a watermark into a host image using DCT and SVD.

    The function resizes the watermark to match the host image so that
    matrix multiplications have compatible shapes.

    Parameters
    ----------
    host_path: str
        Path to the original image that will receive the watermark.
    watermark_path: str
        Path to the image that will be used as watermark.
    alpha: float, optional
        Strength of the watermark. Higher values make the watermark more
        visible. Default is 0.1.
    output_path: str, optional
        Where to save the watermarked image. Default is ``watermarked.png``.
    """
    host = cv2.imread(host_path, cv2.IMREAD_GRAYSCALE)
    if host is None:
        raise FileNotFoundError(f"Host image not found: {host_path}")
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"Watermark image not found: {watermark_path}")

    h, w = host.shape
    # Resize watermark to match the host size to avoid shape mismatch in SVD
    watermark = cv2.resize(watermark, (w, h))

    host_dct = cv2.dct(host.astype(np.float32))
    watermark_dct = cv2.dct(watermark.astype(np.float32))

    U_h, S_h, Vt_h = np.linalg.svd(host_dct, full_matrices=False)
    U_w, S_w, Vt_w = np.linalg.svd(watermark_dct, full_matrices=False)

    # Embed watermark by modifying singular values
    S_embedded = S_h + alpha * S_w
    watermarked_dct = U_h @ np.diag(S_embedded) @ Vt_h
    watermarked = cv2.idct(watermarked_dct)

    cv2.imwrite(output_path, np.clip(watermarked, 0, 255).astype(np.uint8))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embed watermark into an image using DCT-SVD")
    parser.add_argument('host', help='Path to host image')
    parser.add_argument('watermark', help='Path to watermark image')
    parser.add_argument('-o', '--output', default='watermarked.png', help='Output watermarked image path')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Watermark strength')
    args = parser.parse_args()
    embed_watermark(args.host, args.watermark, alpha=args.alpha, output_path=args.output)
