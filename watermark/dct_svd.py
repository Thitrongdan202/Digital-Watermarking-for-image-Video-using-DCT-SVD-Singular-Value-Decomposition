# dct_svd.py  –  FULL REWRITE
import numpy as np
from scipy.fftpack import dct, idct
from numpy.linalg import svd
from PIL import Image, ImageDraw, ImageFont
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
def _embed_single_channel(host_channel, wm_channel, alpha):
    """Embed watermark into a single color channel"""
    # DCT transform
    host_dct = _dct2(host_channel)
    wm_dct = _dct2(wm_channel)
    
    # SVD decomposition
    U, S, Vt = svd(host_dct, full_matrices=False)
    Uw, Sw, Vtw = svd(wm_dct, full_matrices=False)
    
    # Embed watermark
    S_marked = S + alpha * Sw
    watermarked_dct = (U * S_marked) @ Vt
    
    # Inverse DCT
    watermarked_channel = _idct2(watermarked_dct)
    
    return watermarked_channel, S, Uw, Vtw


def embed_watermark(
    host_path: str,
    watermark_path: str,
    output_image_path: str,
    metadata_path: str,
    alpha: float = 0.05,
) -> None:
    """
    Embed watermark into host image using DCT-SVD with color preservation.

    Parameters
    ----------
    host_path : str
        RGB/Grayscale image containing the content.
    watermark_path : str
        Image used as watermark (resized to host dimensions).
    output_image_path : str
        Where to save the watermarked image.
    metadata_path : str
        Path to .npz file storing singular values & matrices for extraction.
    alpha : float, optional
        Embedding strength; 0.02–0.10 for good quality.
    """
    # Load images
    host = Image.open(host_path)
    watermark = Image.open(watermark_path).resize(host.size)
    
    # Detect if image is color or grayscale
    is_color = host.mode in ('RGB', 'RGBA')
    
    if is_color:
        # Convert to RGB if needed
        host = host.convert('RGB')
        watermark = watermark.convert('RGB')
        
        host_arr = np.asarray(host, dtype=np.float64)
        wm_arr = np.asarray(watermark, dtype=np.float64)
        
        # Process each color channel separately
        watermarked_channels = []
        metadata_channels = {'R': {}, 'G': {}, 'B': {}}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            watermarked_channel, S, Uw, Vtw = _embed_single_channel(
                host_arr[:, :, i], wm_arr[:, :, i], alpha
            )
            watermarked_channels.append(watermarked_channel)
            
            # Store metadata for each channel
            metadata_channels[channel] = {
                'S': S,
                'Uw': Uw,
                'Vtw': Vtw
            }
        
        # Combine channels back to RGB
        watermarked = np.stack(watermarked_channels, axis=2)
        watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
        
        # Save as RGB image
        Image.fromarray(watermarked, 'RGB').save(output_image_path)
        
        # Save metadata
        np.savez(
            metadata_path,
            alpha=alpha,
            shape=wm_arr.shape,
            is_color=True,
            **{f'{ch}_{key}': val for ch, data in metadata_channels.items() 
               for key, val in data.items()}
        )
        
    else:
        # Grayscale processing (original method)
        host = host.convert("L")
        watermark = watermark.convert("L")
        
        host_arr = np.asarray(host, dtype=np.float64)
        wm_arr = np.asarray(watermark, dtype=np.float64)
        
        watermarked, S, Uw, Vtw = _embed_single_channel(host_arr, wm_arr, alpha)
        
        # Save grayscale image
        Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8), 'L').save(output_image_path)
        
        # Save metadata
        np.savez(
            metadata_path,
            S=S,
            Uw=Uw,
            Vtw=Vtw,
            alpha=alpha,
            shape=wm_arr.shape,
            is_color=False
        )


def _extract_single_channel(watermarked_channel, S_orig, Uw, Vtw, alpha):
    """Extract watermark from a single color channel"""
    # DCT transform
    wm_dct = _dct2(watermarked_channel)
    
    # SVD decomposition
    _, S_wm, _ = svd(wm_dct, full_matrices=False)
    
    # Extract watermark singular values
    Sw_est = (S_wm - S_orig) / alpha
    
    # Reconstruct watermark
    wm_dct_est = (Uw * Sw_est) @ Vtw
    wm_est = _idct2(wm_dct_est)
    
    return wm_est


def extract_watermark(
    watermarked_path: str,
    metadata_path: str,
    output_path: str,
) -> None:
    """
    Reconstruct watermark from watermarked image using saved metadata with color support.

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
    alpha = float(meta["alpha"])
    shape = tuple(int(x) for x in meta["shape"])
    is_color = meta.get("is_color", False)
    
    watermarked = Image.open(watermarked_path)
    
    if is_color and watermarked.mode in ('RGB', 'RGBA'):
        # Color image extraction
        watermarked = watermarked.convert('RGB')
        watermarked_arr = np.asarray(watermarked, dtype=np.float64)
        
        # Extract from each color channel
        extracted_channels = []
        
        for i, channel in enumerate(['R', 'G', 'B']):
            S_orig = meta[f"{channel}_S"]
            Uw = meta[f"{channel}_Uw"]
            Vtw = meta[f"{channel}_Vtw"]
            
            wm_est = _extract_single_channel(
                watermarked_arr[:, :, i], S_orig, Uw, Vtw, alpha
            )
            extracted_channels.append(wm_est[:shape[0], :shape[1]])
        
        # Combine channels back to RGB
        wm_combined = np.stack(extracted_channels, axis=2)
        wm_combined = np.clip(wm_combined, 0, 255).astype(np.uint8)
        
        # Save as RGB image
        Image.fromarray(wm_combined, 'RGB').save(output_path)
        
    else:
        # Grayscale extraction (backward compatibility)
        S_orig = meta["S"]
        Uw = meta["Uw"]
        Vtw = meta["Vtw"]
        
        watermarked = watermarked.convert("L")
        watermarked_arr = np.asarray(watermarked, dtype=np.float64)
        
        wm_est = _extract_single_channel(watermarked_arr, S_orig, Uw, Vtw, alpha)
        wm_est = wm_est[:shape[0], :shape[1]]
        
        # Save as grayscale image
        Image.fromarray(np.clip(wm_est, 0, 255).astype(np.uint8), 'L').save(output_path)


def create_text_watermark(text: str, size: tuple, font_size: int = 40) -> np.ndarray:
    """
    Create a watermark image from text.
    
    Parameters
    ----------
    text : str
        Text to convert to watermark
    size : tuple
        (width, height) of the output watermark
    font_size : int
        Size of the font
        
    Returns
    -------
    np.ndarray
        Watermark as grayscale array
    """
    # Create blank image
    img = Image.new('L', size, color=0)  # Black background
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font, fallback to default
    try:
        # Try common system fonts
        font_paths = [
            '/System/Library/Fonts/Arial.ttf',  # macOS
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:/Windows/Fonts/arial.ttf',  # Windows
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Linux
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
                
        if font is None:
            font = ImageFont.load_default()
            
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position to center text
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text in white
    draw.text((x, y), text, fill=255, font=font)
    
    return np.asarray(img, dtype=np.float64)


def embed_text_watermark(
    host_path: str,
    text: str,
    output_image_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    font_size: int = 40,
) -> None:
    """
    Embed text watermark into host image using DCT-SVD with color preservation.
    
    Parameters
    ----------
    host_path : str
        Path to host image
    text : str
        Text to embed as watermark
    output_image_path : str
        Where to save watermarked image
    metadata_path : str
        Path to save metadata for extraction
    alpha : float
        Embedding strength
    font_size : int
        Size of text font
    """
    # Load host image
    host = Image.open(host_path)
    is_color = host.mode in ('RGB', 'RGBA')
    
    if is_color:
        # Color processing
        host = host.convert('RGB')
        host_arr = np.asarray(host, dtype=np.float64)
        
        # Create text watermark (grayscale) 
        wm_arr = create_text_watermark(text, host.size, font_size)
        
        # Process each color channel with the same text watermark
        watermarked_channels = []
        metadata_channels = {'R': {}, 'G': {}, 'B': {}}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            watermarked_channel, S, Uw, Vtw = _embed_single_channel(
                host_arr[:, :, i], wm_arr, alpha
            )
            watermarked_channels.append(watermarked_channel)
            
            # Store metadata for each channel
            metadata_channels[channel] = {
                'S': S,
                'Uw': Uw,
                'Vtw': Vtw
            }
        
        # Combine channels back to RGB
        watermarked = np.stack(watermarked_channels, axis=2)
        watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
        
        # Save as RGB image
        Image.fromarray(watermarked, 'RGB').save(output_image_path)
        
        # Save metadata
        np.savez(
            metadata_path,
            alpha=alpha,
            shape=wm_arr.shape,
            text=text,
            font_size=font_size,
            is_text_watermark=True,
            is_color=True,
            **{f'{ch}_{key}': val for ch, data in metadata_channels.items() 
               for key, val in data.items()}
        )
        
    else:
        # Grayscale processing
        host = host.convert("L")
        host_arr = np.asarray(host, dtype=np.float64)
        
        # Create text watermark
        wm_arr = create_text_watermark(text, host.size, font_size)
        
        watermarked, S, Uw, Vtw = _embed_single_channel(host_arr, wm_arr, alpha)
        
        # Save grayscale image
        Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8), 'L').save(output_image_path)
        
        # Save metadata
        np.savez(
            metadata_path,
            S=S,
            Uw=Uw,
            Vtw=Vtw,
            alpha=alpha,
            shape=wm_arr.shape,
            text=text,
            font_size=font_size,
            is_text_watermark=True,
            is_color=False
        )


def extract_text_watermark(
    watermarked_path: str,
    metadata_path: str,
    output_path: str,
) -> str:
    """
    Extract text watermark from watermarked image with color support.
    
    Parameters
    ----------
    watermarked_path : str
        Path to watermarked image
    metadata_path : str
        Path to metadata file
    output_path : str
        Path to save extracted watermark image
        
    Returns
    -------
    str
        Original text if stored in metadata, otherwise None
    """
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    meta = np.load(metadata_path, allow_pickle=True)
    alpha = float(meta["alpha"])
    shape = tuple(int(x) for x in meta["shape"])
    
    # Check if this is a text watermark and color
    is_text = meta.get("is_text_watermark", False)
    is_color = meta.get("is_color", False)
    original_text = meta.get("text", None) if is_text else None

    # Load watermarked image
    watermarked = Image.open(watermarked_path)
    
    if is_color and watermarked.mode in ('RGB', 'RGBA'):
        # Color extraction - same as extract_watermark function
        watermarked = watermarked.convert('RGB')
        watermarked_arr = np.asarray(watermarked, dtype=np.float64)
        
        # Extract from each color channel
        extracted_channels = []
        
        for i, channel in enumerate(['R', 'G', 'B']):
            S_orig = meta[f"{channel}_S"]
            Uw = meta[f"{channel}_Uw"]
            Vtw = meta[f"{channel}_Vtw"]
            
            wm_est = _extract_single_channel(
                watermarked_arr[:, :, i], S_orig, Uw, Vtw, alpha
            )
            extracted_channels.append(wm_est[:shape[0], :shape[1]])
        
        # For text watermarks, average the channels to get cleaner text
        wm_combined = np.mean(extracted_channels, axis=0)
        wm_combined = np.clip(wm_combined, 0, 255).astype(np.uint8)
        
        # Save as grayscale since text watermarks are grayscale
        Image.fromarray(wm_combined, 'L').save(output_path)
        
    else:
        # Grayscale extraction (backward compatibility)
        # Check if we have old format metadata
        if "S" in meta:
            # Old format
            S_orig = meta["S"]
            Uw = meta["Uw"]  
            Vtw = meta["Vtw"]
        else:
            # New format but grayscale
            S_orig = meta.get("R_S", meta.get("B_S", meta.get("G_S", None)))
            Uw = meta.get("R_Uw", meta.get("B_Uw", meta.get("G_Uw", None)))
            Vtw = meta.get("R_Vtw", meta.get("B_Vtw", meta.get("G_Vtw", None)))
            
            if S_orig is None or Uw is None or Vtw is None:
                raise KeyError("Could not find metadata in expected format")
        
        watermarked = watermarked.convert("L")
        watermarked_arr = np.asarray(watermarked, dtype=np.float64)
        
        wm_est = _extract_single_channel(watermarked_arr, S_orig, Uw, Vtw, alpha)
        wm_est = wm_est[:shape[0], :shape[1]]
        
        # Save as grayscale image
        Image.fromarray(np.clip(wm_est, 0, 255).astype(np.uint8), 'L').save(output_path)
    
    return original_text


def singular_values(image_path: str) -> np.ndarray:
    """Compute singular values of an image's DCT – tiện cho AI detector."""
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    _, s, _ = svd(_dct2(arr), full_matrices=False)
    return s
