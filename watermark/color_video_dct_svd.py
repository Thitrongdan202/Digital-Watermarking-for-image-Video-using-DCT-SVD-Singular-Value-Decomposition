import cv2
import numpy as np
from pathlib import Path
from .dct_svd import _dct2, _idct2, create_text_watermark
from numpy.linalg import svd
from PIL import Image
import tempfile
import os


def embed_watermark_video_color(
    host_video_path: str,
    watermark_path: str,
    output_video_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    frame_interval: int = 10
) -> None:
    """
    Embed watermark into video frames using DCT-SVD with color preservation.
    """
    # Load watermark image
    watermark = Image.open(watermark_path).convert("L")
    
    # Open video
    cap = cv2.VideoCapture(host_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {host_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize watermark to video dimensions
    watermark = watermark.resize((width, height))
    wm_arr = np.asarray(watermark, dtype=np.float64)
    wm_dct = _dct2(wm_arr)
    Uw, Sw, Vtw = svd(wm_dct, full_matrices=False)
    
    # Setup video writer - keep color
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
    
    # Store metadata for extraction
    metadata = {
        'watermark_frames': [],
        'original_singular_values': [],
        'Uw': Uw,
        'Sw': Sw, 
        'Vtw': Vtw,
        'alpha': alpha,
        'frame_interval': frame_interval,
        'watermark_shape': wm_arr.shape,
        'is_color': True
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Embed watermark in selected frames while preserving color
        if frame_count % frame_interval == 0:
            # Process each color channel separately
            frame_bgr = frame.astype(np.float64)
            watermarked_channels = []
            frame_metadata = {'B': {}, 'G': {}, 'R': {}}
            
            for i, channel in enumerate(['B', 'G', 'R']):  # OpenCV uses BGR
                # DCT transform
                channel_dct = _dct2(frame_bgr[:, :, i])
                
                # SVD decomposition  
                U, S, Vt = svd(channel_dct, full_matrices=False)
                
                # Store original singular values for this channel
                frame_metadata[channel] = S.copy()
                
                # Embed watermark
                S_marked = S + alpha * Sw
                watermarked_dct = (U * S_marked) @ Vt
                
                # Inverse DCT
                watermarked_channel = _idct2(watermarked_dct)
                watermarked_channels.append(watermarked_channel)
            
            # Combine channels back to BGR
            watermarked_frame = np.stack(watermarked_channels, axis=2)
            watermarked_frame = np.clip(watermarked_frame, 0, 255).astype(np.uint8)
            
            # Store metadata for this frame
            metadata['watermark_frames'].append(frame_count)
            metadata['original_singular_values'].append(frame_metadata)
            
            out.write(watermarked_frame)
        else:
            # Write original frame
            out.write(frame)
            
        frame_count += 1
        
        # Progress callback
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save metadata
    np.savez(metadata_path, **metadata)


def embed_text_watermark_video_color(
    host_video_path: str,
    text: str,
    output_video_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    font_size: int = 40,
    frame_interval: int = 10
) -> None:
    """
    Embed text watermark into video frames using DCT-SVD with color preservation.
    """
    # Open video
    cap = cv2.VideoCapture(host_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {host_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create text watermark
    wm_arr = create_text_watermark(text, (width, height), font_size)
    wm_dct = _dct2(wm_arr)
    Uw, Sw, Vtw = svd(wm_dct, full_matrices=False)
    
    # Setup video writer - keep color
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
    
    # Store metadata for extraction
    metadata = {
        'watermark_frames': [],
        'original_singular_values': [],
        'Uw': Uw,
        'Sw': Sw, 
        'Vtw': Vtw,
        'alpha': alpha,
        'frame_interval': frame_interval,
        'watermark_shape': wm_arr.shape,
        'text': text,
        'font_size': font_size,
        'is_text_watermark': True,
        'is_color': True
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Embed watermark in selected frames while preserving color
        if frame_count % frame_interval == 0:
            # Process each color channel separately
            frame_bgr = frame.astype(np.float64)
            watermarked_channels = []
            frame_metadata = {'B': {}, 'G': {}, 'R': {}}
            
            for i, channel in enumerate(['B', 'G', 'R']):  # OpenCV uses BGR
                # DCT transform
                channel_dct = _dct2(frame_bgr[:, :, i])
                
                # SVD decomposition  
                U, S, Vt = svd(channel_dct, full_matrices=False)
                
                # Store original singular values for this channel
                frame_metadata[channel] = S.copy()
                
                # Embed watermark
                S_marked = S + alpha * Sw
                watermarked_dct = (U * S_marked) @ Vt
                
                # Inverse DCT
                watermarked_channel = _idct2(watermarked_dct)
                watermarked_channels.append(watermarked_channel)
            
            # Combine channels back to BGR
            watermarked_frame = np.stack(watermarked_channels, axis=2)
            watermarked_frame = np.clip(watermarked_frame, 0, 255).astype(np.uint8)
            
            # Store metadata for this frame
            metadata['watermark_frames'].append(frame_count)
            metadata['original_singular_values'].append(frame_metadata)
            
            out.write(watermarked_frame)
        else:
            # Write original frame
            out.write(frame)
            
        frame_count += 1
        
        # Progress callback
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save metadata
    np.savez(metadata_path, **metadata)


def extract_watermark_video_color(
    watermarked_video_path: str,
    metadata_path: str, 
    output_path: str
) -> str:
    """
    Extract watermark from color watermarked video using metadata.
    """
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    # Load metadata
    meta = np.load(metadata_path, allow_pickle=True)
    watermark_frames = meta['watermark_frames']
    original_s_values = meta['original_singular_values'] 
    Uw = meta['Uw']
    Vtw = meta['Vtw']
    alpha = float(meta['alpha'])
    shape = tuple(meta['watermark_shape'])
    
    # Get original text if this is a text watermark
    is_text = meta.get('is_text_watermark', False)
    original_text = meta.get('text', None) if is_text else None
    
    # Open video
    cap = cv2.VideoCapture(watermarked_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {watermarked_video_path}")
        
    extracted_watermarks = []
    
    for i, frame_idx in enumerate(watermark_frames):
        # Seek to watermarked frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Process each color channel
            frame_bgr = frame.astype(np.float64)
            extracted_channels = []
            frame_metadata = original_s_values[i]
            
            for j, channel in enumerate(['B', 'G', 'R']):
                # DCT transform
                channel_dct = _dct2(frame_bgr[:, :, j])
                
                # SVD decomposition
                _, S_wm, _ = svd(channel_dct, full_matrices=False)
                
                # Extract watermark singular values
                S_orig = frame_metadata[channel]
                Sw_est = (S_wm - S_orig) / alpha
                
                # Reconstruct watermark
                wm_dct_est = (Uw * Sw_est) @ Vtw
                wm_est = _idct2(wm_dct_est)[:shape[0], :shape[1]]
                
                extracted_channels.append(wm_est)
            
            # Average across color channels for final watermark
            avg_channel = np.mean(extracted_channels, axis=0)
            extracted_watermarks.append(avg_channel)
    
    cap.release()
    
    # Average extracted watermarks for better quality
    if extracted_watermarks:
        avg_watermark = np.mean(extracted_watermarks, axis=0)
        avg_watermark = np.clip(avg_watermark, 0, 255).astype(np.uint8)
        
        # Save extracted watermark
        Image.fromarray(avg_watermark, 'L').save(output_path)
        
        return original_text
    else:
        raise ValueError("No watermarked frames found")