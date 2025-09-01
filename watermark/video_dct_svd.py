import cv2
import numpy as np
from pathlib import Path
from .dct_svd import _dct2, _idct2, create_text_watermark
from numpy.linalg import svd
from PIL import Image
import tempfile
import os
import subprocess


def _preserve_audio_with_ffmpeg(video_only_path: str, original_video_path: str, final_output_path: str) -> bool:
    """
    Use ffmpeg to combine watermarked video with original audio.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("FFmpeg not found - audio will not be preserved")
            return False
            
        # Combine video and audio using ffmpeg
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', video_only_path,  # Input watermarked video (no audio)
            '-i', original_video_path,  # Input original video (for audio)
            '-c:v', 'copy',  # Copy video stream as-is
            '-c:a', 'aac',   # Re-encode audio to AAC
            '-map', '0:v:0', # Take video from first input
            '-map', '1:a:0', # Take audio from second input
            '-shortest',     # End when shortest stream ends
            final_output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("Audio successfully preserved using ffmpeg")
            return True
        else:
            print(f"FFmpeg failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FFmpeg timeout - audio preservation failed")
        return False
    except FileNotFoundError:
        print("FFmpeg not found - audio will not be preserved")
        return False
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def embed_watermark_video(
    host_video_path: str,
    watermark_path: str,
    output_video_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    frame_interval: int = 10
) -> None:
    """
    Embed watermark into video frames using DCT-SVD.
    
    Parameters
    ----------
    host_video_path : str
        Path to host video file
    watermark_path : str  
        Path to watermark image
    output_video_path : str
        Path for output watermarked video
    metadata_path : str
        Path to save metadata for extraction
    alpha : float
        Embedding strength
    frame_interval : int
        Embed watermark every N frames (default: every 10 frames)
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
    
    # Setup video writer
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
        'watermark_shape': wm_arr.shape
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale for processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Embed watermark in selected frames
        if frame_count % frame_interval == 0:
            # DCT transform
            frame_dct = _dct2(gray_frame)
            
            # SVD decomposition  
            U, S, Vt = svd(frame_dct, full_matrices=False)
            
            # Store original singular values
            metadata['watermark_frames'].append(frame_count)
            metadata['original_singular_values'].append(S)
            
            # Embed watermark
            S_marked = S + alpha * Sw
            watermarked_dct = (U * S_marked) @ Vt
            
            # Inverse DCT
            watermarked_frame = _idct2(watermarked_dct)
            watermarked_frame = np.clip(watermarked_frame, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for video
            watermarked_bgr = cv2.cvtColor(watermarked_frame, cv2.COLOR_GRAY2BGR)
            out.write(watermarked_bgr)
        else:
            # Write original frame
            out.write(frame)
            
        frame_count += 1
        
        # Progress callback could be added here
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save metadata
    np.savez(metadata_path, **metadata)


def extract_watermark_video(
    watermarked_video_path: str,
    metadata_path: str, 
    output_path: str
) -> None:
    """
    Extract watermark from watermarked video using metadata.
    
    Parameters
    ----------
    watermarked_video_path : str
        Path to watermarked video
    metadata_path : str
        Path to metadata file from embedding
    output_path : str  
        Path to save extracted watermark image
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
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            # DCT transform
            frame_dct = _dct2(gray_frame)
            
            # SVD decomposition
            _, S_wm, _ = svd(frame_dct, full_matrices=False)
            
            # Extract watermark singular values
            S_orig = original_s_values[i]
            Sw_est = (S_wm - S_orig) / alpha
            
            # Reconstruct watermark
            wm_dct_est = (Uw * Sw_est) @ Vtw
            wm_est = _idct2(wm_dct_est)[:shape[0], :shape[1]]
            
            extracted_watermarks.append(wm_est)
    
    cap.release()
    
    # Average extracted watermarks for better quality
    if extracted_watermarks:
        avg_watermark = np.mean(extracted_watermarks, axis=0)
        avg_watermark = np.clip(avg_watermark, 0, 255).astype(np.uint8)
        
        # Save extracted watermark
        Image.fromarray(avg_watermark).save(output_path)
    else:
        raise ValueError("No watermarked frames found")


def detect_watermark_video(
    video_path: str,
    frame_sample_rate: int = 30
) -> dict:
    """
    Detect potential watermark presence in video frames.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    frame_sample_rate : int
        Analyze every Nth frame
        
    Returns
    -------
    dict
        Detection results with frame analysis
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    singular_value_stats = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_sample_rate == 0:
            # Convert to grayscale and analyze singular values
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            frame_dct = _dct2(gray_frame)
            _, s, _ = svd(frame_dct, full_matrices=False)
            
            # Store statistics for analysis
            singular_value_stats.append({
                'frame': frame_count,
                'sv_mean': np.mean(s),
                'sv_std': np.std(s),
                'sv_max': np.max(s),
                'sv_entropy': -np.sum(s * np.log(s + 1e-10))
            })
            
        frame_count += 1
        
    cap.release()
    
    # Analyze patterns in singular values that might indicate watermarking
    if singular_value_stats:
        sv_means = [stat['sv_mean'] for stat in singular_value_stats]
        sv_stds = [stat['sv_std'] for stat in singular_value_stats]
        
        # Simple heuristic: look for consistency that might indicate watermarking
        mean_consistency = np.std(sv_means)
        std_consistency = np.std(sv_stds)
        
        # Lower values might indicate more consistent processing (watermarking)
        watermark_likelihood = 1.0 / (1.0 + mean_consistency + std_consistency)
        
        return {
            'total_frames_analyzed': len(singular_value_stats),
            'watermark_likelihood': watermark_likelihood,
            'frame_statistics': singular_value_stats,
            'mean_consistency': mean_consistency,
            'std_consistency': std_consistency
        }
    else:
        return {'error': 'No frames could be analyzed'}


def embed_text_watermark_video(
    host_video_path: str,
    text: str,
    output_video_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    font_size: int = 40,
    frame_interval: int = 10
) -> None:
    """
    Embed text watermark into video frames using DCT-SVD.
    
    Parameters
    ----------
    host_video_path : str
        Path to host video file
    text : str
        Text to embed as watermark
    output_video_path : str
        Path for output watermarked video
    metadata_path : str
        Path to save metadata for extraction
    alpha : float
        Embedding strength
    font_size : int
        Size of text font
    frame_interval : int
        Embed watermark every N frames
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
    
    # Setup video writer
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
        'is_text_watermark': True
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale for processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Embed watermark in selected frames
        if frame_count % frame_interval == 0:
            # DCT transform
            frame_dct = _dct2(gray_frame)
            
            # SVD decomposition  
            U, S, Vt = svd(frame_dct, full_matrices=False)
            
            # Store original singular values
            metadata['watermark_frames'].append(frame_count)
            metadata['original_singular_values'].append(S)
            
            # Embed watermark
            S_marked = S + alpha * Sw
            watermarked_dct = (U * S_marked) @ Vt
            
            # Inverse DCT
            watermarked_frame = _idct2(watermarked_dct)
            watermarked_frame = np.clip(watermarked_frame, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for video
            watermarked_bgr = cv2.cvtColor(watermarked_frame, cv2.COLOR_GRAY2BGR)
            out.write(watermarked_bgr)
        else:
            # Write original frame
            out.write(frame)
            
        frame_count += 1
        
        # Progress callback could be added here
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save metadata
    np.savez(metadata_path, **metadata)


def extract_text_watermark_video(
    watermarked_video_path: str,
    metadata_path: str,
    output_path: str
) -> str:
    """
    Extract text watermark from watermarked video.
    
    Parameters
    ----------
    watermarked_video_path : str
        Path to watermarked video
    metadata_path : str
        Path to metadata file from embedding
    output_path : str
        Path to save extracted watermark image
        
    Returns
    -------
    str
        Original text if this was a text watermark
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
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            # DCT transform
            frame_dct = _dct2(gray_frame)
            
            # SVD decomposition
            _, S_wm, _ = svd(frame_dct, full_matrices=False)
            
            # Extract watermark singular values
            S_orig = original_s_values[i]
            Sw_est = (S_wm - S_orig) / alpha
            
            # Reconstruct watermark
            wm_dct_est = (Uw * Sw_est) @ Vtw
            wm_est = _idct2(wm_dct_est)[:shape[0], :shape[1]]
            
            extracted_watermarks.append(wm_est)
    
    cap.release()
    
    # Average extracted watermarks for better quality
    if extracted_watermarks:
        avg_watermark = np.mean(extracted_watermarks, axis=0)
        avg_watermark = np.clip(avg_watermark, 0, 255).astype(np.uint8)
        
        # Save extracted watermark
        Image.fromarray(avg_watermark).save(output_path)
        
        return original_text
    else:
        raise ValueError("No watermarked frames found")


def get_video_info(video_path: str) -> dict:
    """Get basic information about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info