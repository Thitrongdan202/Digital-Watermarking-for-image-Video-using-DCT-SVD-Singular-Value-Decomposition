import cv2
import numpy as np
from pathlib import Path
from .dct_svd import _dct2, _idct2, create_text_watermark
from numpy.linalg import svd
from PIL import Image
import tempfile
import os
import subprocess
import shutil


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


def embed_watermark_video_color_with_audio(
    host_video_path: str,
    watermark_path: str,
    output_video_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    frame_interval: int = 10
) -> None:
    """
    Embed watermark into video frames using DCT-SVD with color preservation and audio preservation.
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
    
    # Create temporary video file (no audio)
    temp_video_path = output_video_path + '.temp.mp4'
    
    # Setup video writer - keep color
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height), isColor=True)
    
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
    
    # Try to preserve audio using ffmpeg
    audio_preserved = _preserve_audio_with_ffmpeg(temp_video_path, host_video_path, output_video_path)
    
    if audio_preserved:
        # Remove temporary video file
        os.remove(temp_video_path)
        print("Video watermarked successfully with audio preserved!")
    else:
        # Fallback: rename temp file to output (no audio)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        os.rename(temp_video_path, output_video_path)
        print("Video watermarked successfully (audio not preserved - install ffmpeg for audio support)")
    
    # Save metadata
    np.savez(metadata_path, **metadata)


def embed_text_watermark_video_color_with_audio(
    host_video_path: str,
    text: str,
    output_video_path: str,
    metadata_path: str,
    alpha: float = 0.05,
    font_size: int = 40,
    frame_interval: int = 10
) -> None:
    """
    Embed text watermark into video frames using DCT-SVD with color preservation and audio preservation.
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
    
    # Create temporary video file (no audio)
    temp_video_path = output_video_path + '.temp.mp4'
    
    # Setup video writer - keep color
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height), isColor=True)
    
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
    
    # Try to preserve audio using ffmpeg
    audio_preserved = _preserve_audio_with_ffmpeg(temp_video_path, host_video_path, output_video_path)
    
    if audio_preserved:
        # Remove temporary video file
        os.remove(temp_video_path)
        print("Text watermarked video created successfully with audio preserved!")
    else:
        # Fallback: rename temp file to output (no audio)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        os.rename(temp_video_path, output_video_path)
        print("Text watermarked video created successfully (audio not preserved - install ffmpeg for audio support)")
    
    # Save metadata
    np.savez(metadata_path, **metadata)