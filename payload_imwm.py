# payload_imwm.py
# Backend watermark dựa trên thư viện 'imwatermark' (DWT-DCT-SVD) + header MAGIC, CRC.
# Hỗ trợ payload: text / json / image(PNG/JPG). Ảnh xuất PNG, Video xuất AVI(MJPG).

import os, io, json, zlib, cv2, numpy as np
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder

ALGO = 'dwtDctSvd'
MAGIC = b'IMWM'
MAX_BYTES = 4096  # độ dài payload cố định (có header), tăng nếu cần giấu lớn hơn

# header: MAGIC(4) + type(1) + flags(1) + len(4, BE) + data + crc32(4)
# type: 0=text, 1=json, 2=image ; flags bit0=compressed(zlib)

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _to_payload_bytes(payload, kind: str) -> bytes:
    if kind == 'text':
        raw = str(payload).encode('utf-8'); t = 0
    elif kind == 'json':
        # payload có thể là dict hoặc chuỗi JSON
        raw = json.dumps(payload if not isinstance(payload, str) else json.loads(payload),
                         ensure_ascii=False).encode('utf-8')
        t = 1
    elif kind == 'image':
        # chấp nhận đường dẫn hoặc bytes
        if isinstance(payload, (bytes, bytearray)):
            img_bytes = payload
        else:
            with open(payload, 'rb') as f: img_bytes = f.read()
        # chuẩn hoá về PNG để đồng nhất
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        bio = io.BytesIO(); img.save(bio, format='PNG'); raw = bio.getvalue()
        t = 2
    else:
        raise ValueError('Unsupported payload kind (text/json/image)')

    flags = 1
    comp = zlib.compress(raw)
    length = len(comp)
    header = MAGIC + bytes([t]) + bytes([flags]) + length.to_bytes(4, 'big')
    blob = header + comp + zlib.crc32(comp).to_bytes(4, 'big')
    if len(blob) > MAX_BYTES:
        raise ValueError(f'Payload quá lớn ({len(blob)}B) > MAX_BYTES={MAX_BYTES}. Tăng MAX_BYTES hoặc rút gọn.')
    # pad tới MAX_BYTES
    return blob + b'\x00' * (MAX_BYTES - len(blob))

def _from_payload_bytes(blob: bytes):
    if len(blob) < 16 or not blob.startswith(MAGIC):
        raise ValueError('Header MAGIC không khớp.')
    t = blob[4]
    flags = blob[5]
    length = int.from_bytes(blob[6:10], 'big')
    comp = blob[10:10+length]
    crc  = int.from_bytes(blob[10+length:14+length], 'big')
    if zlib.crc32(comp) != crc:
        raise ValueError('CRC sai (payload hỏng).')
    raw = zlib.decompress(comp) if (flags & 1) else comp
    if t == 0:   return 'text', 'txt',  raw
    if t == 1:   return 'json', 'json', raw
    if t == 2:   return 'image','png',  raw
    raise ValueError('Loại payload không hỗ trợ.')

def _encode_bytes_into_image(bgr: np.ndarray, pbytes: bytes, strength: float = 0.12) -> np.ndarray:
    enc = WatermarkEncoder()
    enc.set_watermark('bytes', pbytes)
    return enc.encode(bgr, ALGO, wmStrength=strength)

def _decode_bytes_from_image(bgr: np.ndarray, strength: float = 0.12) -> bytes:
    dec = WatermarkDecoder('bytes', MAX_BYTES)
    for mul in (1.00, 0.95, 1.05, 0.90, 1.10, 0.85, 1.15, 0.80, 1.20):
        try:
            return dec.decode(bgr, ALGO, wmStrength=strength*mul)
        except Exception:
            continue
    raise ValueError('Không tìm thấy payload trong ảnh.')

# ---------------- Public API: IMAGE ----------------
def embed_into_image(host_path: str, out_path: str, *, kind: str, payload, strength: float = 0.12) -> str:
    bgr = cv2.imread(host_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Không mở được ảnh: {host_path}')
    pbytes = _to_payload_bytes(payload, kind)
    stego = _encode_bytes_into_image(bgr, pbytes, strength)
    if not out_path.lower().endswith('.png'):
        out_path = os.path.splitext(out_path)[0] + '_stego.png'
    _ensure_dir(out_path)
    ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok: raise IOError('Ghi ảnh PNG thất bại.')
    return out_path

def extract_from_image(stego_path: str, save_dir: str, strength: float = 0.12) -> str:
    bgr = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Không mở được ảnh: {stego_path}')
    blob = _decode_bytes_from_image(bgr, strength)
    kind, ext, raw = _from_payload_bytes(blob)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'payload.{ext}')
    with open(out, 'wb') as f: f.write(raw)
    return out

# ---------------- Public API: VIDEO ----------------
def embed_into_video(host_video: str, out_video: str, *, kind: str, payload, strength: float = 0.12, frame_interval: int = 1) -> str:
    cap = cv2.VideoCapture(host_video)
    if not cap.isOpened():
        raise ValueError(f'Không mở được video: {host_video}')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not out_video.lower().endswith('.avi'):
        out_video = os.path.splitext(out_video)[0] + '_stego.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # bền với watermark
    vw = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
    if not vw.isOpened():
        cap.release()
        raise ValueError('Không mở được VideoWriter (codec MJPG).')

    pbytes = _to_payload_bytes(payload, kind)
    enc = WatermarkEncoder(); enc.set_watermark('bytes', pbytes)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % max(1, frame_interval) == 0:
            try:
                frame = enc.encode(frame, ALGO, wmStrength=strength)
            except Exception:
                pass
        vw.write(frame)
        idx += 1

    cap.release(); vw.release()
    return out_video

def extract_from_video(stego_video: str, save_dir: str, strength: float = 0.12, frame_interval: int = 1) -> str:
    cap = cv2.VideoCapture(stego_video)
    if not cap.isOpened():
        raise ValueError(f'Không mở được video: {stego_video}')
    dec = WatermarkDecoder('bytes', MAX_BYTES)

    ok_any = False; blob_best = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % max(1, frame_interval) == 0:
            for mul in (1.00, 0.95, 1.05, 0.90, 1.10):
                try:
                    blob = dec.decode(frame, ALGO, wmStrength=strength*mul)
                    if blob is not None and blob.startswith(MAGIC):
                        blob_best = blob; ok_any = True
                        break
                except Exception:
                    continue
        if ok_any: break
        idx += 1

    cap.release()
    if not ok_any:
        raise ValueError('Không tìm thấy payload trong video.')

    kind, ext, raw = _from_payload_bytes(blob_best)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'payload.{ext}')
    with open(out, 'wb') as f: f.write(raw)
    return out
