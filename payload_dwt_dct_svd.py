# payload_dwt_dct_svd.py
# Backend thuần OpenCV + PyWavelets: DWT (haar) + SVD trên các block của subband, QIM với repetition 3x.
# Hỗ trợ payload: text / json / image (PNG chuẩn hoá). Ảnh xuất PNG (lossless), Video xuất AVI (MJPG).

import os, io, json, zlib
from pathlib import Path
import numpy as np
import cv2
import pywt
from PIL import Image

MAGIC = b'PDSV'          # 4B magic để nhận diện
MAX_BYTES = 4096         # tổng byte payload (gồm header), tăng nếu muốn giấu lớn hơn
BLOCK = 8                # kích thước khối trong subband
REP = 3                  # mã lặp 3x để tăng độ bền
WAVELET = 'haar'         # dwt2/ idwt2
SUBBAND = 'LH'           # dùng LH để nhúng (ổn định hơn HH)

# ----------------- Utils: pack/unpack payload -----------------
# header = MAGIC(4) + type(1) + flags(1) + length(4, BE) + data + crc32(4)
# type: 0=text, 1=json, 2=image ; flags bit0=compressed(zlib)
def _to_payload_bytes(payload, kind: str) -> bytes:
    if kind == 'text':
        raw = str(payload).encode('utf-8'); t = 0
    elif kind == 'json':
        if isinstance(payload, str):
            obj = json.loads(payload)
        else:
            obj = payload
        raw = json.dumps(obj, ensure_ascii=False).encode('utf-8'); t = 1
    elif kind == 'image':
        # chấp nhận path hoặc bytes -> chuẩn hoá PNG
        if isinstance(payload, (bytes, bytearray)):
            img_bytes = payload
        else:
            img_bytes = Path(str(payload)).read_bytes()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        bio = io.BytesIO(); img.save(bio, format='PNG'); raw = bio.getvalue(); t = 2
    else:
        raise ValueError('Unsupported payload kind (text/json/image)')

    flags = 1
    comp = zlib.compress(raw)
    length = len(comp)
    header = MAGIC + bytes([t]) + bytes([flags]) + length.to_bytes(4, 'big')
    blob = header + comp + zlib.crc32(comp).to_bytes(4, 'big')
    if len(blob) > MAX_BYTES:
        raise ValueError(f'Payload quá lớn ({len(blob)}B) > MAX_BYTES={MAX_BYTES}. Tăng MAX_BYTES hoặc rút gọn.')
    return blob + b'\x00' * (MAX_BYTES - len(blob))

def _from_payload_bytes(blob: bytes):
    if len(blob) < 16 or not blob.startswith(MAGIC):
        raise ValueError('MAGIC không khớp.')
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

def _bytes_to_bits(b: bytes) -> np.ndarray:
    if not b: return np.zeros((0,), np.uint8)
    arr = np.frombuffer(b, dtype=np.uint8)
    return np.unpackbits(arr)

def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) == 0: return b''
    m = (len(bits)+7)//8*8
    if m > len(bits):
        bits = np.pad(bits, (0, m-len(bits)), 'constant')
    return np.packbits(bits).tobytes()

def _repeat3(bits: np.ndarray) -> np.ndarray:
    return np.repeat(bits.astype(np.uint8), REP)

def _unrepeat3_try_find_magic(bits: np.ndarray):
    """Thử 3 lệch pha (0,1,2) cho repetition 3x. Trả về stream bytes và idx MAGIC nếu tìm thấy đầy đủ."""
    for off in range(REP):
        usable = (len(bits)-off)//REP*REP
        if usable <= 0: continue
        b = bits[off:off+usable].reshape(-1, REP)
        maj = (b.sum(axis=1) >= 2).astype(np.uint8)
        stream = _bits_to_bytes(maj)
        idx = stream.find(MAGIC)
        if idx != -1 and len(stream) >= idx + 16:
            # đọc độ dài cần thiết
            t = stream[idx+4]
            length = int.from_bytes(stream[idx+6:idx+10], 'big')
            need = 4 + 1 + 1 + 4 + length + 4  # MAGIC + type + flags + len + data + crc
            if len(stream) >= idx + need:
                return stream, idx, need
    return None, -1, -1

# --------------- DWT/SVD/QIM core ----------------
def _iter_blocks(mat: np.ndarray, B=BLOCK):
    H, W = mat.shape
    Hc = H - H % B
    Wc = W - W % B
    for y in range(0, Hc, B):
        for x in range(0, Wc, B):
            yield y, x

def _median_s0(mat: np.ndarray, B=BLOCK) -> float:
    vals = []
    for y, x in _iter_blocks(mat, B):
        S = np.linalg.svd(mat[y:y+B, x:x+B], compute_uv=False)
        vals.append(float(S[0]))
    return float(np.median(vals)) if vals else 1.0

def _delta_from_strength(strength: float, med_s0: float) -> float:
    # adaptive step theo median S0 của subband -> bền hơn
    base = 0.08 + 0.60 * float(strength)   # 0.08..0.26
    delta = max(0.5, med_s0 * base)
    return float(delta)

def _qim_embed_s0(s0: float, bit: int, delta: float) -> float:
    q = np.round(s0 / delta)
    if (int(q) & 1) != (int(bit) & 1):
        up   = (q + 1) * delta
        down = (q - 1) * delta
        q = (q + 1) if abs(up - s0) <= abs(down - s0) else (q - 1)
    return float(q * delta)

def _qim_decode_s0(s0: float, delta: float) -> int:
    q = int(np.round(s0 / delta))
    return q & 1

def _embed_bits_into_subband(sub: np.ndarray, bits: np.ndarray, delta: float, B=BLOCK):
    out = sub.copy().astype(np.float32)
    idx = 0
    for y, x in _iter_blocks(out, B):
        if idx >= len(bits): break
        blk = out[y:y+B, x:x+B]
        U, S, Vt = np.linalg.svd(blk, full_matrices=False)
        S[0] = _qim_embed_s0(S[0], int(bits[idx]), delta)
        out[y:y+B, x:x+B] = (U @ np.diag(S) @ Vt).astype(np.float32)
        idx += 1
    return out, idx

def _extract_bits_from_subband(sub: np.ndarray, nbits: int, delta: float, B=BLOCK):
    bits = []
    for y, x in _iter_blocks(sub, B):
        if len(bits) >= nbits: break
        S = np.linalg.svd(sub[y:y+B, x:x+B], compute_uv=False)
        bits.append(_qim_decode_s0(float(S[0]), delta))
    return np.array(bits, dtype=np.uint8)

# --------------- IMAGE API ----------------
def embed_into_image(host_path: str, out_path: str, *, kind: str, payload, strength: float = 0.12) -> str:
    bgr = cv2.imread(host_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Không mở được ảnh: {host_path}')
    YCrCb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = YCrCb[:,:,0]

    # DWT
    LL, (LH, HL, HH) = pywt.dwt2(Y, WAVELET)

    # payload -> bits (repetition 3x)
    blob = _to_payload_bytes(payload, kind)
    bits = _bytes_to_bits(blob)
    bits = _repeat3(bits)

    # delta adaptive
    med = _median_s0(LH, BLOCK)
    delta = _delta_from_strength(strength, med)

    cap_bits = (LH.shape[0]//BLOCK)*(LH.shape[1]//BLOCK)
    if len(bits) > cap_bits:
        raise ValueError(f'Host quá nhỏ: capacity {cap_bits} bits < {len(bits)} bits.')

    LH2, used = _embed_bits_into_subband(LH, bits, delta, BLOCK)
    if used < len(bits):
        raise ValueError('Nhúng chưa hết payload (thiếu khối).')

    # I-DWT
    Y2 = pywt.idwt2((LL, (LH2, HL, HH)), WAVELET)
    Y2 = np.clip(Y2, 0, 255)
    YCrCb[:,:,0] = Y2[:Y.shape[0], :Y.shape[1]]
    stego = cv2.cvtColor(YCrCb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    # luôn PNG để tránh mất mát
    if not out_path.lower().endswith('.png'):
        out_path = os.path.splitext(out_path)[0] + '_stego.png'
    ok = cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise IOError('Ghi PNG thất bại.')
    return out_path

def extract_from_image(stego_path: str, save_dir: str, strength: float = 0.12) -> str:
    bgr = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Không mở được ảnh: {stego_path}')
    YCrCb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = YCrCb[:,:,0]

    LL, (LH, HL, HH) = pywt.dwt2(Y, WAVELET)
    med = _median_s0(LH, BLOCK)
    delta = _delta_from_strength(strength, med)

    # đọc toàn bộ bit (tối đa capacity)
    cap_bits = (LH.shape[0]//BLOCK)*(LH.shape[1]//BLOCK)
    bits = _extract_bits_from_subband(LH, cap_bits, delta, BLOCK)

    # gỡ lặp và tìm MAGIC
    stream, idx, need = _unrepeat3_try_find_magic(bits)
    if stream is None:
        raise ValueError('Không tìm thấy payload.')
    packed = stream[idx: idx+need]
    kind, ext, raw = _from_payload_bytes(packed)

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'payload.{ext}')
    with open(out, 'wb') as f:
        f.write(raw)
    return out

# --------------- VIDEO API ----------------
def embed_into_video(host_video: str, out_video: str, *, kind: str, payload, strength: float = 0.12, frame_interval: int = 1) -> str:
    cap = cv2.VideoCapture(host_video)
    if not cap.isOpened():
        raise ValueError(f'Không mở được video: {host_video}')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # writer: AVI (MJPG) bền hơn
    if not out_video.lower().endswith('.avi'):
        out_video = os.path.splitext(out_video)[0] + '_stego.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
    if not vw.isOpened():
        cap.release()
        raise ValueError('Không mở được VideoWriter (MJPG).')

    # payload bits (repetition 3x)
    blob = _to_payload_bytes(payload, kind)
    bits = _repeat3(_bytes_to_bits(blob))
    bit_idx = 0

    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if (fid % max(1, frame_interval) == 0) and (bit_idx < len(bits)):
            YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            Y = YCrCb[:,:,0]
            LL, (LH, HL, HH) = pywt.dwt2(Y, WAVELET)

            med = _median_s0(LH, BLOCK)
            delta = _delta_from_strength(strength, med)
            # capacity frame
            cap_bits = (LH.shape[0]//BLOCK)*(LH.shape[1]//BLOCK)
            take = min(cap_bits, len(bits)-bit_idx)
            LH2, used = _embed_bits_into_subband(LH, bits[bit_idx:bit_idx+take], delta, BLOCK)
            bit_idx += used

            Y2 = pywt.idwt2((LL, (LH2, HL, HH)), WAVELET)
            Y2 = np.clip(Y2, 0, 255)
            YCrCb[:,:,0] = Y2[:H, :W]
            frame = cv2.cvtColor(YCrCb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        vw.write(frame)
        fid += 1

    cap.release(); vw.release()
    if bit_idx < len(bits):
        raise ValueError(f'Video capacity thiếu: embedded {bit_idx}/{len(bits)} bits.')
    return out_video

def extract_from_video(stego_video: str, save_dir: str, strength: float = 0.12, frame_interval: int = 1) -> str:
    cap = cv2.VideoCapture(stego_video)
    if not cap.isOpened():
        raise ValueError(f'Không mở được video: {stego_video}')

    bits_acc = []
    fid = 0
    ok_any = False
    packed = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        if (fid % max(1, frame_interval)) == 0:
            YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            Y = YCrCb[:,:,0]
            LL, (LH, HL, HH) = pywt.dwt2(Y, WAVELET)

            med = _median_s0(LH, BLOCK)
            delta = _delta_from_strength(strength, med)

            cap_bits = (LH.shape[0]//BLOCK)*(LH.shape[1]//BLOCK)
            bits = _extract_bits_from_subband(LH, cap_bits, delta, BLOCK)
            bits_acc.extend(bits.tolist())

            # mỗi khung lại thử gỡ lặp & tìm MAGIC sớm
            stream, idx, need = _unrepeat3_try_find_magic(np.array(bits_acc, dtype=np.uint8))
            if stream is not None:
                packed = stream[idx: idx+need]
                ok_any = True
                break
        fid += 1

    cap.release()
    if not ok_any:
        raise ValueError('Không tìm thấy payload trong video.')

    kind, ext, raw = _from_payload_bytes(packed)
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'payload.{ext}')
    with open(out, 'wb') as f:
        f.write(raw)
    return out
