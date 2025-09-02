# payload_qim.py
import cv2, numpy as np
from pathlib import Path
import struct, zlib

# --------- header & mime ----------
MAGIC = b"QIMW"           # 4B
VER   = 1                  # 1B
MIME_TO_EXT = {
    0: ".bin", 1: ".txt", 2: ".json", 3: ".png", 4: ".jpg", 5: ".jpeg", 6: ".bmp", 7: ".webp"
}
EXT_TO_MIME = {
    ".txt":1, ".json":2, ".png":3, ".jpg":4, ".jpeg":5, ".bmp":6, ".webp":7
}

def _mime_code(mime: str, name: str) -> int:
    mime = (mime or "").lower()
    if mime.startswith("text/"): return 1
    if mime == "application/json": return 2
    ext = Path(name).suffix.lower()
    return EXT_TO_MIME.get(ext, 0)

def pack_payload(name: str, mime: str, data: bytes) -> bytes:
    name_b = Path(name).name.encode("utf-8")
    code = _mime_code(mime, name)
    crc  = zlib.crc32(data) & 0xFFFFFFFF
    head = MAGIC + struct.pack(">BBHI I", VER, code, len(name_b), len(data), crc)
    return head + name_b + data

def unpack_payload(packed: bytes):
    if not packed.startswith(MAGIC):
        raise ValueError("bad magic")
    ver, code, nlen, dlen, crc = struct.unpack(">BBHI I", packed[4:4+10+2])
    off = 4+12
    name = packed[off:off+nlen].decode("utf-8", errors="ignore")
    data = packed[off+nlen:off+nlen+dlen]
    if (zlib.crc32(data) & 0xFFFFFFFF) != crc:
        raise ValueError("CRC mismatch")
    mime = MIME_TO_EXT.get(code, ".bin")
    return {"version": ver, "mime": mime, "name": name, "data": data}

# --------- QIM embedding on one coef ---------
# choose mid-frequency position
DCT_POS = (3, 2)  # (u,v) in 8x8 block

# --- QIM: lượng tử hoá bền vững bằng round + parity ---

def _qim_embed(val: float, bit: int, delta: float) -> float:
    """
    Lượng tử hoá theo bước Δ, đảm bảo parity (chẵn/lẻ) của chỉ số lượng tử
    trùng với bit cần giấu. Dùng round thay vì floor và chọn điểm gần nhất.
    """
    q = np.round(val / delta)            # chỉ số lượng tử (float -> int)
    parity  = int(q) & 1
    target  = int(bit) & 1
    if parity != target:
        up   = (q + 1) * delta
        down = (q - 1) * delta
        # dịch về phía gần giá trị gốc hơn để giảm méo
        q = (q + 1) if abs(up - val) <= abs(down - val) else (q - 1)
    return float(q * delta)

def _qim_decode(val: float, delta: float) -> int:
    """
    Giải mã QIM: bit = parity(chỉ số lượng tử), với chỉ số lấy bằng round(val/Δ).
    """
    q = int(np.round(val / delta))
    return q & 1


def _blocks(imgY: np.ndarray):
    h, w = imgY.shape
    for y in range(0, h - h % 8, 8):
        for x in range(0, w - w % 8, 8):
            yield y, x, imgY[y:y+8, x:x+8]

def _embed_stream_Y(Y: np.ndarray, bits: np.ndarray, delta: float):
    out = Y.copy().astype(np.float32)
    h, w = out.shape
    idx = 0
    for y, x, blk in _blocks(out):
        if idx >= len(bits): break
        d = cv2.dct(blk.astype(np.float32))
        u, v = DCT_POS
        d[u, v] = _qim_embed(d[u, v], int(bits[idx]), delta)
        idx += 1
        out[y:y+8, x:x+8] = cv2.idct(d)
    return np.clip(out, 0, 255).astype(np.uint8), idx

def _extract_stream_Y(Y: np.ndarray, nbits: int, delta: float):
    bits = []
    for _, _, blk in _blocks(Y.astype(np.float32)):
        if len(bits) >= nbits: break
        d = cv2.dct(blk)
        bits.append(_qim_decode(d[DCT_POS], delta))
    return np.array(bits, dtype=np.uint8)

def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) == 0: return b""
    m = int(np.ceil(len(bits)/8.0))*8
    pad = m - len(bits)
    if pad: bits = np.pad(bits, (0, pad))
    b = np.packbits(bits).tobytes()
    return b

def _bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

# --------- IMAGE ---------
def embed_to_image(host_path: str, out_path: str, payload: dict, delta: float = 3.0):
    img = cv2.imread(host_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(host_path)
    packed = pack_payload(payload["name"], payload.get("mime",""), payload["data"])
    bits = _bytes_to_bits(packed)

    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0]
    Y_new, used = _embed_stream_Y(Y, bits, delta=delta)
    if used < len(bits):
        raise ValueError(f"Host too small: capacity {used} bits < {len(bits)} bits")
    YCrCb[:,:,0] = Y_new
    out = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)

    # ALWAYS save PNG (lossless)
    ok = cv2.imwrite(out_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok: raise IOError("Failed to write PNG")
    return {"used_bits": int(used), "delta": float(delta), "out": out_path}

def _extract_image_once(stego_path: str, delta: float, save_dir: str|None):
    img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(stego_path)
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)

    # đọc “đủ lớn”, sau đó tìm MAGIC trong dòng bytes
    nbits = (Y.shape[0]//8) * (Y.shape[1]//8)  # 1 bit/khối
    stream = _bits_to_bytes(_extract_stream_Y(Y, nbits, delta))
    idx = stream.find(MAGIC)
    if idx == -1:
        raise ValueError("No payload found")

    # parse tối thiểu header để biết độ dài
    hdr = stream[idx: idx+4+12]
    if len(hdr) < 16:
        raise ValueError("Truncated header")
    ver, code, nlen, dlen, crc = struct.unpack(">BBHI I", hdr[4:16])
    need = 4+12 + nlen + dlen
    packed = stream[idx: idx+need]
    meta = unpack_payload(packed)

    ext = meta["mime"] if meta["mime"].startswith(".") else ".bin"
    name = Path(meta["name"]).name or ("payload" + ext)
    save_dir = Path(save_dir or Path(stego_path).with_suffix("").as_posix() + "_extracted")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / name
    out_file.write_bytes(meta["data"])
    return str(out_file)

def extract_from_image(stego_path: str, save_dir: str|None = None, delta: float = 3.0):
    # thử delta lân cận để “bắt” được payload sau nén/biến đổi
    for mul in (1.0, 0.9, 1.1, 0.8, 1.2, 1.3):
        try:
            return _extract_image_once(stego_path, delta*mul, save_dir)
        except Exception:
            continue
    raise ValueError("No payload found")

# --------- VIDEO ---------
def _open_writer(out_path: str, w: int, h: int, fps: float):
    ext = Path(out_path).suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # bền hơn
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))

def embed_to_video(host_path: str, out_path: str, payload: dict, delta: float = 3.0, frame_interval: int = 1):
    cap = cv2.VideoCapture(host_path)
    if not cap.isOpened(): raise FileNotFoundError(host_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    packed = pack_payload(payload["name"], payload.get("mime",""), payload["data"])
    bits = _bytes_to_bits(packed)
    idx = 0

    writer = _open_writer(out_path, w, h, fps)
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx < len(bits) and (frame_id % max(1,frame_interval) == 0):
            YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            Y = YCrCb[:,:,0]
            Y_new, used = _embed_stream_Y(Y, bits[idx:], delta=delta)
            idx += used
            YCrCb[:,:,0] = Y_new
            out = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
        else:
            out = frame
        writer.write(out)
        frame_id += 1

    writer.release(); cap.release()
    if idx < len(bits):
        raise ValueError(f"Video capacity {idx} bits < {len(bits)} bits")
    return {"used_bits": int(idx), "delta": float(delta), "out": out_path}

def _extract_video_once(stego_path: str, delta: float, frame_interval: int, save_dir: str|None):
    cap = cv2.VideoCapture(stego_path)
    if not cap.isOpened(): raise FileNotFoundError(stego_path)
    stream = bytearray()
    fid = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        if fid % max(1,frame_interval) == 0:
            Y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)
            nbits = (Y.shape[0]//8) * (Y.shape[1]//8)
            stream.extend(_bits_to_bytes(_extract_stream_Y(Y, nbits, delta)))
            # thử tìm MAGIC sớm để tiết kiệm thời gian
            idx = stream.find(MAGIC)
            if idx != -1 and len(stream) - idx >= 16:
                ver, code, nlen, dlen, crc = struct.unpack(">BBHI I", stream[idx+4:idx+16])
                need = 4+12 + nlen + dlen
                if len(stream) - idx >= need:
                    packed = bytes(stream[idx: idx+need])
                    meta = unpack_payload(packed)
                    ext = meta["mime"] if meta["mime"].startswith(".") else ".bin"
                    name = Path(meta["name"]).name or ("payload" + ext)
                    save_dir = Path(save_dir or Path(stego_path).with_suffix("").as_posix() + "_extracted")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out_file = save_dir / name
                    out_file.write_bytes(meta["data"])
                    cap.release()
                    return str(out_file)
        fid += 1

    cap.release()
    raise ValueError("No payload found")

def extract_from_video(stego_path: str, save_dir: str|None = None, delta: float = 3.0, frame_interval: int = 1):
    for mul in (1.0, 0.9, 1.1, 0.8, 1.2, 1.3):
        try:
            return _extract_video_once(stego_path, delta*mul, frame_interval, save_dir)
        except Exception:
            continue
    raise ValueError("No payload found")
