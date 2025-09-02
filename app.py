# app.py
import io, json, zlib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from payload_qim import (
    embed_to_image, embed_to_video,
    extract_from_image, extract_from_video
)

# -------------------------- helpers --------------------------
def is_image(p: str) -> bool:
    return Path(p).suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

def is_video(p: str) -> bool:
    return Path(p).suffix.lower() in [".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"]

def map_strength_to_delta(alpha: float) -> float:
    # QIM step (Δ). Tăng nhẹ để bền hơn trước nén/biến đổi.
    return 2.2 + float(alpha) * 12.0

# -------------------------- GUI --------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DCT-SVD Watermarking (QIM payload)")
        self.geometry("1080x640")

        nb = ttk.Notebook(self)
        self.tab_embed = ttk.Frame(nb)
        self.tab_detect = ttk.Frame(nb)
        nb.add(self.tab_embed, text="EMBED")
        nb.add(self.tab_detect, text="DETECT")
        nb.pack(fill="both", expand=True)

        # ---------- EMBED ----------
        # host
        frmH = ttk.LabelFrame(self.tab_embed, text="Host Media")
        frmH.pack(fill="x", padx=8, pady=6)
        self.host_path = tk.StringVar()
        ttk.Button(frmH, text="Browse", command=self._pick_host).pack(side="left", padx=6, pady=6)
        ttk.Entry(frmH, textvariable=self.host_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)

        # payload area
        frmP = ttk.LabelFrame(self.tab_embed, text="Payload")
        frmP.pack(fill="x", padx=8, pady=6)

        self.pl_type = tk.StringVar(value="text")
        row = ttk.Frame(frmP); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Type:").pack(side="left")
        for k, v in [("Text", "text"), ("Image", "image"), ("JSON", "json"), ("File", "file")]:
            ttk.Radiobutton(row, text=k, value=v, variable=self.pl_type).pack(side="left", padx=6)

        self.ent_text = tk.Text(frmP, height=5)
        self.ent_text.pack(fill="both", padx=6, pady=6)

        # For image / file path
        row2 = ttk.Frame(frmP); row2.pack(fill="x", padx=4, pady=2)
        self.payload_path = tk.StringVar()
        ttk.Button(row2, text="Pick payload file", command=self._pick_payload).pack(side="left")
        ttk.Entry(row2, textvariable=self.payload_path).pack(side="left", fill="x", expand=True, padx=6)

        # Strength & frame interval
        frmS = ttk.LabelFrame(self.tab_embed, text="Settings")
        frmS.pack(fill="x", padx=8, pady=6)
        self.strength = tk.DoubleVar(value=0.12)  # 0..1
        ttk.Label(frmS, text="Watermark Strength").pack(side="left", padx=8)
        ttk.Scale(frmS, from_=0.02, to=0.5, variable=self.strength, orient="horizontal", length=220).pack(side="left")
        ttk.Label(frmS, text="Frame interval (video)").pack(side="left", padx=10)
        self.fint = tk.IntVar(value=1)
        ttk.Spinbox(frmS, from_=1, to=10, textvariable=self.fint, width=6).pack(side="left")

        # embed button
        self.embed_btn = ttk.Button(self.tab_embed, text="EMBED WATERMARK", command=self._do_embed)
        self.embed_btn.pack(pady=10)
        self.embed_prog = ttk.Progressbar(self.tab_embed, length=360, mode="determinate")
        self.embed_prog.pack(pady=2)

        # ---------- DETECT ----------
        frmD = ttk.LabelFrame(self.tab_detect, text="Watermarked Media")
        frmD.pack(fill="x", padx=8, pady=6)
        self.water_path = tk.StringVar()
        ttk.Button(frmD, text="Browse", command=self._pick_stego).pack(side="left", padx=6, pady=6)
        ttk.Entry(frmD, textvariable=self.water_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)

        frmDS = ttk.LabelFrame(self.tab_detect, text="Detect Settings")
        frmDS.pack(fill="x", padx=8, pady=6)
        self.det_strength = tk.DoubleVar(value=0.12)
        ttk.Label(frmDS, text="Strength (same as embed)").pack(side="left", padx=8)
        ttk.Scale(frmDS, from_=0.02, to=0.5, variable=self.det_strength, orient="horizontal", length=220).pack(side="left")
        ttk.Label(frmDS, text="Frame interval").pack(side="left", padx=10)
        self.det_fint = tk.IntVar(value=1)
        ttk.Spinbox(frmDS, from_=1, to=10, textvariable=self.det_fint, width=6).pack(side="left")

        self.detect_btn = ttk.Button(self.tab_detect, text="DETECT WATERMARK", command=self._do_detect)
        self.detect_btn.pack(pady=10)
        self.detect_prog = ttk.Progressbar(self.tab_detect, length=360, mode="determinate"); self.detect_prog.pack()

        self.preview = ttk.LabelFrame(self.tab_detect, text="Preview / Results")
        self.preview.pack(fill="both", expand=True, padx=8, pady=6)
        self.preview_canvas = tk.Label(self.preview)
        self.preview_canvas.pack(padx=6, pady=6)

    # -------------- callbacks --------------
    def _pick_host(self):
        p = filedialog.askopenfilename(title="Pick host image/video")
        if p: self.host_path.set(p)

    def _pick_payload(self):
        p = filedialog.askopenfilename(title="Pick payload file (image/json/any)")
        if p: self.payload_path.set(p)

    def _pick_stego(self):
        p = filedialog.askopenfilename(title="Pick watermarked media")
        if p: self.water_path.set(p)

    def _payload_bytes(self):
        t = self.pl_type.get()
        if t == "text":
            data = self.ent_text.get("1.0", "end").encode("utf-8")
            mime = "text/plain"; name = "message.txt"
        elif t == "json":
            raw = self.ent_text.get("1.0", "end")
            try:
                # normalize JSON
                data = json.dumps(json.loads(raw), ensure_ascii=False).encode("utf-8")
            except Exception:
                data = raw.encode("utf-8")
            mime = "application/json"; name = "data.json"
        elif t == "image":
            f = self.payload_path.get()
            if not f: raise ValueError("Please choose payload image")
            data = Path(f).read_bytes()
            mime = "image/" + Path(f).suffix.lower().strip(".")
            name = Path(f).name
        else:  # file
            f = self.payload_path.get()
            if not f: raise ValueError("Please choose payload file")
            data = Path(f).read_bytes()
            mime = "application/octet-stream"; name = Path(f).name
        return name, mime, data

    def _do_embed(self):
        host = self.host_path.get().strip()
        if not host:
            messagebox.showwarning("Embed", "Pick a host image/video"); return
        if not (is_image(host) or is_video(host)):
            messagebox.showwarning("Embed", "Unsupported host"); return

        try:
            name, mime, data = self._payload_bytes()
        except Exception as ex:
            messagebox.showerror("Embed", str(ex)); return

        # output path: image->PNG (lossless), video->AVI (MJPG)
        host_p = Path(host)
        if is_image(host):
            out = host_p.with_name(host_p.stem + "_stego.png")
        else:
            out = host_p.with_name(host_p.stem + "_stego.avi")

        delta = map_strength_to_delta(self.strength.get())
        fint = int(self.fint.get())

        self.embed_btn.config(state="disabled"); self.embed_prog.config(value=10)
        try:
            if is_image(host):
                info = embed_to_image(str(host_p), str(out),
                                      payload={"name": name, "mime": mime, "data": data},
                                      delta=delta)
            else:
                info = embed_to_video(str(host_p), str(out),
                                      payload={"name": name, "mime": mime, "data": data},
                                      delta=delta, frame_interval=fint)
            self.embed_prog.config(value=100)
            messagebox.showinfo("Embed", f"Done!\nOutput: {out}\nInfo: {info}")
        except Exception as ex:
            messagebox.showerror("Embed", f"Failed:\n{ex}")
        finally:
            self.embed_btn.config(state="normal"); self.embed_prog.config(value=0)

    def _preview_extracted(self, fpath: str):
        self.preview_canvas.configure(image="", text="")
        p = Path(fpath)
        ext = p.suffix.lower()
        try:
            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
                im = Image.open(p)
                im.thumbnail((640, 480))
                imtk = ImageTk.PhotoImage(im)
                self.preview_canvas.image = imtk
                self.preview_canvas.configure(image=imtk)
            elif ext in [".txt", ".json", ".md", ".csv", ".ini"]:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if len(txt) > 1200: txt = txt[:1200] + "\n...(truncated)"
                self.preview_canvas.configure(text=txt, anchor="nw", justify="left")
            else:
                self.preview_canvas.configure(text=str(p))
        except Exception as ex:
            self.preview_canvas.configure(text=f"{p}\n{ex}")

    def _do_detect(self):
        stego = self.water_path.get().strip()
        if not stego:
            messagebox.showwarning("Detect", "Pick stego image/video"); return
        if not (is_image(stego) or is_video(stego)):
            messagebox.showwarning("Detect", "Unsupported media"); return

        delta = map_strength_to_delta(self.det_strength.get())
        fint = int(self.det_fint.get())

        self.detect_btn.config(state="disabled"); self.detect_prog.config(value=10)
        try:
            if is_image(stego):
                out_file = extract_from_image(stego_path=stego, save_dir=None, delta=delta)
            else:
                out_file = extract_from_video(stego_path=stego, save_dir=None, delta=delta, frame_interval=fint)
            self.detect_prog.config(value=100)
            self._preview_extracted(out_file)
            messagebox.showinfo("Detect", f"Recovered payload:\n{out_file}")
        except Exception as ex:
            messagebox.showerror("Detect", f"Failed:\n{ex}")
        finally:
            self.detect_btn.config(state="normal"); self.detect_prog.config(value=0)

if __name__ == "__main__":
    App().mainloop()
