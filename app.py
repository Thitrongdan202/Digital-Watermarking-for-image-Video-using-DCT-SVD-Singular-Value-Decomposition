import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from watermark.dct_svd import embed_watermark, extract_watermark, singular_values
from watermark.ai import WatermarkDetector


class WatermarkApp(tk.Tk):
    """Tkinter UI for DCT-SVD watermarking with a simple AI detector."""

    def __init__(self) -> None:
        super().__init__()
        self.title("DCT-SVD Watermarking")
        self.geometry("500x250")
        self.detector = WatermarkDetector()
        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        embed_frame = ttk.Frame(notebook, padding=10)
        extract_frame = ttk.Frame(notebook, padding=10)
        detect_frame = ttk.Frame(notebook, padding=10)

        notebook.add(embed_frame, text="Embed")
        notebook.add(extract_frame, text="Extract")
        notebook.add(detect_frame, text="Detect")

        # Embed tab
        self.host_var = tk.StringVar()
        self.wm_var = tk.StringVar()
        self.output_var = tk.StringVar(value="watermarked.png")
        self.meta_var = tk.StringVar(value="metadata.npz")

        ttk.Button(embed_frame, text="Host Image", command=self._pick_host).grid(row=0, column=0, sticky="ew")
        ttk.Entry(embed_frame, textvariable=self.host_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(embed_frame, text="Watermark", command=self._pick_wm).grid(row=1, column=0, sticky="ew")
        ttk.Entry(embed_frame, textvariable=self.wm_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(embed_frame, text="Embed", command=self._embed).grid(row=2, column=0, columnspan=2, pady=5)

        # Extract tab
        self.wm_image_var = tk.StringVar()
        self.meta_extract_var = tk.StringVar(value="metadata.npz")
        self.extracted_var = tk.StringVar(value="extracted.png")

        ttk.Button(extract_frame, text="Watermarked", command=self._pick_wm_image).grid(row=0, column=0, sticky="ew")
        ttk.Entry(extract_frame, textvariable=self.wm_image_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(extract_frame, text="Metadata", command=self._pick_meta_extract).grid(row=1, column=0, sticky="ew")
        ttk.Entry(extract_frame, textvariable=self.meta_extract_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(extract_frame, text="Extract", command=self._extract).grid(row=2, column=0, columnspan=2, pady=5)

        # Detect tab
        self.detect_var = tk.StringVar()
        ttk.Button(detect_frame, text="Image", command=self._pick_detect).grid(row=0, column=0, sticky="ew")
        ttk.Entry(detect_frame, textvariable=self.detect_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(detect_frame, text="Detect", command=self._detect).grid(row=1, column=0, columnspan=2, pady=5)

    # Helper methods for file dialogs
    def _pick_host(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.host_var.set(path)

    def _pick_wm(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.wm_var.set(path)

    def _embed(self) -> None:
        if not self.host_var.get() or not self.wm_var.get():
            messagebox.showerror("Error", "Please select host and watermark images")
            return
        try:
            embed_watermark(self.host_var.get(), self.wm_var.get(),
                            self.output_var.get(), self.meta_var.get())
            messagebox.showinfo("Success", f"Watermarked saved to {self.output_var.get()}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _pick_wm_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.wm_image_var.set(path)

    def _pick_meta_extract(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz")])
        if path:
            self.meta_extract_var.set(path)

    def _extract(self) -> None:
        if not self.wm_image_var.get() or not self.meta_extract_var.get():
            messagebox.showerror("Error", "Select watermarked image and metadata")
            return
        try:
            extract_watermark(self.wm_image_var.get(), self.meta_extract_var.get(),
                              self.extracted_var.get())
            messagebox.showinfo("Success", f"Watermark saved to {self.extracted_var.get()}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _pick_detect(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.detect_var.set(path)

    def _detect(self) -> None:
        if not self.detect_var.get():
            messagebox.showerror("Error", "Select image to detect")
            return
        try:
            s = singular_values(self.detect_var.get())
            pred = self.detector.predict(s.reshape(1, -1))[0]
            msg = "Watermark detected" if pred == 1 else "No watermark detected"
            messagebox.showinfo("Result", msg)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    app = WatermarkApp()
    app.mainloop()
