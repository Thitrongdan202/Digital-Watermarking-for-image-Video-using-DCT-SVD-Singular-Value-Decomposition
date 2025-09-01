import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import os
import time
import cv2
from pathlib import Path
from watermark.dct_svd import embed_watermark, extract_watermark, singular_values
from watermark.video_dct_svd import (
    embed_watermark_video, 
    extract_watermark_video, 
    detect_watermark_video,
    get_video_info
)
from watermark.ai import WatermarkDetector


class ModernTooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
    
    def on_enter(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        self.tooltip.configure(bg="#2d2d2d")
        label = tk.Label(self.tooltip, text=self.text, background="#2d2d2d", 
                        foreground="white", font=("Segoe UI", 9))
        label.pack()
    
    def on_leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class WatermarkApp(tk.Tk):
    """Modern Tkinter UI for DCT-SVD watermarking with Material Design elements."""

    def __init__(self) -> None:
        super().__init__()
        self.title("DCT-SVD Watermarking")
        self.geometry("1200x800")
        self.configure(bg="#1a1a1a")
        self.detector = WatermarkDetector()
        self.dark_theme = True
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark theme colors
        self.colors = {
            'bg': '#1a1a1a',
            'card_bg': '#2d2d2d',
            'accent': '#3f51b5',
            'text': '#ffffff',
            'text_secondary': '#b0b0b0',
            'success': '#4caf50',
            'error': '#f44336'
        }
        
        # Configure styles
        self.style.configure('Dark.TNotebook', background=self.colors['bg'])
        self.style.configure('Dark.TNotebook.Tab', background=self.colors['card_bg'], 
                           foreground=self.colors['text'], padding=[20, 10])
        self.style.configure('Modern.TFrame', background=self.colors['bg'])
        self.style.configure('Card.TFrame', background=self.colors['card_bg'], relief='flat')
        self.style.configure('Modern.TButton', background=self.colors['accent'], 
                           foreground='white', font=('Segoe UI', 10))

    def _build_ui(self) -> None:
        # Menu bar
        self._create_menu()
        
        # Main container
        main_frame = tk.Frame(self, bg=self.colors['bg'])
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Notebook with custom styling
        notebook = ttk.Notebook(main_frame, style='Dark.TNotebook')
        notebook.pack(fill="both", expand=True)

        # Create tabs with modern styling
        embed_frame = self._create_embed_tab()
        extract_frame = self._create_extract_tab()
        detect_frame = self._create_detect_tab()

        notebook.add(embed_frame, text="🔧 EMBED")
        notebook.add(extract_frame, text="📤 EXTRACT")  
        notebook.add(detect_frame, text="🔍 DETECT")
        
        # Status bar
        self._create_status_bar()

    def _create_menu(self):
        menubar = tk.Menu(self, bg=self.colors['card_bg'], fg=self.colors['text'])
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['card_bg'], fg=self.colors['text'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self._pick_host)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['card_bg'], fg=self.colors['text'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_embed_tab(self):
        frame = tk.Frame(self, bg=self.colors['bg'])
        
        # Main content area
        content_frame = tk.Frame(frame, bg=self.colors['bg'])
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Left panel - Host Media
        left_panel = self._create_card_frame(content_frame, "Host Media")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.host_var = tk.StringVar()
        self._create_file_input(left_panel, "Select host image or video", self.host_var, self._pick_host)
        self.host_preview = self._create_preview_panel(left_panel)
        
        # Right panel - Watermark
        right_panel = self._create_card_frame(content_frame, "Watermark")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.wm_var = tk.StringVar()
        self._create_file_input(right_panel, "Select watermark image", self.wm_var, self._pick_wm)
        self.wm_preview = self._create_preview_panel(right_panel)
        
        # Settings sidebar
        settings_frame = self._create_settings_panel(frame)
        settings_frame.pack(side="right", fill="y", padx=(10, 0))
        
        # Action button
        self._create_action_button(frame, "🔧 EMBED WATERMARK", self._embed, self.colors['accent'])
        
        return frame

    def _create_extract_tab(self):
        frame = tk.Frame(self, bg=self.colors['bg'])
        
        content_frame = tk.Frame(frame, bg=self.colors['bg'])
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Input panel
        input_panel = self._create_card_frame(content_frame, "Watermarked Media")
        input_panel.pack(fill="both", expand=True, pady=(0, 10))
        
        self.wm_image_var = tk.StringVar()
        self.meta_extract_var = tk.StringVar(value="metadata.npz")
        self.extracted_var = tk.StringVar(value="extracted.png")
        
        self._create_file_input(input_panel, "Select watermarked media", self.wm_image_var, self._pick_wm_image)
        self._create_file_input(input_panel, "Select metadata file", self.meta_extract_var, self._pick_meta_extract)
        
        # Preview panel
        self.extract_preview = self._create_preview_panel(input_panel)
        
        # Action button
        self._create_action_button(frame, "📤 EXTRACT WATERMARK", self._extract, self.colors['success'])
        
        return frame

    def _create_detect_tab(self):
        frame = tk.Frame(self, bg=self.colors['bg'])
        
        content_frame = tk.Frame(frame, bg=self.colors['bg'])
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Input panel
        input_panel = self._create_card_frame(content_frame, "Detection Analysis")
        input_panel.pack(fill="both", expand=True)
        
        self.detect_var = tk.StringVar()
        self._create_file_input(input_panel, "Select image to analyze", self.detect_var, self._pick_detect)
        
        # Results panel
        results_panel = self._create_card_frame(content_frame, "Detection Results")
        results_panel.pack(fill="both", expand=True, pady=(10, 0))
        
        self.result_label = tk.Label(results_panel, text="No analysis performed", 
                                   bg=self.colors['card_bg'], fg=self.colors['text_secondary'],
                                   font=('Segoe UI', 12))
        self.result_label.pack(pady=20)
        
        # Action button
        self._create_action_button(frame, "🔍 DETECT WATERMARK", self._detect, self.colors['error'])
        
        return frame

    def _create_card_frame(self, parent, title):
        card = tk.Frame(parent, bg=self.colors['card_bg'], relief='flat', bd=1)
        
        title_label = tk.Label(card, text=title, bg=self.colors['card_bg'], 
                              fg=self.colors['text'], font=('Segoe UI', 12, 'bold'))
        title_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        return card

    def _create_file_input(self, parent, placeholder, var, command):
        input_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        input_frame.pack(fill="x", padx=15, pady=5)
        
        btn = tk.Button(input_frame, text="📁 Browse", command=command,
                       bg=self.colors['accent'], fg='white', font=('Segoe UI', 9),
                       relief='flat', padx=15, pady=8)
        btn.pack(side="left")
        ModernTooltip(btn, "Click to select file")
        
        entry = tk.Entry(input_frame, textvariable=var, font=('Segoe UI', 10),
                        bg='#3a3a3a', fg=self.colors['text'], relief='flat', bd=5)
        entry.pack(side="right", fill="x", expand=True, padx=(10, 0))

    def _create_preview_panel(self, parent):
        preview_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        preview_frame.pack(fill="both", expand=True, padx=15, pady=(10, 15))
        
        canvas = tk.Canvas(preview_frame, bg='#3a3a3a', height=200, relief='flat')
        canvas.pack(fill="both", expand=True)
        
        placeholder_label = tk.Label(canvas, text="Preview will appear here", 
                                   bg='#3a3a3a', fg=self.colors['text_secondary'])
        canvas.create_window(150, 100, window=placeholder_label)
        
        return canvas

    def _create_settings_panel(self, parent):
        settings = self._create_card_frame(parent, "Settings")
        settings.configure(width=250)
        
        # Watermark strength
        strength_frame = tk.Frame(settings, bg=self.colors['card_bg'])
        strength_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Label(strength_frame, text="Watermark Strength", bg=self.colors['card_bg'], 
                fg=self.colors['text'], font=('Segoe UI', 10)).pack(anchor="w")
        
        self.strength_var = tk.DoubleVar(value=0.5)
        strength_scale = tk.Scale(strength_frame, from_=0.1, to=1.0, resolution=0.1,
                                orient="horizontal", variable=self.strength_var,
                                bg=self.colors['card_bg'], fg=self.colors['text'],
                                highlightthickness=0, troughcolor='#3a3a3a')
        strength_scale.pack(fill="x", pady=5)
        
        # Quality settings
        quality_frame = tk.Frame(settings, bg=self.colors['card_bg'])
        quality_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Label(quality_frame, text="Quality Options", bg=self.colors['card_bg'], 
                fg=self.colors['text'], font=('Segoe UI', 10)).pack(anchor="w")
        
        self.preserve_quality = tk.BooleanVar(value=True)
        tk.Checkbutton(quality_frame, text="Preserve original quality", 
                      variable=self.preserve_quality, bg=self.colors['card_bg'], 
                      fg=self.colors['text'], selectcolor='#3a3a3a').pack(anchor="w", pady=2)
        
        return settings

    def _create_action_button(self, parent, text, command, color):
        button_frame = tk.Frame(parent, bg=self.colors['bg'])
        button_frame.pack(side="bottom", pady=20)
        
        btn = tk.Button(button_frame, text=text, command=command,
                       bg=color, fg='white', font=('Segoe UI', 12, 'bold'),
                       relief='flat', padx=30, pady=15, cursor='hand2')
        btn.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate', length=300)
        self.progress.pack(pady=10)

    def _create_status_bar(self):
        status_frame = tk.Frame(self, bg=self.colors['card_bg'], height=30)
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready", bg=self.colors['card_bg'], 
                                   fg=self.colors['text_secondary'], font=('Segoe UI', 9))
        self.status_label.pack(side="left", padx=10, pady=5)

    def _show_about(self):
        messagebox.showinfo("About", "DCT-SVD Watermarking Tool\nModern UI with Material Design\nVersion 2.0")

    def _update_status(self, message):
        self.status_label.config(text=message)
        self.update_idletasks()

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video format"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return Path(file_path).suffix.lower() in video_extensions

    def _update_preview(self, canvas, file_path):
        """Update preview canvas with image/video thumbnail"""
        try:
            if os.path.exists(file_path):
                # Clear canvas
                canvas.delete("all")
                
                if self._is_video_file(file_path):
                    # Video preview - extract first frame
                    cap = cv2.VideoCapture(file_path)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        img.thumbnail((300, 200), Image.Resampling.LANCZOS)
                        
                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(img)
                        
                        # Center image on canvas
                        canvas_width = canvas.winfo_width() or 300
                        canvas_height = canvas.winfo_height() or 200
                        x = canvas_width // 2
                        y = canvas_height // 2
                        
                        canvas.create_image(x, y, image=photo, anchor="center")
                        canvas.image = photo  # Keep reference
                        
                        # Add video icon overlay
                        canvas.create_text(x, y + 70, text="🎥 VIDEO", 
                                         fill=self.colors['accent'], font=('Segoe UI', 10, 'bold'))
                        
                        # Add video info
                        try:
                            info = get_video_info(file_path)
                            info_text = f"{info['width']}x{info['height']} • {info['fps']:.1f}fps • {info['duration_seconds']:.1f}s"
                            canvas.create_text(x, y + 90, text=info_text, 
                                             fill=self.colors['text_secondary'], font=('Segoe UI', 8))
                        except:
                            pass
                    else:
                        canvas.create_text(150, 100, text="Could not load video preview", 
                                         fill=self.colors['text_secondary'])
                else:
                    # Image preview
                    img = Image.open(file_path)
                    img.thumbnail((300, 200), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    # Center image on canvas
                    canvas_width = canvas.winfo_width() or 300
                    canvas_height = canvas.winfo_height() or 200
                    x = canvas_width // 2
                    y = canvas_height // 2
                    
                    canvas.create_image(x, y, image=photo, anchor="center")
                    canvas.image = photo  # Keep reference
                    
                    # Add image icon
                    canvas.create_text(x, y + 70, text="📷 IMAGE", 
                                     fill=self.colors['success'], font=('Segoe UI', 10, 'bold'))
                
        except Exception as e:
            # Show error in canvas
            canvas.delete("all")
            canvas.create_text(150, 100, text=f"Preview error: {str(e)[:30]}...", 
                             fill=self.colors['text_secondary'])

    # Helper methods for file dialogs
    def _pick_host(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Host Media",
            filetypes=[
                ("Media files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.host_var.set(path)
            self._update_status(f"Host media selected: {os.path.basename(path)}")
            if hasattr(self, 'host_preview'):
                self.after(100, lambda: self._update_preview(self.host_preview, path))

    def _pick_wm(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Watermark Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.wm_var.set(path)
            self._update_status(f"Watermark selected: {os.path.basename(path)}")
            if hasattr(self, 'wm_preview'):
                self.after(100, lambda: self._update_preview(self.wm_preview, path))

    def _embed(self) -> None:
        if not self.host_var.get() or not self.wm_var.get():
            self._show_error("Please select both host media and watermark image")
            return
        
        host_path = self.host_var.get()
        watermark_path = self.wm_var.get()
        
        # Check if host is video or image
        is_video = self._is_video_file(host_path)
        
        self._update_status("Embedding watermark...")
        self.progress.start(10)
        
        try:
            timestamp = str(int(time.time()))
            
            if is_video:
                # Video watermarking
                output_path = f"watermarked_video_{timestamp}.mp4"
                meta_path = f"metadata_video_{timestamp}.npz"
                
                # Get watermark strength from UI
                alpha = self.strength_var.get() if hasattr(self, 'strength_var') else 0.05
                
                embed_watermark_video(
                    host_path,
                    watermark_path, 
                    output_path,
                    meta_path,
                    alpha=alpha,
                    frame_interval=10  # Watermark every 10th frame
                )
                
                self.progress.stop()
                self._update_status("Video watermark embedded successfully!")
                self._show_success(f"Watermarked video saved as:\n{output_path}\n\nMetadata: {meta_path}")
            else:
                # Image watermarking
                output_path = f"watermarked_{timestamp}.png"
                meta_path = f"metadata_{timestamp}.npz"
                
                embed_watermark(
                    host_path,
                    watermark_path,
                    output_path, 
                    meta_path
                )
                
                self.progress.stop()
                self._update_status("Image watermark embedded successfully!")
                self._show_success(f"Watermarked image saved as:\n{output_path}")
                
        except Exception as exc:
            self.progress.stop()
            self._update_status("Embedding failed")
            # Print detailed error to console for debugging
            import traceback
            print(f"Embedding error: {exc}")
            print("Full traceback:")
            traceback.print_exc()
            self._show_error(f"Embedding failed: {str(exc)}")

    def _pick_wm_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Watermarked Media",
            filetypes=[
                ("Media files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.wm_image_var.set(path)
            self._update_status(f"Watermarked media selected: {os.path.basename(path)}")
            if hasattr(self, 'extract_preview'):
                self.after(100, lambda: self._update_preview(self.extract_preview, path))

    def _pick_meta_extract(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Metadata File",
            filetypes=[("Metadata files", "*.npz"), ("All files", "*.*")]
        )
        if path:
            self.meta_extract_var.set(path)
            self._update_status(f"Metadata file selected: {os.path.basename(path)}")

    def _extract(self) -> None:
        if not self.wm_image_var.get() or not self.meta_extract_var.get():
            self._show_error("Please select watermarked media and metadata file")
            return
        
        watermarked_path = self.wm_image_var.get()
        metadata_path = self.meta_extract_var.get()
        
        # Check if watermarked media is video or image
        is_video = self._is_video_file(watermarked_path)
        
        self._update_status("Extracting watermark...")
        self.progress.start(10)
        
        try:
            timestamp = str(int(time.time()))
            extracted_path = f"extracted_{timestamp}.png"
            
            if is_video:
                # Extract from video
                extract_watermark_video(
                    watermarked_path,
                    metadata_path,
                    extracted_path
                )
                
                self.progress.stop()
                self._update_status("Watermark extracted from video!")
                self._show_success(f"Extracted watermark saved as:\n{extracted_path}")
            else:
                # Extract from image
                extract_watermark(
                    watermarked_path,
                    metadata_path,
                    extracted_path
                )
                
                self.progress.stop()
                self._update_status("Watermark extracted from image!")
                self._show_success(f"Extracted watermark saved as:\n{extracted_path}")
            
        except Exception as exc:
            self.progress.stop()
            self._update_status("Extraction failed")
            # Print detailed error to console for debugging
            import traceback
            print(f"Extraction error: {exc}")
            print("Full traceback:")
            traceback.print_exc()
            self._show_error(f"Extraction failed: {str(exc)}")

    def _pick_detect(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Media to Analyze", 
            filetypes=[
                ("Media files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.detect_var.set(path)
            self._update_status(f"Analysis image selected: {os.path.basename(path)}")

    def _detect(self) -> None:
        if not self.detect_var.get():
            self._show_error("Please select media to analyze")
            return
        
        media_path = self.detect_var.get()
        is_video = self._is_video_file(media_path)
        
        self._update_status("Analyzing media for watermark...")
        self.progress.start(10)
        
        try:
            if is_video:
                # Video watermark detection
                results = detect_watermark_video(media_path, frame_sample_rate=30)
                
                self.progress.stop()
                
                if 'error' in results:
                    result_text = f"❌ DETECTION FAILED\n{results['error']}"
                    self.result_label.config(text=result_text, fg=self.colors['error'])
                    self._update_status("Video analysis failed")
                else:
                    likelihood = results['watermark_likelihood']
                    frames_analyzed = results['total_frames_analyzed']
                    
                    if likelihood > 0.6:  # Threshold for watermark detection
                        result_text = f"🎥 VIDEO WATERMARK LIKELY\nLikelihood: {likelihood:.1%}\nFrames analyzed: {frames_analyzed}"
                        self.result_label.config(text=result_text, fg=self.colors['success'])
                        self._update_status("Video watermark likely detected!")
                    else:
                        result_text = f"❌ NO VIDEO WATERMARK\nLikelihood: {likelihood:.1%}\nFrames analyzed: {frames_analyzed}"
                        self.result_label.config(text=result_text, fg=self.colors['error'])  
                        self._update_status("No video watermark detected")
            else:
                # Image watermark detection using AI
                s = singular_values(media_path)
                pred = self.detector.predict(s.reshape(1, -1))[0]
                confidence = self.detector.predict_proba(s.reshape(1, -1))[0]
                
                self.progress.stop()
                
                if pred == 1:
                    result_text = f"📷 IMAGE WATERMARK DETECTED\nConfidence: {confidence[1]:.2%}"
                    self.result_label.config(text=result_text, fg=self.colors['success'])
                    self._update_status("Image watermark detected!")
                else:
                    result_text = f"❌ NO IMAGE WATERMARK\nConfidence: {confidence[0]:.2%}"
                    self.result_label.config(text=result_text, fg=self.colors['error'])
                    self._update_status("No image watermark detected")
                
        except Exception as exc:
            self.progress.stop()
            self._update_status("Detection failed")
            self.result_label.config(text=f"❌ DETECTION FAILED\n{str(exc)[:50]}...", 
                                   fg=self.colors['error'])

    def _show_success(self, message):
        """Show success message with modern styling"""
        win = tk.Toplevel(self)
        win.title("Success")
        win.geometry("400x150")
        win.configure(bg=self.colors['card_bg'])
        win.resizable(False, False)
        
        # Center on parent
        win.transient(self)
        win.grab_set()
        
        tk.Label(win, text="✅ Success", bg=self.colors['card_bg'], 
                fg=self.colors['success'], font=('Segoe UI', 14, 'bold')).pack(pady=15)
        
        tk.Label(win, text=message, bg=self.colors['card_bg'], 
                fg=self.colors['text'], font=('Segoe UI', 10), wraplength=350).pack(pady=10)
        
        tk.Button(win, text="OK", command=win.destroy, bg=self.colors['success'], 
                 fg='white', font=('Segoe UI', 10), relief='flat', padx=20).pack(pady=15)

    def _show_error(self, message):
        """Show error message with modern styling"""
        win = tk.Toplevel(self)
        win.title("Error")
        win.geometry("400x150")
        win.configure(bg=self.colors['card_bg'])
        win.resizable(False, False)
        
        # Center on parent
        win.transient(self)
        win.grab_set()
        
        tk.Label(win, text="❌ Error", bg=self.colors['card_bg'], 
                fg=self.colors['error'], font=('Segoe UI', 14, 'bold')).pack(pady=15)
        
        tk.Label(win, text=message, bg=self.colors['card_bg'], 
                fg=self.colors['text'], font=('Segoe UI', 10), wraplength=350).pack(pady=10)
        
        tk.Button(win, text="OK", command=win.destroy, bg=self.colors['error'], 
                 fg='white', font=('Segoe UI', 10), relief='flat', padx=20).pack(pady=15)


if __name__ == "__main__":
    app = WatermarkApp()
    app.mainloop()
