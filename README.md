# Digital Watermarking with DCT-SVD and AI

This project provides a simple example of digital watermarking for images using a
combination of Discrete Cosine Transform (DCT) and Singular Value Decomposition (SVD).

Features:

- Embed a grayscale watermark into a host image and save the watermarked result.
- Extract the watermark from a watermarked image using stored metadata.
- Simple AI component with logistic regression to detect whether an image contains a watermark.
- Graphical user interface built with `tkinter`.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

Use the tabs to embed a watermark, extract a watermark, or run the AI-based
watermark detector.
