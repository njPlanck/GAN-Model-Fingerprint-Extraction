#!/usr/bin/env python3
"""
compare_fourier.py

Usage:
    python compare_fourier.py path/to/original.png path/to/cleaned.png [-o OUTPUT] [--mode gray|color]

Description:
    Takes two images (original and cleaned) and produces a 2x2 grid:
        Top row:    Original, Cleaned
        Bottom row: |FFT(Original)|, |FFT(Cleaned)|  (log-magnitude, centered)
    The grid is saved to the current working directory.
    Also prints the SSIM between the two images.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim


def load_image(path: Path, mode: str = "gray") -> np.ndarray:
    """
    Load an image as numpy array.
    If mode == 'gray', returns float32 grayscale in [0,1].
    If mode == 'color', returns float32 RGB in [0,1].
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if mode == "gray":
        # Convert to luminance (Rec. 601)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return gray
    elif mode == "color":
        return arr
    else:
        raise ValueError("mode must be 'gray' or 'color'")


def fft_magnitude_gray(img_gray: np.ndarray) -> np.ndarray:
    """Compute centered log-magnitude FFT for grayscale image."""
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    mag -= mag.min()
    if mag.max() > 0:
        mag /= mag.max()
    return (mag * 255).astype(np.uint8)


def fft_magnitude_color(img_rgb: np.ndarray) -> np.ndarray:
    """Compute per-channel FFT magnitude for RGB and combine."""
    channels = []
    for c in range(3):
        f = np.fft.fft2(img_rgb[..., c])
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))
        channels.append(mag)

    mag_rgb = np.stack(channels, axis=-1)
    for c in range(3):
        m = mag_rgb[..., c]
        m -= m.min()
        if m.max() > 0:
            m /= m.max()
        mag_rgb[..., c] = m
    return (mag_rgb * 255).astype(np.uint8)


def to_pil(img: np.ndarray, mode: str = "gray") -> Image.Image:
    """Convert numpy array to PIL.Image."""
    if mode == "gray":
        return Image.fromarray(img, mode="L")
    else:
        return Image.fromarray(img, mode="RGB")


def make_grid(original_path: Path, cleaned_path: Path, mode: str = "gray", output: Path | None = None):
    # Load images
    img1 = load_image(original_path, mode=mode)
    img2 = load_image(cleaned_path, mode=mode)

    # FFT magnitudes
    mag1 = fft_magnitude_gray(img1) if mode == "gray" else fft_magnitude_color(img1)
    mag2 = fft_magnitude_gray(img2) if mode == "gray" else fft_magnitude_color(img2)

    # Convert to PIL
    img1_pil = to_pil((img1 * 255).astype(np.uint8) if mode == "gray" else (img1 * 255).astype(np.uint8), mode)
    img2_pil = to_pil((img2 * 255).astype(np.uint8) if mode == "gray" else (img2 * 255).astype(np.uint8), mode)
    mag1_pil = to_pil(mag1, mode)
    mag2_pil = to_pil(mag2, mode)

    # Resize all to same size
    w, h = img1_pil.size
    img2_pil = img2_pil.resize((w, h))
    mag1_pil = mag1_pil.resize((w, h))
    mag2_pil = mag2_pil.resize((w, h))

    # Padding between images
    pad = 20
    grid_w = 2 * w + pad
    grid_h = 2 * h + pad
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # Paste images
    grid.paste(img1_pil.convert("RGB"), (0, 0))
    grid.paste(img2_pil.convert("RGB"), (w + pad, 0))
    grid.paste(mag1_pil.convert("RGB"), (0, h + pad))
    grid.paste(mag2_pil.convert("RGB"), (w + pad, h + pad))

    # Add titles
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    titles = [
        f"Original ({original_path.name})",
        f"Cleaned ({cleaned_path.name})",
        "FFT |Original|",
        "FFT |Cleaned|",
    ]
    positions = [(10, 10), (w + pad + 10, 10), (10, h + pad + 10), (w + pad + 10, h + pad + 10)]
    for text, pos in zip(titles, positions):
        draw.text(pos, text, fill=(255, 0, 0), font=font)

    # Save to current working directory
    if output is None:
        output = Path.cwd() / f"fft_grid_{original_path.stem}_vs_{cleaned_path.stem}.png"
    grid.save(output)
    print(f"Saved grid to: {output}")

    # Compute SSIM (always on grayscale)
    img1_gray = (img1 * 255).astype(np.uint8) if img1.ndim == 2 else (0.299*img1[...,0] + 0.587*img1[...,1] + 0.114*img1[...,2])*255
    img2_gray = (img2 * 255).astype(np.uint8) if img2.ndim == 2 else (0.299*img2[...,0] + 0.587*img2[...,1] + 0.114*img2[...,2])*255
    img1_gray = img1_gray.astype(np.uint8)
    img2_gray = img2_gray.astype(np.uint8)

    ssim_val = ssim(img1_gray, img2_gray, data_range=255)
    print(f"SSIM between original and cleaned: {ssim_val:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two images and show their Fourier transforms.")
    parser.add_argument("original", type=str, help="Path to the original image")
    parser.add_argument("cleaned", type=str, help="Path to the cleaned/processed image")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image path (PNG/JPG)")
    parser.add_argument("--mode", type=str, choices=["gray", "color"], default="gray",
                        help="FFT mode: grayscale (default) or per-channel color")
    return parser.parse_args()


def main():
    args = parse_args()
    original_path = Path(args.original)
    cleaned_path = Path(args.cleaned)
    output_path = Path(args.output) if args.output else None
    make_grid(original_path, cleaned_path, mode=args.mode, output=output_path)


if __name__ == "__main__":
    main()
