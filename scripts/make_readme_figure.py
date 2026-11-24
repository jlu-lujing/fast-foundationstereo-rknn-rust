#!/usr/bin/env python3
"""Render a README comparison from a real rectified pair and raw disparity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = np.asarray(
    [
        [22, 20, 61],
        [33, 68, 134],
        [31, 145, 169],
        [99, 194, 107],
        [238, 216, 80],
        [239, 111, 59],
        [162, 26, 65],
    ],
    dtype=np.float32,
)
BG = (12, 15, 20)
PANEL = (21, 25, 33)
TEXT = (239, 242, 247)
MUTED = (163, 173, 188)
ACCENT = (77, 205, 177)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("Arial Bold.ttf", "Arial.ttf") if bold else ("Arial.ttf",)
    paths = [Path("/System/Library/Fonts/Supplemental") / name for name in names]
    paths.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def gradient(values: np.ndarray) -> np.ndarray:
    position = np.clip(values, 0.0, 1.0) * (len(COLORS) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(COLORS) - 1)
    fraction = (position - lower)[..., None]
    return COLORS[lower] * (1.0 - fraction) + COLORS[upper] * fraction


def overlap_mask(disparity: np.ndarray) -> np.ndarray:
    height, width = disparity.shape
    valid = np.isfinite(disparity) & (disparity > 0)
    x_right = np.arange(width, dtype=np.float32)[None, :] - disparity
    common_fov = valid & (x_right >= 0) & (x_right < width)
    rounded_x = np.rint(x_right).astype(np.int64)
    rows, columns = np.nonzero(common_fov)
    competing = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(competing, (rows, rounded_x[rows, columns]), disparity[rows, columns])
    sampled = competing[
        np.arange(height, dtype=np.int64)[:, None],
        np.clip(rounded_x, 0, width - 1),
    ]
    return common_fov & (disparity + 1.0 >= sampled)


def panel(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    width, height = 600, 300
    canvas = Image.new("RGB", (width, height + 52), PANEL)
    canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (0, 52))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 7), title, font=font(18, True), fill=TEXT)
    draw.text((14, 30), subtitle, font=font(13), fill=MUTED)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    with np.load(args.inputs) as values:
        left = Image.fromarray(
            np.clip(values["left"][0].transpose(1, 2, 0), 0, 255).astype(np.uint8),
            mode="RGB",
        )
    reference = np.load(args.reference).astype(np.float32).squeeze()
    prediction = np.fromfile(args.prediction, dtype="<f4")
    if prediction.size != args.height * args.width:
        raise ValueError(
            f"expected {args.height * args.width} disparity values, got {prediction.size}"
        )
    prediction = prediction.reshape(args.height, args.width)
    if reference.shape != prediction.shape:
        raise ValueError(f"reference shape {reference.shape} != prediction {prediction.shape}")
    visible = overlap_mask(reference)
    valid = visible & np.isfinite(reference) & np.isfinite(prediction)
    error = np.abs(prediction - reference)
    low, high = np.percentile(reference[valid], [2, 98])

    def disparity_image(disparity: np.ndarray) -> Image.Image:
        normalized = (disparity - low) / max(float(high - low), 1e-6)
        color = gradient(normalized).astype(np.uint8)
        color[~visible] = 0
        return Image.fromarray(color, mode="RGB")

    error_color = gradient(np.clip(error / 2.0, 0.0, 1.0)).astype(np.uint8)
    error_color[~valid] = 0

    panels = [
        panel(left, "NVIDIA project input", "original rectified left image"),
        panel(disparity_image(reference), "NVIDIA FP32", "original project reference"),
        panel(disparity_image(prediction), "RK3588 RKNN", "single graph / FP16"),
        panel(
            Image.fromarray(error_color, mode="RGB"),
            "Absolute difference",
            "clipped at 2 px / binocular-valid pixels only",
        ),
    ]
    canvas = Image.new("RGB", (1200, 758), BG)
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % 2) * 600, 4 + (index // 2) * 352))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 704),
        "Fast-FoundationStereo on RK3588 · comparison with NVIDIA FP32",
        font=font(18, True),
        fill=TEXT,
    )
    draw.text(
        (16, 728),
        f"Binocular-valid pixels: {valid.sum():,}   MAE: {error[valid].mean():.3f} px   "
        f"P95: {np.percentile(error[valid], 95):.3f} px",
        font=font(13),
        fill=ACCENT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
