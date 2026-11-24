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
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--disparity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    left = Image.open(args.left).convert("RGB")
    disparity = np.fromfile(args.disparity, dtype="<f4")
    if disparity.size != args.height * args.width:
        raise ValueError(
            f"expected {args.height * args.width} disparity values, got {disparity.size}"
        )
    disparity = disparity.reshape(args.height, args.width)
    visible = overlap_mask(disparity)
    low, high = np.percentile(disparity[visible], [2, 98])
    normalized = (disparity - low) / max(float(high - low), 1e-6)
    color = gradient(normalized).astype(np.uint8)
    color[~visible] = 0
    color_image = Image.fromarray(color, mode="RGB")

    left_array = np.asarray(left.resize(color_image.size, Image.Resampling.LANCZOS), dtype=np.float32)
    overlay = left_array.copy()
    overlay[visible] = left_array[visible] * 0.38 + color[visible].astype(np.float32) * 0.62
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")

    panels = [
        panel(left, "Rectified left", "original camera image"),
        panel(color_image, "RK3588 disparity", f"single RKNN graph · {visible.sum():,} overlap px"),
        panel(overlay_image, "Depth overlay", "only binocular-overlap pixels shown"),
    ]
    canvas = Image.new("RGB", (1800, 408), BG)
    for index, item in enumerate(panels):
        canvas.paste(item, (index * 600, 4))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 365), "Fast-FoundationStereo · real rectified pair · 192x384 model", font=font(18, True), fill=TEXT)
    draw.text((16, 388), "All three RK3588 NPU cores · full-hardware CPU frequencies", font=font(13), fill=ACCENT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
