#!/usr/bin/env python3
"""Build the README comparison from an original Fast-FoundationStereo export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = (13, 15, 20)
PANEL_BG = (20, 23, 30)
TEXT = (238, 240, 244)
MUTED = (164, 171, 184)
ACCENT = (76, 201, 176)

DISPARITY_COLORS = np.asarray(
    [
        [20, 20, 55],
        [42, 73, 139],
        [46, 148, 159],
        [103, 197, 96],
        [238, 211, 77],
        [239, 112, 61],
        [156, 24, 66],
    ],
    dtype=np.float32,
)
ERROR_COLORS = np.asarray(
    [
        [22, 27, 38],
        [35, 93, 117],
        [54, 185, 143],
        [238, 211, 77],
        [239, 112, 61],
        [174, 28, 56],
    ],
    dtype=np.float32,
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(
            "/System/Library/Fonts/Supplemental/"
            + ("Arial Bold.ttf" if bold else "Arial.ttf")
        ),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def gradient(values: np.ndarray, colors: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    position = np.clip(values, 0.0, 1.0) * (len(colors) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(colors) - 1)
    fraction = (position - lower)[..., None]
    return colors[lower] * (1.0 - fraction) + colors[upper] * fraction


def original_rgb(nchw: np.ndarray) -> Image.Image:
    image = np.asarray(nchw, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected RGB NCHW input, got {image.shape}")
    image = np.rint(image.transpose(1, 2, 0))
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def disparity_image(
    values: np.ndarray,
    maximum: float,
    visible: np.ndarray,
) -> Image.Image:
    values = np.asarray(values, dtype=np.float32).squeeze()
    normalized = np.clip(values / max(maximum, 1e-6), 0.0, 1.0)
    rgb = gradient(normalized, DISPARITY_COLORS).astype(np.uint8)
    rgb[~visible] = 0
    return Image.fromarray(rgb, mode="RGB")


def error_image(
    values: np.ndarray,
    visible: np.ndarray,
    limit: float,
) -> Image.Image:
    normalized = np.clip(np.abs(values) / max(limit, 1e-9), 0.0, 1.0)
    rgb = gradient(normalized, ERROR_COLORS).astype(np.uint8)
    rgb[~visible] = 0
    return Image.fromarray(rgb, mode="RGB")


def stereo_overlap_mask(
    disparity: np.ndarray,
    visibility_tolerance_px: float = 1.0,
) -> np.ndarray:
    """Keep pixels in the common field of view that are not occluded."""
    disparity = np.asarray(disparity, dtype=np.float32).squeeze()
    if disparity.ndim != 2:
        raise ValueError(f"disparity must be 2-D, got {disparity.shape}")

    height, width = disparity.shape
    valid = np.isfinite(disparity) & (disparity > 0)
    x_left = np.arange(width, dtype=np.float32)[None, :]
    x_right = x_left - disparity
    common_fov = valid & (x_right >= 0) & (x_right < width)
    rounded_x = np.rint(x_right).astype(np.int64)
    rows, columns = np.nonzero(common_fov)
    nearest_competing = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(
        nearest_competing,
        (rows, rounded_x[rows, columns]),
        disparity[rows, columns],
    )
    competing_at_pixel = nearest_competing[
        np.arange(height, dtype=np.int64)[:, None],
        np.clip(rounded_x, 0, width - 1),
    ]
    visible_in_right = (
        disparity + float(visibility_tolerance_px) >= competing_at_pixel
    )
    return common_fov & visible_in_right


def panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    size: tuple[int, int],
) -> Image.Image:
    width, header = size[0], 52
    canvas = Image.new("RGB", (width, size[1] + header), PANEL_BG)
    canvas.paste(image.convert("RGB").resize(size, Image.Resampling.LANCZOS), (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 7), title, font=font(17, True), fill=TEXT)
    draw.text((14, 29), subtitle, font=font(13), fill=MUTED)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rust-disparity", type=Path, required=True)
    parser.add_argument("--error-limit", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float32).squeeze()
    rust = np.fromfile(args.rust_disparity, dtype="<f4").reshape(reference.shape)
    with np.load(args.inputs) as inputs:
        left = original_rgb(inputs["left"])

    visible = stereo_overlap_mask(reference)
    visible &= np.isfinite(reference) & np.isfinite(rust)
    error = np.abs(rust - reference)
    overlap_error = error[visible]
    maximum = float(np.percentile(reference[visible], 99))
    mae = float(overlap_error.mean())
    p99 = float(np.percentile(overlap_error, 99))
    maximum_error = float(overlap_error.max())
    correlation = float(
        np.corrcoef(rust[visible].astype(np.float64), reference[visible])[0, 1]
    )

    height, width = reference.shape
    cell = (410, 205)
    panels = [
        panel(
            left,
            "Original left input",
            f"real Fast-FoundationStereo export · {width}x{height}",
            cell,
        ),
        panel(
            disparity_image(reference, maximum, visible),
            "Original FP32 pipeline",
            f"disparity 0-{maximum:.1f} px",
            cell,
        ),
        panel(
            disparity_image(rust, maximum, visible),
            "Rust + RK3588",
            f"{int(visible.sum()):,} mutually visible pixels",
            cell,
        ),
        panel(
            error_image(error, visible, args.error_limit),
            "Absolute difference",
            f"0-{args.error_limit:.2f} px, clipped",
            cell,
        ),
    ]

    panel_height = cell[1] + 52
    canvas = Image.new("RGB", (len(panels) * cell[0], 86 + panel_height + 42), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 10),
        "Fast-FoundationStereo on RK3588",
        font=font(24, True),
        fill=TEXT,
    )
    draw.text(
        (20, 47),
        f"Original stereo export · {width}x{height} · 2 refinement iterations",
        font=font(14),
        fill=MUTED,
    )
    for index, image in enumerate(panels):
        canvas.paste(image, (index * cell[0], 86))
    draw.text(
        (20, 86 + panel_height + 7),
        (
            f"Rust vs original FP32: MAE {mae:.5f} px · "
            f"P99 {p99:.5f} px · correlation {correlation:.7f}"
        ),
        font=font(14, True),
        fill=ACCENT,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "shape": [height, width],
                "overlap_pixels": int(visible.sum()),
                "mae_px": mae,
                "p99_px": p99,
                "max_px": maximum_error,
                "correlation": correlation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
