"""Procedural lunar surface renderer shared by the synthetic dataset and the
fallback basemap stage.

The goal is not photo-realism for its own sake: boulders must be learnable by a
small YOLOv8n model, so each boulder gets a bright, textured body, a top-left
sun highlight and a cast shadow to the bottom-right.  Craters and micro-rocks
act as distractors.  Everything is numpy/OpenCV so it runs in seconds on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Boulder:
    x1: int
    y1: int
    x2: int
    y2: int
    diameter_px: float
    cx: float
    cy: float


@dataclass
class Scene:
    image: np.ndarray  # float32, 0..1, grayscale
    boulders: list[Boulder] = field(default_factory=list)


def _smooth_noise(h: int, w: int, rng: np.random.Generator, scale: int) -> np.ndarray:
    """Low-frequency random field resized up to (h, w)."""
    sh = max(2, int(round(h / scale)) + 1)
    sw = max(2, int(round(w / scale)) + 1)
    small = rng.normal(0.0, 1.0, (sh, sw)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return field


def fbm(h: int, w: int, rng: np.random.Generator, octaves: int = 5,
        base_scale: int = 96, persistence: float = 0.52) -> np.ndarray:
    """Fractal noise normalised to zero mean / unit std."""
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    scale = base_scale
    for _ in range(octaves):
        out += amp * _smooth_noise(h, w, rng, scale)
        amp *= persistence
        scale = max(2, scale // 2)
    out -= out.mean()
    std = float(out.std()) or 1.0
    return out / std


def _sun_vector(rng: np.random.Generator) -> tuple[float, float, float]:
    """Return a unit light vector pointing *towards* the sun (image coords)."""
    azimuth = math.radians(rng.uniform(210.0, 240.0))  # NW-ish: light from upper-left
    elevation = math.radians(rng.uniform(38.0, 62.0))
    # Image y grows downwards, so a NW sun has lx<0, ly<0.
    lx = math.cos(elevation) * math.cos(azimuth)
    ly = math.cos(elevation) * math.sin(azimuth)
    lz = math.sin(elevation)
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    return lx / n, ly / n, lz / n


def add_crater(img: np.ndarray, cx: float, cy: float, radius: float,
               rng: np.random.Generator, light: tuple[float, float, float],
               strength: float = 1.0) -> None:
    h, w = img.shape
    pad = int(radius * 1.35) + 4
    x0, x1 = max(0, int(cx - pad)), min(w, int(cx + pad))
    y0, y1 = max(0, int(cy - pad)), min(h, int(cy + pad))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    rn = r / max(radius, 1.0)
    inside = rn < 1.0

    # Bowl normal and Lambert shading.
    depth = radius * rng.uniform(0.12, 0.22)
    dzdx = -2.0 * depth * dx / (radius * radius)
    dzdy = -2.0 * depth * dy / (radius * radius)
    nx, ny, nz = -dzdx, -dzdy, np.ones_like(dzdx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = (nx * light[0] + ny * light[1] + nz * light[2]) * inv
    shade = np.clip(shade, 0.0, 1.2)

    patch = img[y0:y1, x0:x1]
    albedo = rng.uniform(0.92, 1.06)
    # Bowls are slightly darker inside; the anti-sun wall falls into shadow.
    relief = (shade - 0.72) * rng.uniform(0.10, 0.20) * strength
    patch[inside] += relief[inside] * albedo

    # Fresh bright rim, strongest on the sun-facing side.
    rim = np.exp(-((rn - 1.0) / 0.10) ** 2)
    sun_side = np.clip((-(dx * light[0] + dy * light[1]) / np.maximum(r, 1.0)), -1, 1)
    patch += (rim * (0.035 + 0.055 * sun_side) * strength).astype(np.float32)

    # A little ejecta texture outside the rim.
    ejecta = np.exp(-((rn - 1.35) / 0.45) ** 2) * (rng.random() > 0.55)
    patch += ejecta * rng.normal(0, 0.012, patch.shape).astype(np.float32)


def _boulder_mask(diameter: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Irregular boulder silhouette on a square patch."""
    pad = int(diameter * 1.6) + 8
    size = pad * 2 + 1
    mask = np.zeros((size, size), np.uint8)
    c = size // 2
    if diameter < 6.0:
        axes = (max(1, int(diameter / 2)), max(1, int(diameter / 2.4)))
        cv2.ellipse(mask, (c, c), axes, rng.uniform(0, 180), 0, 360, 255, -1)
    else:
        n = int(rng.integers(8, 13))
        angles = np.sort(rng.uniform(0, 2 * math.pi, n))
        radii = diameter / 2.0 * rng.uniform(0.66, 1.0, n)
        # A few angular boulders, most rounded.
        sharp = rng.random() < 0.25
        if sharp:
            radii = diameter / 2.0 * rng.uniform(0.55, 1.0, n)
        pts = np.stack([c + radii * np.cos(angles), c + radii * np.sin(angles)], axis=1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask, c


def add_boulder(img: np.ndarray, cx: float, cy: float, diameter: float,
                rng: np.random.Generator, light: tuple[float, float, float],
                labelled: bool = True) -> Boulder | None:
    h, w = img.shape
    mask, c = _boulder_mask(diameter, rng)
    pad = c
    x0, y0 = int(round(cx)) - pad, int(round(cy)) - pad
    x1, y1 = x0 + mask.shape[1], y0 + mask.shape[0]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    body = mask > 0
    body_f = body.astype(np.float32)

    # Contact shadow and cast shadow to the lower-right.
    offset = max(1.0, diameter * rng.uniform(0.28, 0.45))
    k = max(3, int(offset * 2) | 1)
    shadow = cv2.GaussianBlur(body_f, (k, k), max(0.7, offset * 0.45))
    sh = np.zeros_like(shadow)
    # Light vector points to the sun, so the shadow goes the other way.
    dx = int(round(-offset * math.cos(math.atan2(light[1], light[0]))))
    dy = int(round(-offset * math.sin(math.atan2(light[1], light[0]))))
    if dx < 0:
        sh[:, :dx] = shadow[:, -dx:]
    elif dx > 0:
        sh[:, dx:] = shadow[:, :-dx]
    else:
        sh = shadow
    if dy < 0:
        sh[:dy, :] = sh[-dy:, :]
    elif dy > 0:
        sh[dy:, :] = sh[:-dy, :]
    else:
        pass
    sh = cv2.GaussianBlur(sh, (k, k), max(0.7, offset * 0.4))

    patch = img[y0:y1, x0:x1]
    patch -= sh * rng.uniform(0.42, 0.62)

    # Body albedo and top-left illumination gradient.
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(np.float32)
    nx = (xx - c) / max(diameter * 0.5, 1.0)
    ny = (yy - c) / max(diameter * 0.5, 1.0)
    dot = nx * light[0] + ny * light[1]
    lambert = np.clip(0.66 + 0.42 * dot, 0.22, 1.18)

    base = rng.uniform(0.44, 0.66)
    tex = cv2.GaussianBlur(rng.normal(0, 1, mask.shape).astype(np.float32),
                           (3, 3), 0) * 0.07
    body_value = (base + tex) * lambert
    patch[body] = np.clip(patch[body] * 0.18 + body_value[body], 0, 1.2)

    # Sunlit rim on the upper-left, dark contact edge on the lower-right.
    er = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    rim = (body & (er == 0)).astype(np.float32)
    rim = cv2.GaussianBlur(rim, (3, 3), 0.6)
    rim_w = np.clip(0.42 + 0.8 * dot, 0.0, 1.0)
    patch += (rim * rim_w * rng.uniform(0.07, 0.18)).astype(np.float32)

    body_ys, body_xs = np.nonzero(body)
    if len(body_xs) == 0:
        return None
    bx1, bx2 = int(body_xs.min()) + x0, int(body_xs.max()) + x0
    by1, by2 = int(body_ys.min()) + y0, int(body_ys.max()) + y0
    if not labelled:
        return None
    return Boulder(bx1, by1, bx2, by2, float(diameter), cx, cy)


def add_micro_rocks(img: np.ndarray, rng: np.random.Generator, count: int,
                    light: tuple[float, float, float]) -> None:
    h, w = img.shape
    for _ in range(count):
        cx, cy = rng.uniform(2, w - 2), rng.uniform(2, h - 2)
        r = rng.uniform(0.6, 2.4)
        p = max(2, int(r * 3))
        x0, x1 = int(cx) - p, int(cx) + p + 1
        y0, y1 = int(cy) - p, int(cy) + p + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        patch = img[y0:y1, x0:x1]
        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]].astype(np.float32)
        d = np.sqrt((xx - p) ** 2 + (yy - p) ** 2)
        body = d <= r
        shadow = (d > r) & (d < r * 2.3) & (xx > p + r * 0.4) & (yy > p + r * 0.4)
        patch[body] = np.clip(patch[body] + rng.uniform(0.05, 0.22), 0, 1.2)
        patch[shadow] -= 0.05


def render_scene(
    size: int,
    rng: np.random.Generator,
    *,
    n_boulders: int = 12,
    boulder_size_px: tuple[float, float] = (3.0, 40.0),
    n_craters: int = 2,
    n_micro_rocks: int = 12,
    boulder_size_bias: float = 3.0,
) -> Scene:
    """Render one square grayscale lunar scene in [0, 1]."""
    light = _sun_vector(rng)

    # --- base regolith -----------------------------------------------------
    base = 0.49 + 0.042 * fbm(size, size, rng, octaves=6, base_scale=max(32, size // 5))
    base += 0.012 * fbm(size, size, rng, octaves=3, base_scale=5)
    # Very broad albedo patches (mare / highland units).
    base += 0.022 * cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32),
                                     (0, 0), size / 4.0)
    img = base.astype(np.float32)

    # --- craters -----------------------------------------------------------
    for _ in range(n_craters):
        r = rng.uniform(7, size / 7.5)
        cx, cy = rng.uniform(r * 1.2, size - r * 1.2, 2)
        add_crater(img, cx, cy, r, rng, light, strength=rng.uniform(0.6, 1.2))

    # --- boulders ----------------------------------------------------------
    boulders: list[Boulder] = []
    attempts = 0
    lo, hi = boulder_size_px
    # A power-law-ish size mix: many small rocks, a few large ones.  Guarantee
    # a couple of >1.5 m boulders when the scene has enough objects.
    forced_large = 2 if n_boulders >= 8 else (1 if n_boulders >= 3 else 0)
    while len(boulders) < n_boulders and attempts < n_boulders * 12:
        attempts += 1
        if len(boulders) < forced_large:
            d = rng.uniform(max(lo, 14.0), hi)
        else:
            u = rng.random() ** boulder_size_bias
            d = float(lo + (hi - lo) * u)
        # Avoid the outer 8% so boxes never clip at the tile edge.
        margin = d / 2 + 3
        cx = rng.uniform(margin, size - margin)
        cy = rng.uniform(margin, size - margin)
        b = add_boulder(img, cx, cy, d, rng, light)
        if b is None:
            continue
        boulders.append(b)

    # --- micro-rocks and texture ------------------------------------------
    add_micro_rocks(img, rng, n_micro_rocks, light)
    img += rng.normal(0, rng.uniform(0.006, 0.018), (size, size)).astype(np.float32)

    # Vignetting + gentle contrast stretch.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * size / 2)
    img *= 1.0 - 0.16 * rr
    lo_p, hi_p = np.percentile(img, [1.0, 99.0])
    img = (img - lo_p) / max(hi_p - lo_p, 1e-6)
    img = np.clip(img, 0, 1) ** rng.uniform(0.88, 1.12)

    return Scene(image=img.astype(np.float32), boulders=boulders)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)
