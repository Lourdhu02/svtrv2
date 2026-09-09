"""Preprocessing (MSR fit-pad) and augmentation for scene text recognition.

The augmentation set is intentionally display-like and corruption-aware rather
than a generic image-classifier recipe.
"""
from __future__ import annotations

import random
from functools import partial
from typing import Any, Dict

import cv2
import numpy as np

from .config import MSR_BINS

_BIN_BY_NAME = {b["name"]: b for b in MSR_BINS}

_PAD_MODES = {
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
    "black": cv2.BORDER_CONSTANT,
}


# ------------------------------------------------------------------ MSR bins


def select_msr_bin(height: int, width: int) -> Dict[str, Any]:
    """Pick the MSR bin from the raw crop aspect ratio."""
    ar = width / max(height, 1)
    for b in MSR_BINS:
        if ar < b["max_ar"]:
            return b
    return MSR_BINS[-1]


def msr_timesteps(bin_info: Dict[str, Any]) -> int:
    """Read the configured value rather than recomputing width // 8, so the
    backbone's width stride and the CTC input lengths cannot drift apart."""
    return int(bin_info["timesteps"])


def require_rgb(image: np.ndarray, where: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{where}: expected a 3-channel RGB image, got shape {image.shape}. "
            "This engine is RGB-only end to end."
        )
    return image


# --------------------------------------------------------------- resize / pad


def fit_pad(image: np.ndarray, target_h: int = 64, target_w: int = 288,
            pad_mode: str = "edge", **_) -> np.ndarray:
    """Aspect-preserving resize centred on a padded canvas."""
    require_rgb(image, "fit_pad")
    h, w = image.shape[:2]
    s = min(target_h / max(h, 1), target_w / max(w, 1))
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)

    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    return cv2.copyMakeBorder(
        resized, top, target_h - nh - top, left, target_w - nw - left,
        _PAD_MODES.get(pad_mode, cv2.BORDER_REPLICATE), value=0,
    )


# ------------------------------------------------- display-specific corruption


def glare(image: np.ndarray, **_) -> np.ndarray:
    """Specular hotspot from a transparent cover or glossy surface."""
    h, w = image.shape[:2]
    out = image.astype(np.float32)

    for _ in range(random.randint(1, 2)):
        cx, cy = random.uniform(0.1, 0.9) * w, random.uniform(0.1, 0.9) * h
        # Wide and shallow: cover reflections streak along the glass, they are
        # rarely circular.
        ax = random.uniform(0.10, 0.35) * w
        ay = random.uniform(0.25, 0.90) * h
        ang = random.uniform(-25, 25)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), ang, 0, 360, 1.0, -1)
        blur_k = max(3, (int(min(ax, ay)) // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        out += (mask[..., None] * random.uniform(90, 200))

    return np.clip(out, 0, 255).astype(np.uint8)


def segment_fade(image: np.ndarray, **_) -> np.ndarray:
    """Attenuate a thin stroke toward the local background."""
    h, w = image.shape[:2]
    out = image.astype(np.float32)

    for _ in range(random.randint(1, 3)):
        horizontal = random.random() < 0.5
        if horizontal:
            bh = max(2, int(h * random.uniform(0.05, 0.12)))
            bw = max(4, int(w * random.uniform(0.03, 0.07)))
        else:
            bh = max(4, int(h * random.uniform(0.20, 0.40)))
            bw = max(2, int(w * random.uniform(0.010, 0.025)))
        y0 = random.randint(0, max(0, h - bh))
        x0 = random.randint(0, max(0, w - bw))

        patch = out[y0:y0 + bh, x0:x0 + bw]
        # Fade toward the median of the surrounding area rather than toward black.
        bg = float(np.median(out[max(0, y0 - bh):y0 + 2 * bh, max(0, x0 - bw):x0 + 2 * bw]))
        alpha = random.uniform(0.35, 0.75)
        out[y0:y0 + bh, x0:x0 + bw] = patch * alpha + bg * (1.0 - alpha)

    return np.clip(out, 0, 255).astype(np.uint8)


def backlight_bloom(image: np.ndarray, **_) -> np.ndarray:
    """Stroke bleeding where edges spread outward and thin gaps close."""
    k = random.choice([5, 7, 9, 11])
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (k, k), 0).astype(np.float32)
    strength = random.uniform(0.35, 0.75)

    panel = float(np.median(img))
    emissive = (float(img.max()) - panel) > (panel - float(img.min()))
    if emissive:
        out = img + strength * np.maximum(blurred - img, 0)
    else:
        out = img - strength * np.maximum(img - blurred, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def viewing_angle_washout(image: np.ndarray, **_) -> np.ndarray:
    """Contrast collapse across one axis, as with an oblique viewing angle."""
    h, w = image.shape[:2]
    out = image.astype(np.float32)
    lo, hi = random.uniform(0.25, 0.6), random.uniform(0.0, 0.35)
    ramp = np.linspace(lo, hi, h, dtype=np.float32)
    if random.random() < 0.5:
        ramp = ramp[::-1]
    contrast = ramp[:, None, None]
    mid = float(out.mean())
    out = (out - mid) * contrast + mid + random.uniform(-15, 15)
    return np.clip(out, 0, 255).astype(np.uint8)


def moire(image: np.ndarray, **_) -> np.ndarray:
    """Fine periodic interference from photographing a pixel/segment grid."""
    h, w = image.shape[:2]
    period = random.uniform(2.5, 6.0)
    ang = random.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = (xx * np.cos(ang) + yy * np.sin(ang)) * (2 * np.pi / period)
    pattern = np.sin(phase) * random.uniform(6, 18)
    return np.clip(image.astype(np.float32) + pattern[..., None], 0, 255).astype(np.uint8)


def _safe(make):
    """Albumentations renames arguments between majors; skip an op rather than
    crashing the whole pipeline on a version we did not pin."""
    try:
        return make()
    except Exception:
        return None


# ------------------------------------------------------------------ pipelines


def build_transforms(img_h: int, img_w: int, training: bool = True,
                     aug_level: str = "digital", pad_mode: str = "edge"):
    """Build the 3-channel pipeline.  Never flips: digit orientation is fixed."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ops = [A.Lambda(image=partial(fit_pad, target_h=img_h, target_w=img_w,
                                  pad_mode=pad_mode), name="fit_pad")]

    if training and aug_level != "none":
        # Display-specific corruptions first, on the un-normalized image.  Only
        # one display effect per sample: stacking glare on top of washout on top
        # of bloom produces images no camera would ever produce, and the model
        # wastes capacity on them.
        display = [
            _safe(lambda: A.Lambda(image=glare, name="glare", p=1.0)),
            _safe(lambda: A.Lambda(image=viewing_angle_washout, name="washout", p=1.0)),
            _safe(lambda: A.Lambda(image=backlight_bloom, name="bloom", p=1.0)),
            _safe(lambda: A.Lambda(image=segment_fade, name="segment_fade", p=1.0)),
            _safe(lambda: A.Lambda(image=moire, name="moire", p=1.0)),
        ]
        display = [d for d in display if d is not None]
        if display:
            ops.append(A.OneOf(display, p=0.65))

        # Geometry stays modest; large rotations are rarely needed for short STR
        # benchmarks and can teach invariances the model never uses.
        candidates = [
            lambda: A.Affine(scale=(0.92, 1.08), rotate=(-4, 4), shear=(-3, 3),
                             translate_percent=(-0.03, 0.03), fill=0, p=0.4),
            lambda: A.Perspective(scale=(0.02, 0.05), keep_size=True, fill=0, p=0.25),
            # Auto-exposure swing + day/night: this range is intentionally wider
            # than the analog engine's, which used +/-0.15.
            lambda: A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
            lambda: A.RandomGamma(gamma_limit=(65, 145), p=0.3),
            lambda: A.OneOf([A.MotionBlur(blur_limit=(3, 7)),
                             A.GaussianBlur(blur_limit=(3, 7)),
                             A.Defocus(radius=(1, 3))], p=0.3),
            _gauss_noise,
            lambda: A.ImageCompression(quality_range=(45, 92), p=0.25),
        ]
        if aug_level == "light":
            candidates = candidates[:4]
        ops.extend([t for t in (_safe(c) for c in candidates) if t is not None])

    ops.extend([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
    return A.Compose(ops)


def _gauss_noise():
    """Version-safe GaussNoise: albumentations >= 2.0 renamed ``var_limit``
    (absolute pixel range) to ``std_range`` (fraction of 255)."""
    import albumentations as A

    major = int(A.__version__.split(".")[0]) if getattr(A, "__version__", "2") != "unknown" else 2
    if major >= 2:
        return A.GaussNoise(std_range=(0.02, 0.10), p=0.25)
    return A.GaussNoise(var_limit=(5.0, 25.0), p=0.25)


def build_msr_transforms(training: bool = True, aug_level: str = "digital",
                         pad_mode: str = "edge") -> Dict[str, Any]:
    """One pipeline per MSR bin, keyed by bin name."""
    return {
        str(b["name"]): build_transforms(int(b["height"]), int(b["width"]),
                                         training=training, aug_level=aug_level,
                                         pad_mode=pad_mode)
        for b in MSR_BINS
    }


class MSRTransform:
    """Callable transform that picks the bin from the raw image aspect ratio."""

    def __init__(self, training: bool = False, aug_level: str = "none",
                 pad_mode: str = "edge") -> None:
        self.transforms = build_msr_transforms(training, aug_level, pad_mode)

    def __call__(self, *, image):
        require_rgb(image, "MSRTransform")
        b = select_msr_bin(image.shape[0], image.shape[1])
        return self.transforms[str(b["name"])](image=image)
