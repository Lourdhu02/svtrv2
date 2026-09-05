"""Charset, MSR canvases, SVTRv2 variants, and training defaults.

This repository is organized as a paper-first SVTRv2 implementation for scene
text recognition research and benchmark reproduction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Charset: the standard STR benchmark set -- 10 digits, 26 lowercase, 26
# uppercase, 32 punctuation marks, and space (the PARSeq / Union14M order).
# Index 0 is the CTC blank, so characters occupy 1..len(CHARSET) and
# NUM_CLASSES = len+1.  Labels containing characters outside this set are
# skipped (counted and reported) by the dataset loaders.
CHARSET: str = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    " "
)
BLANK_IDX: int = 0
NUM_CLASSES: int = len(CHARSET) + 1  # 96

# --------------------------------------------------------------- MSR canvases

# Multi-size resizing bins selected from the raw crop aspect ratio.
#
# The bucket boundaries are paper-facing defaults.  In a research run, measure
# the target benchmark distribution and adjust these to match the data.
MSR_BINS: Tuple[Dict[str, Any], ...] = (
    dict(name="short", min_ar=0.0, max_ar=1.5, height=32, width=128, feature_h=4, timesteps=16),
    dict(name="medium", min_ar=1.5, max_ar=2.5, height=32, width=192, feature_h=4, timesteps=24),
    dict(name="long", min_ar=2.5, max_ar=3.5, height=32, width=256, feature_h=4, timesteps=32),
    dict(name="xlong", min_ar=3.5, max_ar=float("inf"), height=32, width=384, feature_h=4, timesteps=48),
)

# ------------------------------------------------------------------- variants

# SVTRv2 variants following Table 8 of the paper (arXiv 2411.15858v2).
#
# `mixers` encodes the paper's [L]_m[G]_n permutation: the first m mixing blocks
# use *local* mixing (two consecutive grouped convolutions) and the last n use
# *global* mixing (MHSA).  A block is one or the other -- never both.  Heads and
# conv groups are both D_i / 32, which reproduces the paper's head counts.
#
#   name        dims             depths     heads        permutation
#   svtrv2-t    (64, 128, 256)   (3, 6, 3)  (2, 4, 8)    [L]6[G]6
#   svtrv2-s    (96, 192, 384)   (3, 6, 3)  (3, 6, 12)   [L]6[G]6
#   svtrv2-b    (128, 256, 384)  (6, 6, 6)  (4, 8, 12)   [L]8[G]10
#   svtrv2-xl   (192, 384, 512)  (6, 9, 9)  (6, 12, 16)  [L]8[G]16
#
# XL is the extrapolation: it keeps every structural rule and only scales width
# and depth.
_L, _G = "Local", "Global"

MODELS: Dict[str, Dict[str, Any]] = {
    "svtrv2-t": dict(
        dims=(64, 128, 256), depths=(3, 6, 3),
        mixers=((_L,) * 3, (_L,) * 3 + (_G,) * 3, (_G,) * 3),
        drop=0.05, drop_path=0.05,
    ),
    "svtrv2-s": dict(
        dims=(96, 192, 384), depths=(3, 6, 3),
        mixers=((_L,) * 3, (_L,) * 3 + (_G,) * 3, (_G,) * 3),
        drop=0.05, drop_path=0.08,
    ),
    "svtrv2-b": dict(
        dims=(128, 256, 384), depths=(6, 6, 6),
        mixers=((_L,) * 6, (_L,) * 2 + (_G,) * 4, (_G,) * 6),
        drop=0.08, drop_path=0.10,
    ),
    "svtrv2-xl": dict(
        dims=(192, 384, 512), depths=(6, 9, 9),
        mixers=((_L,) * 6, (_L,) * 2 + (_G,) * 7, (_G,) * 9),
        drop=0.10, drop_path=0.15,
    ),
}
VARIANTS = tuple(MODELS.keys())

# Short aliases for CLI convenience.
MODEL_ALIASES: Dict[str, str] = {
    "t": "svtrv2-t", "s": "svtrv2-s", "b": "svtrv2-b", "xl": "svtrv2-xl",
}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name.lower())


# ------------------------------------------------------------------- defaults

DEFAULTS: Dict[str, Any] = dict(
    # input
    img_h=32,
    img_w=128,
    resize_mode="pad",
    msr=True,
    pad_mode="edge",
    # optimization
    epochs=250,
    batch=256,
    lr=3e-4,
    min_lr=1e-6,
    weight_decay=1e-4,
    warmup_epochs=5,
    patience=60,
    grad_clip=5.0,
    amp=True,
    amp_dtype="bf16",
    compile=True,
    workers=8,
    seed=42,
    ema_decay=0.999,
    eval_ema=True,
    scheduler="cosine",
    save_every=10,
    # data
    split=(0.9, 0.05, 0.05),
    group_split=True,     # keep related samples together when a group id exists
    aug_level="digital",  # 'digital' | 'light' | 'none'
    # SVTRv2 two-phase recipe from the paper.
    ctc_weight=1.0,
    align_weight=0.5,
    align_warmup_epochs=40,
    sgm_weight=1.0,
    sgm_start_epoch=5,
    sgm_warmup_epochs=10,
    blank_bias=-2.0,
)


def load_config(path: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
    """DEFAULTS <- yaml file <- explicit overrides (None values ignored)."""
    cfg: Dict[str, Any] = dict(DEFAULTS)
    if path and Path(path).exists():
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items() if v is not None})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
