"""Charset, MSR canvases, SVTRv2 variants, and training defaults.

Target domain: **digital** utility meters — seven-segment LCD and LED registers,
photographed through the meter's transparent enclosure.  This is a different
problem from rolling-drum analog meters: the digits are crisp and high-contrast
when the light cooperates, and nearly invisible when it does not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Charset: 10 digits + '.' (the integer/fraction boundary).  Index 0 is the CTC
# blank, so characters occupy 1..len(CHARSET) and NUM_CLASSES = len+1.
CHARSET: str = "0123456789."
BLANK_IDX: int = 0
NUM_CLASSES: int = len(CHARSET) + 1  # 12

# --------------------------------------------------------------- MSR canvases

# Multi-size resizing bins, chosen from the *raw* crop aspect ratio.
#
# Two rules carried over from the analog engine, both of which cost real
# accuracy when violated:
#
# 1. Height binds.  fit-pad preserves aspect ratio, so a short canvas downscales
#    the crop and squeezes each digit into too few pixels.  Seven-segment digits
#    fail differently from drum digits when starved of resolution: the thin gap
#    between the upper-left and lower-left segments closes, and 8/9, 6/8 and 0/8
#    collapse into each other.  64px of canvas height keeps a 6-digit register
#    at ~40px per digit, which holds those gaps open.
#
# 2. Canvas aspect ratio must track the bin's image aspect ratio, or most of the
#    canvas is padding and the model burns timesteps on nothing.  Widths here are
#    ~AR * 64.
#
# WBSEDCL's single-phase spec requires the kWh register to show **at least 6
# digits**.  A tight crop of six seven-segment digits runs about 4:1; crops that
# include the unit label ("kWh") or a leading tariff digit run wider, hence the
# long bin.  These are *starting* values -- run `python -m svtrv2 bins --data DIR`
# on the real dataset and replace them with the measured percentiles.
MSR_BINS: Tuple[Dict[str, Any], ...] = (
    dict(name="short", min_ar=0.0, max_ar=3.5, height=64, width=192, feature_h=8, timesteps=24),
    dict(name="medium", min_ar=3.5, max_ar=5.0, height=64, width=288, feature_h=8, timesteps=36),
    dict(name="long", min_ar=5.0, max_ar=float("inf"), height=64, width=384, feature_h=8, timesteps=48),
)

# ------------------------------------------------------------------- variants

# SVTRv2 variants following Table 8 of the paper (arXiv 2411.15858v2) and the
# official OpenOCR implementation.
#
# `mixers` encodes the paper's [L]_m[G]_n permutation: the first m mixing blocks
# use *local* mixing (two consecutive grouped convolutions) and the last n use
# *global* mixing (MHSA).  A block is one or the other -- never both.  Heads and
# conv groups are both D_i / 32, which reproduces the paper's head counts.
#
#   name  paper   dims             depths     heads        permutation
#   S     T       (64, 128, 256)   (3, 6, 3)  (2, 4, 8)    [L]6[G]6
#   M     S       (96, 192, 384)   (3, 6, 3)  (3, 6, 12)   [L]6[G]6
#   L     B       (128, 256, 384)  (6, 6, 6)  (4, 8, 12)   [L]8[G]10
#   XL    --      (192, 384, 512)  (6, 9, 9)  (6, 12, 16)  [L]8[G]16
#
# S/M/L are the paper's three configurations verbatim, renamed.  XL is the one
# extrapolation: it keeps every structural rule (heads = D/32, local blocks
# first, global last, stage 2 as the crossover) and only scales width and depth.
_L, _G = "Local", "Global"

MODELS: Dict[str, Dict[str, Any]] = {
    "svtrv2-s": dict(
        dims=(64, 128, 256), depths=(3, 6, 3),
        mixers=((_L,) * 3, (_L,) * 3 + (_G,) * 3, (_G,) * 3),
        drop=0.05, drop_path=0.05,
    ),
    "svtrv2-m": dict(
        dims=(96, 192, 384), depths=(3, 6, 3),
        mixers=((_L,) * 3, (_L,) * 3 + (_G,) * 3, (_G,) * 3),
        drop=0.05, drop_path=0.08,
    ),
    "svtrv2-l": dict(
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

# Short aliases, so `--model s` works.
MODEL_ALIASES: Dict[str, str] = {
    "s": "svtrv2-s", "m": "svtrv2-m", "l": "svtrv2-l", "xl": "svtrv2-xl",
}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name.lower())


# ------------------------------------------------------------------- defaults

DEFAULTS: Dict[str, Any] = dict(
    # input
    img_h=64,
    img_w=288,
    resize_mode="pad",
    msr=True,
    # Digital registers sit on light-grey LCD glass or in a dark LED window, so
    # there is no single constant that blends in the way black did against a
    # drum.  Replicating the edge pixel avoids inventing a hard artificial
    # border that the model could mistake for a digit stroke.
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
    group_split=True,     # every photo of one meter_id stays in one split
    aug_level="digital",  # 'digital' | 'light' | 'none'
    # Loss schedule, SVTRv2 two-phase recipe (paper Sec. 4.1).  Phase 1 is
    # CTC-only with a uniform-alignment warmup that decays to zero, which stops
    # the early all-blank collapse.  Phase 2 ramps the semantic guidance module
    # in once the visual path is already emitting digits.
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
