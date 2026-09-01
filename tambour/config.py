"""Charset, model-variant presets, and training defaults."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# Charset: 10 digits + '.' (integer/fraction boundary). Index 0 is the CTC blank,
# so characters occupy 1..len(CHARSET) and NUM_CLASSES = len(CHARSET) + 1.
CHARSET: str = "0123456789."
BLANK_IDX: int = 0
NUM_CLASSES: int = len(CHARSET) + 1  # 12

# SVTRv2-style variants. dims/depths/heads grow n -> s -> b -> l.
MODELS: Dict[str, Dict[str, Any]] = {
    "tambour-n": dict(dims=(32, 64, 128), depths=(2, 4, 6), heads=(2, 4, 8), drop=0.05, drop_path=0.05),
    "tambour-s": dict(dims=(48, 96, 192), depths=(3, 6, 9), heads=(3, 6, 8), drop=0.08, drop_path=0.10),
    "tambour-b": dict(dims=(64, 128, 256), depths=(3, 6, 9), heads=(2, 4, 8), drop=0.08, drop_path=0.10),
    "tambour-l": dict(dims=(96, 192, 384), depths=(3, 6, 9), heads=(3, 6, 12), drop=0.10, drop_path=0.15),
}
VARIANTS = tuple(MODELS.keys())

DEFAULTS: Dict[str, Any] = dict(
    # input (MSR: aspect-preserving pad into this canvas). 4:1 matches the digit-strip
    # aspect; a wider canvas wastes timesteps on blank padding and worsens CTC blank-bias.
    img_h=48,
    img_w=192,
    # optimization
    epochs=80,
    batch=128,
    lr=1e-3,
    min_lr=1e-6,
    weight_decay=0.01,   # 0.05 over-regularizes a from-scratch CTC model early on
    warmup_epochs=5,
    patience=20,
    grad_clip=5.0,
    amp=True,            # bf16/fp16 autocast on CUDA only
    workers=8,
    seed=42,
    ema_decay=0.999,     # validation/inference use EMA weights (smooth; CTC live weights are peaky)
    scheduler="cosine",
    save_every=10,
    # data / domains
    split=(0.9, 0.05, 0.05),
    group_split=True,    # never let one meter_id span two splits
    domain_balanced=True,
    domain_temperature=0.5,   # p(domain) ~ N_domain ** temperature (0=uniform, 1=proportional)
    num_domains=1,
    # augmentation
    aug_level="analog",  # 'analog' | 'light' | 'none'
    rolling_aug=0.15,    # prob of synthetic rolling-digit augmentation
    # losses (opt-in; standard CTC by default)
    aug_loss=False,      # rolling-tolerant CTC (AugLoss)
    aug_loss_positions=1,  # tolerate rolling on the last N digits
    sgm_weight=0.0,      # semantic-guidance auxiliary head (train-only)
    center_weight=0.0,
    focal_gamma=0.0,
    # confidence / abstain
    abstain_tau=0.0,     # 0 disables; set per precision target after calibration
    # logging / monitoring
    log_interval=50,
    wandb_project="tambour",
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
