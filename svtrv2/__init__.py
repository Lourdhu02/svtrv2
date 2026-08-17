"""svtrv2 — OCR engine for digital utility meter displays.

Reads the digit string from a pre-cropped seven-segment LCD or LED register with
an SVTRv2 CTC recognizer built to the paper (arXiv 2411.15858v2): local/global
mixing blocks, a feature rearrangement module, and a training-only semantic
guidance module that fuses away at inference.

Top-level names are resolved lazily, so importing the package stays cheap and
never drags in torch.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "CHARSET", "NUM_CLASSES", "MODELS", "VARIANTS", "MSR_BINS",
    "load_config", "resolve_model_name", "SVTRNet", "CTCCodec", "__version__",
]

_CONFIG_NAMES = {"CHARSET", "NUM_CLASSES", "MODELS", "VARIANTS", "MSR_BINS",
                 "load_config", "resolve_model_name"}


def __getattr__(name):
    if name in _CONFIG_NAMES:
        from . import config

        return getattr(config, name)
    if name == "SVTRNet":
        from .model import SVTRNet

        return SVTRNet
    if name == "CTCCodec":
        from .text import CTCCodec

        return CTCCodec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
