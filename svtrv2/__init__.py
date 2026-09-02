"""svtrv2 — a paper-first SVTRv2 scene text recognition implementation.

Top-level names are resolved lazily so importing the package stays cheap and
does not pull in torch unless it is actually needed.
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
