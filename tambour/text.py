"""CTC label codec + the rolling-counter labeling/billing conventions.

Labeling convention (UFPR/Copel): when a wheel is mid-rotation, the ground truth
is the **lowest fully-or-mostly visible digit** (the 9->0 transition is labeled 9).
The red fractional wheel is recorded after a '.' but excluded from the billed read.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .config import BLANK_IDX, CHARSET

_C2I = {c: i + 1 for i, c in enumerate(CHARSET)}   # blank=0, chars 1..N
_I2C = {i + 1: c for i, c in enumerate(CHARSET)}


class CTCCodec:
    """Encode label strings to class indices and greedy-decode predictions."""

    blank = BLANK_IDX
    charset = CHARSET
    num_classes = len(CHARSET) + 1
    c2i = _C2I
    i2c = _I2C

    def encode(self, text: str) -> List[int]:
        return [_C2I[c] for c in text]

    def decode(self, indices: Sequence[int]) -> str:
        """Greedy CTC contraction: merge repeats, drop blanks."""
        out: List[str] = []
        prev = -1
        for idx in indices:
            idx = int(idx)
            if idx != self.blank and idx != prev:
                out.append(_I2C.get(idx, ""))
            prev = idx
        return "".join(out)

    def decode_with_conf(self, log_probs: "np.ndarray") -> Tuple[str, float, List[float]]:
        """Greedy-decode one sample's (T, C) log-probs with per-character confidence.

        Returns (text, min_confidence, per_char_confidences). The min over emitted
        characters (the weakest digit) is the conservative sequence confidence used
        for abstention — one shaky digit makes the whole exact-match read uncertain.
        """
        probs = np.exp(log_probs)
        idx = probs.argmax(-1)
        maxp = probs.max(-1)
        out: List[str] = []
        confs: List[float] = []
        prev = -1
        for t, i in enumerate(idx):
            i = int(i)
            if i != self.blank and i != prev:
                c = _I2C.get(i, "")
                if c:
                    out.append(c)
                    confs.append(float(maxp[t]))
            prev = i
        text = "".join(out)
        return text, (min(confs) if confs else 0.0), confs


def split_integer_fraction(label: str) -> Tuple[str, str]:
    """'01234.5' -> ('01234', '5'); '01234' -> ('01234', '')."""
    if "." in label:
        a, b = label.split(".", 1)
        return a, b
    return label, ""


def billed_reading(label: str) -> str:
    """The integer part only - what utilities actually bill (red wheel excluded).

    Leading zeros are preserved (they are billing-significant).
    """
    return split_integer_fraction(label)[0]


def is_valid_label(label: str) -> bool:
    return len(label) > 0 and all(c in _C2I for c in label)


def get_charset_size() -> int:
    """Return the number of characters in the charset (excluding CTC blank)."""
    return len(CHARSET)
