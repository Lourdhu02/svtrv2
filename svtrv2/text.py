"""CTC label codec for scene text recognition.

The codec is intentionally minimal: encode the target string to class indices
and greedily decode predicted index sequences back to text.
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
        """Greedy CTC contraction.

        A class is emitted when it differs from the previous timestep's class.
        Adjacent duplicates collapse, but a duplicate separated by a blank is a
        real repeat.
        """
        out: List[str] = []
        for t, idx in enumerate(indices):
            idx = int(idx)
            if idx == self.blank:
                continue
            if t > 0 and int(indices[t - 1]) == idx:
                continue
            c = _I2C.get(idx, "")
            if c:
                out.append(c)
        return "".join(out)

    def decode_with_conf(self, log_probs: "np.ndarray") -> Tuple[str, float, List[float]]:
        """Greedy-decode one sample's (T, C) log-probs with per-character confidence.

        Returns ``(text, min_confidence, per_char_confidences)``.  The minimum
        over emitted characters is the conservative sequence confidence.
        """
        probs = np.exp(log_probs)
        idx = probs.argmax(-1)
        maxp = probs.max(-1)
        out: List[str] = []
        confs: List[float] = []
        for t, i in enumerate(idx):
            i = int(i)
            if i == self.blank:
                continue
            if t > 0 and int(idx[t - 1]) == i:
                continue
            c = _I2C.get(i, "")
            if c:
                out.append(c)
                confs.append(float(maxp[t]))
        text = "".join(out)
        return text, (min(confs) if confs else 0.0), confs


def split_integer_fraction(label: str) -> Tuple[str, str]:
    """'text.part' -> ('text', 'part'); 'text' -> ('text', '')."""
    if "." in label:
        a, b = label.split(".", 1)
        return a, b
    return label, ""


def is_valid_label(label: str) -> bool:
    return len(label) > 0 and all(c in _C2I for c in label)
