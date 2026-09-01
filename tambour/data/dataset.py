"""Map-style dataset + collate for the folder layout (images/ + labels.txt)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..text import CTCCodec
from .manifest import MeterSample


class MeterDataset(Dataset):
    def __init__(self, images_dir, samples: Sequence[MeterSample], transform,
                 codec: CTCCodec, domain_to_id: Dict[str, int]):
        self.images_dir = Path(images_dir)
        self.samples = list(samples)
        self.transform = transform
        self.codec = codec
        self.domain_to_id = domain_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = cv2.imread(str(self.images_dir / s.fname), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((32, 128, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)["image"]
        target = torch.tensor(self.codec.encode(s.label), dtype=torch.long)
        domain_id = self.domain_to_id.get(s.domain, 0)
        return tensor, target, len(s.label), s.label, domain_id

    def __repr__(self) -> str:
        return f"MeterDataset(samples={len(self.samples)}, domains={len(self.domain_to_id)})"


def collate_fn(batch):
    imgs, targets, lengths, labels, domain_ids = zip(*batch)
    images = torch.stack(imgs, 0)
    target_concat = torch.cat(targets, 0) if targets else torch.zeros(0, dtype=torch.long)
    return (images, target_concat, torch.tensor(lengths, dtype=torch.long),
            list(labels), torch.tensor(domain_ids, dtype=torch.long))
