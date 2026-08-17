"""Manifest parsing, leakage-safe splits, MSR-aware dataset, batching.

Layout::

    <data_dir>/
      images/
      labels.txt      # "filename<TAB>reading" per line

`meter_id` is the filename text before the first '-'.  Every photo of one
physical meter is kept inside a single split: the same register photographed
twice is close to a duplicate, and letting one copy sit in train while the other
sits in val inflates validation accuracy without improving the model.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import MSR_BINS
from .text import CTCCodec, is_valid_label
from .transforms import MSRTransform, build_msr_transforms, select_msr_bin

cv2.setNumThreads(0)

_BIN_BY_NAME = {b["name"]: b for b in MSR_BINS}


@dataclass(frozen=True)
class MeterSample:
    fname: str
    label: str
    meter_id: str


def meter_id_of(fname: str) -> str:
    """All photos of one physical meter share this id."""
    return fname.split("-", 1)[0]


def parse_manifest(data_dir: str, verbose: bool = True) -> List[MeterSample]:
    """Read ``labels.txt``: ``fname<TAB>label`` (or whitespace-separated) per line.

    Unusable rows are skipped but **counted and reported**.  Silently dropping
    them is how a run ends up training on a fraction of the dataset without
    anyone noticing until the accuracy is inexplicably bad.
    """
    data_path = Path(data_dir)
    labels_file = data_path / "labels.txt"
    images_dir = data_path / "images"
    if not labels_file.exists():
        raise FileNotFoundError(f"{labels_file} not found")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found")

    out: List[MeterSample] = []
    skipped = {"malformed": 0, "bad_label": 0, "missing_image": 0}
    examples: Dict[str, str] = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split("\t") if "\t" in raw else raw.split()
            if len(parts) < 2:
                skipped["malformed"] += 1
                examples.setdefault("malformed", raw[:60])
                continue
            fname, label = parts[0].strip(), parts[1].strip()
            if not is_valid_label(label):
                skipped["bad_label"] += 1
                examples.setdefault("bad_label", f"{fname} -> {label!r}")
                continue
            if not (images_dir / fname).exists():
                skipped["missing_image"] += 1
                examples.setdefault("missing_image", fname)
                continue
            out.append(MeterSample(fname, label, meter_id_of(fname)))

    total_skipped = sum(skipped.values())
    if verbose and total_skipped:
        print(f"  manifest: kept {len(out):,}, skipped {total_skipped:,}")
        for reason, n in skipped.items():
            if n:
                print(f"    {reason:>14}: {n:,}  (e.g. {examples[reason]})")
    return out


def group_split(samples: Sequence[MeterSample],
                ratios: Tuple[float, float, float] = (0.9, 0.05, 0.05),
                seed: int = 42, by_group: bool = True) -> Dict[str, List[MeterSample]]:
    """Split into train/val/test, optionally keeping each meter_id intact."""
    rng = random.Random(seed)
    if by_group:
        groups: Dict[str, List[MeterSample]] = defaultdict(list)
        for s in samples:
            groups[s.meter_id].append(s)
        units = list(groups.values())
    else:
        units = [[s] for s in samples]
    rng.shuffle(units)

    n = len(samples)
    n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    splits: Dict[str, List[MeterSample]] = {"train": [], "val": [], "test": []}
    c_train = c_val = 0
    for u in units:
        if c_train < n_train:
            splits["train"].extend(u); c_train += len(u)
        elif c_val < n_val:
            splits["val"].extend(u); c_val += len(u)
        else:
            splits["test"].extend(u)
    return splits


def _jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    """Read (h, w) straight from JPEG SOF headers, without decoding pixels."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                if not b:
                    return None
                if b != b"\xff":
                    continue
                op = f.read(1)
                if op in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6",
                          b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                    f.read(3)
                    h = int.from_bytes(f.read(2), "big")
                    w = int.from_bytes(f.read(2), "big")
                    return h, w
                length = f.read(2)
                if not length:
                    return None
                f.seek(int.from_bytes(length, "big") - 2, 1)
    except Exception:
        return None


class MeterDataset(Dataset):
    """Dataset that assigns each sample an MSR bin from its raw aspect ratio.

    Bin assignment is cached to ``msr_cache.json`` beside ``images/``: it only
    needs the image header, but at hundreds of thousands of files even that is
    slow enough to notice on every epoch-zero.
    """

    def __init__(self, images_dir, samples: Sequence[MeterSample], transform,
                 codec: Optional[CTCCodec] = None) -> None:
        self.images_dir = Path(images_dir)
        self.samples = list(samples)
        self.transform = transform
        self.codec = codec or CTCCodec()

        cache_path = self.images_dir.parent / "msr_cache.json"
        cache: Dict[str, str] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        self.sample_bins: List[Dict[str, Any]] = []
        dirty = False
        for s in self.samples:
            if s.fname in cache:
                info = _BIN_BY_NAME.get(cache[s.fname], _BIN_BY_NAME["medium"])
            else:
                path = self.images_dir / s.fname
                size = _jpeg_size(path)
                if size is None:
                    img = cv2.imread(str(path))
                    size = (img.shape[0], img.shape[1]) if img is not None else (64, 288)
                info = select_msr_bin(size[0], size[1])
                cache[s.fname] = str(info["name"])
                dirty = True
            self.sample_bins.append(info)

        if dirty:
            try:
                cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            except OSError:
                pass

        self.bin_to_indices: Dict[str, List[int]] = {str(b["name"]): [] for b in MSR_BINS}
        for i, info in enumerate(self.sample_bins):
            self.bin_to_indices[str(info["name"])].append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # IMREAD_COLOR always yields 3 channels, so a single-channel file on disk
        # still satisfies the RGB-only contract.
        img = cv2.imread(str(self.images_dir / s.fname), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 288, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        info = self.sample_bins[idx]
        tf = self.transform[str(info["name"])] if isinstance(self.transform, dict) else self.transform
        tensor = tf(image=img)["image"]
        target = torch.tensor(self.codec.encode(s.label), dtype=torch.long)
        return tensor, target, len(s.label), s.label, str(info["name"])


class MSRBatchSampler(Sampler):
    """Yield batches drawn from a single MSR bin.

    Bins have different canvas widths, so images from two bins cannot be stacked
    into one tensor.  Keeping a batch inside one bin also means `torch.compile`
    sees exactly three static shapes instead of a dynamic one.
    """

    def __init__(self, dataset: MeterDataset, batch_size: int,
                 shuffle: bool = True, drop_last: bool = True, seed: int = 42) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._cache: Optional[List[List[int]]] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cache = None

    def _batches(self) -> List[List[int]]:
        # Cached per epoch: __len__ and __iter__ both need this, and rebuilding
        # it walks every index each time (tqdm alone calls len once per epoch).
        if self._cache is not None:
            return self._cache
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for indices in self.dataset.bin_to_indices.values():
            idx = list(indices)
            if not idx:
                continue
            if self.shuffle:
                rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                chunk = idx[start:start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        self._cache = batches
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def collate_fn(batch):
    """Stack a single-bin batch.

    Returns ``(images, targets, lengths, labels, padded_targets, bin_name)``.
    ``padded_targets`` is the right-padded (B, L) form with 0 as padding, which
    is what the semantic guidance module and the alignment warmup consume; the
    flat concatenated form is what ``nn.CTCLoss`` wants.
    """
    imgs, targets, lengths, labels, bin_names = zip(*batch)
    images = torch.stack(imgs, 0)
    flat = torch.cat(targets, 0) if targets else torch.zeros(0, dtype=torch.long)

    max_len = max((len(t) for t in targets), default=0)
    padded = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, :len(t)] = t

    return (images, flat, torch.tensor(lengths, dtype=torch.long),
            list(labels), padded, bin_names[0])


def build_loaders(data_dir: str, cfg: Dict[str, Any], codec: Optional[CTCCodec] = None,
                  split_dir: Optional[str] = None
                  ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Build train/val/test loaders, reusing persisted splits when present."""
    codec = codec or CTCCodec()
    samples = parse_manifest(data_dir)
    if not samples:
        raise FileNotFoundError(
            f"No usable samples in {data_dir} (need images/ + labels.txt with existing images)."
        )

    splits = _load_or_make_splits(samples, cfg, split_dir)

    images_dir = Path(data_dir) / "images"
    pad_mode = cfg.get("pad_mode", "edge")
    train_tf = build_msr_transforms(True, cfg.get("aug_level", "digital"), pad_mode)
    eval_tf = build_msr_transforms(False, "none", pad_mode)

    workers = min(cfg.get("workers", 8), os.cpu_count() or 1)
    kw: Dict[str, Any] = dict(num_workers=workers, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    if workers > 0:
        kw["persistent_workers"] = True

    train_ds = MeterDataset(images_dir, splits["train"], train_tf, codec)
    train_sampler = MSRBatchSampler(train_ds, cfg["batch"], shuffle=True,
                                    drop_last=True, seed=cfg.get("seed", 42))
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **kw)

    def eval_loader(split: str) -> Optional[DataLoader]:
        if not splits[split]:
            return None
        ds = MeterDataset(images_dir, splits[split], eval_tf, codec)
        sampler = MSRBatchSampler(ds, cfg["batch"], shuffle=False, drop_last=False)
        return DataLoader(ds, batch_sampler=sampler, **kw)

    meta = dict(counts={k: len(v) for k, v in splits.items()}, splits=splits)
    return train_loader, eval_loader("val"), eval_loader("test"), meta


def _load_or_make_splits(samples: Sequence[MeterSample], cfg: Dict[str, Any],
                         split_dir: Optional[str]) -> Dict[str, List[MeterSample]]:
    """Reuse persisted splits so a resumed run never reshuffles the val set."""
    by_name = {s.fname: s for s in samples}
    if split_dir and Path(split_dir).is_dir():
        loaded: Dict[str, List[MeterSample]] = {}
        for name in ("train", "val", "test"):
            path = Path(split_dir) / f"{name}.txt"
            if not path.exists():
                loaded = {}
                break
            names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            loaded[name] = [by_name[n] for n in names if n in by_name]
        if loaded:
            return loaded
    return group_split(samples, tuple(cfg.get("split", (0.9, 0.05, 0.05))),
                       cfg.get("seed", 42), cfg.get("group_split", True))


def save_splits(splits: Dict[str, List[MeterSample]], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (split_dir / f"{name}.txt").write_text(
            "\n".join(s.fname for s in items), encoding="utf-8")


def measure_bins(data_dir: str) -> Dict[str, Any]:
    """Report the aspect-ratio distribution of a dataset and suggest MSR bins.

    The shipped bins are starting values.  This reads every image header and
    prints the percentiles the bin edges should be fitted to.
    """
    samples = parse_manifest(data_dir)
    images_dir = Path(data_dir) / "images"
    ars: List[float] = []
    for s in samples:
        size = _jpeg_size(images_dir / s.fname)
        if size is None:
            img = cv2.imread(str(images_dir / s.fname))
            if img is None:
                continue
            size = (img.shape[0], img.shape[1])
        ars.append(size[1] / max(size[0], 1))

    if not ars:
        return {"n": 0}
    a = np.asarray(ars)
    pct = {p: float(np.percentile(a, p)) for p in (1, 10, 33, 50, 67, 90, 99)}
    lengths = [len(s.label) for s in samples]
    return {
        "n": len(a), "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()), "percentiles": pct,
        "label_len": {"min": min(lengths), "max": max(lengths),
                      "mean": float(np.mean(lengths))},
        "suggested_edges": (pct[33], pct[67]),
    }
