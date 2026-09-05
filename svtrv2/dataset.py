"""Manifest parsing, leakage-safe splits, MSR-aware dataset, batching.

Layout::

    <data_dir>/
      images/
  labels.txt      # "filename<TAB>text" per line

`group_id` is the filename text before the first '-'.  Samples that share a
group id should stay in a single split: near-duplicates inflate validation
scores without improving the model.

Training corpora that ship as OpenOCR-style LMDBs (Union14M-L) are supported
through :class:`LMDBTextDataset`, which reads images straight from the LMDB --
no image extraction -- and exposes the same item contract as TextDataset.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import CHARSET, MSR_BINS
from .text import CTCCodec, is_valid_label
from .transforms import build_msr_transforms, select_msr_bin

cv2.setNumThreads(0)

_BIN_BY_NAME = {b["name"]: b for b in MSR_BINS}


@dataclass(frozen=True)
class Sample:
    fname: str
    label: str
    group_id: str


def group_id_of(fname: str) -> str:
    """Samples with the same filename prefix share this group id."""
    return fname.split("-", 1)[0]



def parse_manifest(data_dir: str, verbose: bool = True) -> List[Sample]:
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

    out: List[Sample] = []
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
            out.append(Sample(fname, label, group_id_of(fname)))

    total_skipped = sum(skipped.values())
    if verbose and total_skipped:
        print(f"  manifest: kept {len(out):,}, skipped {total_skipped:,}")
        for reason, n in skipped.items():
            if n:
                print(f"    {reason:>14}: {n:,}  (e.g. {examples[reason]})")
    return out


def group_split(samples: Sequence[Sample],
                ratios: Tuple[float, float, float] = (0.9, 0.05, 0.05),
                seed: int = 42, by_group: bool = True) -> Dict[str, List[Sample]]:
    """Split into train/val/test, optionally keeping each group intact."""
    rng = random.Random(seed)
    if by_group:
        groups: Dict[str, List[Sample]] = defaultdict(list)
        for s in samples:
            groups[s.group_id].append(s)
        units = list(groups.values())
    else:
        units = [[s] for s in samples]
    rng.shuffle(units)

    n = len(samples)
    n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
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


class TextDataset(Dataset):
    """Dataset that assigns each sample an MSR bin from its raw aspect ratio.

    Bin assignment is cached to ``msr_cache.json`` beside ``images/``: it only
    needs the image header, but at hundreds of thousands of files even that is
    slow enough to notice on every epoch-zero.
    """

    def __init__(self, images_dir, samples: Sequence[Sample], transform,
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

# ------------------------------------------------------------ LMDB data source


def _jpeg_size_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    """(h, w) from a JPEG byte stream -- the SOF walk of `_jpeg_size`, no file."""
    try:
        f = io.BytesIO(data)
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


def _png_size_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        if h > 0 and w > 0:
            return h, w
    return None


def _image_size_from_bytes(data: bytes) -> Tuple[int, int]:
    """(h, w) from encoded image bytes; full decode is the last resort."""
    size = _jpeg_size_from_bytes(data) or _png_size_from_bytes(data)
    if size is None:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        size = (arr.shape[0], arr.shape[1]) if arr is not None else (32, 128)
    return size


def _charset_stamp() -> str:
    """Fingerprint of the charset, so LMDB scan caches die with the charset."""
    return hashlib.sha1(CHARSET.encode("utf-8")).hexdigest()[:12]


def _scan_lmdb_dir(lmdb_dir: Path, verbose: bool = True
                   ) -> Tuple[int, np.ndarray, np.ndarray]:
    """One-time scan of an OpenOCR-style text LMDB.

    Returns ``(num_samples, valid_mask, bin_codes)``: which 1-based indices
    carry a usable label (charset + utf-8), and which MSR bin each image
    header belongs to.  Cached to ``<lmdb_dir>/.svtrv2_scan.npz``; the cache
    is rebuilt when the sample count or the charset changes.
    """
    import lmdb

    env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True,
                    lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        raw = txn.get(b"num-samples")
    if raw is None:
        env.close()
        raise ValueError(f"{lmdb_dir}: no 'num-samples' key -- not an OpenOCR text LMDB")
    count = int(raw)

    cache_path = lmdb_dir / ".svtrv2_scan.npz"
    if cache_path.exists():
        try:
            z = np.load(cache_path)
            if int(z["count"]) == count and str(z["charset"]) == _charset_stamp():
                env.close()
                return count, z["valid"].astype(bool), z["bins"].astype(np.uint8)
        except Exception:
            pass  # corrupt/stale cache: fall through to a fresh scan

    valid = np.zeros(count, dtype=bool)
    bins = np.zeros(count, dtype=np.uint8)
    n_bad_label = 0
    with env.begin(write=False) as txn:
        t0 = time.time()
        for i in range(1, count + 1):
            lab = txn.get(f"label-{i:09d}".encode())
            if lab is None:
                continue
            try:
                text = lab.decode("utf-8").strip()
            except UnicodeDecodeError:
                n_bad_label += 1
                continue
            if is_valid_label(text):
                valid[i - 1] = True
            else:
                n_bad_label += 1
            if verbose and i % 2_000_000 == 0:
                print(f"    scan labels {lmdb_dir.name}: {i:,}/{count:,} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        t0 = time.time()
        for i in np.flatnonzero(valid) + 1:
            img = txn.get(f"image-{i:09d}".encode())
            if img is None:
                valid[i - 1] = False
                continue
            h, w = _image_size_from_bytes(img)
            info = select_msr_bin(h, w)
            bins[i - 1] = next(k for k, b in enumerate(MSR_BINS)
                               if str(b["name"]) == str(info["name"]))
            if verbose and i % 2_000_000 == 0:
                print(f"    scan images {lmdb_dir.name}: {i:,}/{count:,} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    env.close()

    try:
        np.savez_compressed(cache_path, count=count, valid=valid, bins=bins,
                            charset=_charset_stamp())
    except OSError:
        pass  # read-only media: rescan next time
    if verbose:
        print(f"    {lmdb_dir.name}: {count:,} samples, {int(valid.sum()):,} usable, "
              f"{n_bad_label:,} skipped labels", flush=True)
    return count, valid, bins


class LMDBTextDataset(Dataset):
    """TextDataset twin that reads samples straight from OpenOCR-style LMDBs.

    Built for the Union14M-L training corpus (~14M images): the images stay
    inside the LMDB instead of being extracted to tens of GB of files, while
    the item contract ``(tensor, target, len, label, bin_name)``, the
    ``bin_to_indices`` map and the collate output stay identical to
    TextDataset, so MSRBatchSampler and collate_fn work unchanged.

    Labels outside the charset are dropped (counted) at scan time, exactly
    like ``parse_manifest`` drops bad manifest rows.  Each worker process
    opens its own LMDB env lazily -- LMDB envs must not cross a fork/spawn
    boundary.
    """

    def __init__(self, lmdb_dirs, transform, codec: Optional[CTCCodec] = None,
                 verbose: bool = True) -> None:
        import lmdb  # fail here, at construction, with a clear name

        self._lmdb = lmdb
        if isinstance(lmdb_dirs, (str, Path)):
            lmdb_dirs = [lmdb_dirs]
        self.dirs = [Path(d) for d in lmdb_dirs]
        for d in self.dirs:
            if not (d / "data.mdb").exists():
                raise FileNotFoundError(f"{d}: no data.mdb")
        self.transform = transform
        self.codec = codec or CTCCodec()

        counts, valids, binss = [], [], []
        for d in self.dirs:
            c, v, b = _scan_lmdb_dir(d, verbose=verbose)
            counts.append(c)
            valids.append(v)
            binss.append(b)

        self._offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(counts)
        valid_all = np.concatenate(valids) if valids else np.zeros(0, dtype=bool)
        bins_all = np.concatenate(binss) if binss else np.zeros(0, dtype=np.uint8)
        # Position p enumerates usable (dir, 1-based-index) pairs, concatenated
        # across dirs in order.
        self._valid_pos = np.flatnonzero(valid_all).astype(np.int64)
        self._bins_valid = bins_all[self._valid_pos]

        self.bin_to_indices: Dict[str, np.ndarray] = {
            str(b["name"]): np.flatnonzero(self._bins_valid == k).astype(np.int64)
            for k, b in enumerate(MSR_BINS)
        }
        self._envs: Dict[int, list] = {}

    def __len__(self) -> int:
        return int(len(self._valid_pos))

    def _env_for(self, pid: int) -> list:
        envs = self._envs.get(pid)
        if envs is None:
            envs = [self._lmdb.open(str(d), max_readers=32, readonly=True,
                                    lock=False, readahead=False, meminit=False)
                    for d in self.dirs]
            self._envs[pid] = envs
        return envs

    def __getitem__(self, p: int):
        pos = int(self._valid_pos[p])
        d = int(np.searchsorted(self._offsets, pos, side="right") - 1)
        i = pos - int(self._offsets[d]) + 1  # 1-based LMDB index

        envs = self._env_for(os.getpid())
        with envs[d].begin(write=False) as txn:
            raw_label = txn.get(f"label-{i:09d}".encode())
            raw_img = txn.get(f"image-{i:09d}".encode())

        try:
            label = raw_label.decode("utf-8").strip() if raw_label else ""
        except UnicodeDecodeError:
            label = ""
        arr = (cv2.imdecode(np.frombuffer(raw_img, np.uint8), cv2.IMREAD_COLOR)
               if raw_img else None)
        img = (cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr is not None
               else np.zeros((32, 128, 3), dtype=np.uint8))

        info = MSR_BINS[int(self._bins_valid[p])]
        tf = self.transform[str(info["name"])] if isinstance(self.transform, dict) else self.transform
        tensor = tf(image=img)["image"]
        target = torch.tensor(self.codec.encode(label), dtype=torch.long)
        return tensor, target, len(label), label, str(info["name"])


class MSRBatchSampler(Sampler):
    """Yield batches drawn from a single MSR bin.

    Bins have different canvas widths, so images from two bins cannot be stacked
    into one tensor.  Keeping a batch inside one bin also means `torch.compile`
    sees exactly three static shapes instead of a dynamic one.
    """

    def __init__(self, dataset: TextDataset, batch_size: int,
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

    train_ds = TextDataset(images_dir, splits["train"], train_tf, codec)
    train_sampler = MSRBatchSampler(train_ds, cfg["batch"], shuffle=True,
                                    drop_last=True, seed=cfg.get("seed", 42))
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **kw)

    def eval_loader(split: str) -> Optional[DataLoader]:
        if not splits[split]:
            return None
        ds = TextDataset(images_dir, splits[split], eval_tf, codec)
        sampler = MSRBatchSampler(ds, cfg["batch"], shuffle=False, drop_last=False)
        return DataLoader(ds, batch_sampler=sampler, **kw)

    meta = dict(counts={k: len(v) for k, v in splits.items()}, splits=splits)
    return train_loader, eval_loader("val"), eval_loader("test"), meta


def build_lmdb_loaders(lmdb_dirs, cfg: Dict[str, Any], codec: Optional[CTCCodec] = None,
                       val_data_dir: Optional[str] = None
                       ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """Loaders for LMDB-backed training corpora (Union14M-L, OpenOCR layout).

    The train loader streams straight from the LMDB(s).  When ``val_data_dir``
    points at a manifest dataset (``images/`` + ``labels.txt``), its val and
    test splits become the val/test loaders -- held-out benchmark folders are
    a far better model-selection signal than a random slice of the training
    corpus.  Without it, no val loader is produced and training runs to
    ``epochs`` (or until patience is exhausted on a caller-supplied loader).

    Returns the same ``(train, val, test, meta)`` tuple as ``build_loaders``.
    """
    codec = codec or CTCCodec()
    train_tf = build_msr_transforms(True, cfg.get("aug_level", "digital"),
                                    cfg.get("pad_mode", "edge"))
    eval_tf = build_msr_transforms(False, "none", cfg.get("pad_mode", "edge"))

    train_ds = LMDBTextDataset(lmdb_dirs, train_tf, codec)
    train_sampler = MSRBatchSampler(train_ds, cfg["batch"], shuffle=True,
                                    drop_last=True, seed=cfg.get("seed", 42))
    workers = min(cfg.get("workers", 8), os.cpu_count() or 1)
    kw: Dict[str, Any] = dict(num_workers=workers, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    if workers > 0:
        kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **kw)

    val_loader = test_loader = None
    counts: Dict[str, int] = {"train": len(train_ds), "val": 0, "test": 0}
    if val_data_dir:
        samples = parse_manifest(val_data_dir)
        # Reuse a persisted split when one exists beside the manifest, so the
        # val set used for model selection never reshuffles between runs.
        splits = _load_or_make_splits(samples, {**cfg, "split": (0.9, 0.05, 0.05)}, None)
        val_ds = TextDataset(Path(val_data_dir) / "images", splits["val"], eval_tf, codec)
        test_ds = TextDataset(Path(val_data_dir) / "images", splits["test"], eval_tf, codec)
        val_loader = (DataLoader(val_ds, batch_sampler=MSRBatchSampler(
            val_ds, cfg["batch"], shuffle=False, drop_last=False), **kw)
            if len(val_ds) else None)
        test_loader = (DataLoader(test_ds, batch_sampler=MSRBatchSampler(
            test_ds, cfg["batch"], shuffle=False, drop_last=False), **kw)
            if len(test_ds) else None)
        counts["val"], counts["test"] = len(val_ds), len(test_ds)

    meta = dict(counts=counts, source="lmdb", lmdb_dirs=[str(d) for d in train_ds.dirs])
    return train_loader, val_loader, test_loader, meta


def build_eval_loader(data_dir: str, cfg: Dict[str, Any],
                      codec: Optional[CTCCodec] = None) -> Optional[DataLoader]:
    """One eval loader over *all* valid samples of a manifest dataset.

    This is the benchmark path: each folder under data/evaluation/ is one
    benchmark, every usable row counts, and results are compared per dataset
    against the paper's tables.
    """
    codec = codec or CTCCodec()
    samples = parse_manifest(data_dir)
    if not samples:
        return None
    eval_tf = build_msr_transforms(False, "none", cfg.get("pad_mode", "edge"))
    ds = TextDataset(Path(data_dir) / "images", samples, eval_tf, codec)
    workers = min(cfg.get("workers", 8), os.cpu_count() or 1)
    kw: Dict[str, Any] = dict(num_workers=workers, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    if workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, batch_sampler=MSRBatchSampler(
        ds, cfg.get("batch", 256), shuffle=False, drop_last=False), **kw)


def _load_or_make_splits(samples: Sequence[Sample], cfg: Dict[str, Any],
                         split_dir: Optional[str]) -> Dict[str, List[Sample]]:
    """Reuse persisted splits so a resumed run never reshuffles the val set."""
    by_name = {s.fname: s for s in samples}
    if split_dir and Path(split_dir).is_dir():
        loaded: Dict[str, List[Sample]] = {}
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


def save_splits(splits: Dict[str, List[Sample]], split_dir: Path) -> None:
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
