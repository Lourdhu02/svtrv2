"""Convert OpenOCR-style LMDB dir(s) into this repo's images/ + labels.txt layout.

The repo's engine consumes:
    <root>/images/<fname>   (RGB image files)
    <root>/labels.txt       "fname<TAB>label" per line

Usage:
    python tools/lmdb_to_manifest.py data/evaluation/IIIT5k -o data/manifests/IIIT5k
    python tools/lmdb_to_manifest.py data/u14m/curve data/u14m/artistic -o data/manifests/u14m_curve+artistic

Image bytes are taken verbatim from the LMDB value (JPEG/PNG already stored), so
nothing is re-encoded and the files stay identical to the source.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import lmdb


def iter_samples(lmdb_dir: Path, limit: int | None = None) -> List[Tuple[str, str, bytes]]:
    """Return (fname, label, image_bytes) pairs in 1-based index order."""
    if not (lmdb_dir / "data.mdb").exists():
        raise FileNotFoundError(f"no data.mdb in {lmdb_dir}")
    env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True,
                    lock=False, readahead=False, meminit=False)
    out: List[Tuple[str, str, bytes]] = []
    with env.begin(write=False) as txn:
        raw_count = txn.get(b"num-samples")
        count = int(raw_count) if raw_count else 0
        for i in range(1, count + 1):
            if limit is not None and len(out) >= limit:
                break
            img_key = f"image-{i:09d}".encode()
            label_key = f"label-{i:09d}".encode()
            img = txn.get(img_key)
            label = txn.get(label_key)
            if img is None or label is None:
                continue
            ext = _guess_ext(img)
            fname = f"{i:05d}{ext}"
            out.append((fname, label.decode("utf-8"), img))
    env.close()
    return out


def _guess_ext(img: bytes) -> str:
    if img[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if img[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".img"


def convert(inputs: List[Path], out_root: Path, limit: int | None = None) -> int:
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    total = 0
    seen: set[str] = set()
    for lmdb_dir in inputs:
        samples = iter_samples(lmdb_dir, limit=limit)
        for fname, label, img_bytes in samples:
            if fname in seen:
                continue
            seen.add(fname)
            (images_dir / fname).write_bytes(img_bytes)
            lines.append(f"{fname}\t{label}")
            total += 1
        print(f"  {lmdb_dir.name}: {len(samples):,} samples")
    if not lines:
        print("  warning: no samples converted", file=sys.stderr)
        return 1

    labels_path = out_root / "labels.txt"
    labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {total:,} samples -> {labels_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lmdb_dir", nargs="+", help="one or more LMDB dirs (data.mdb + lock.mdb)")
    ap.add_argument("-o", "--out", required=True, help="output dir for images/ + labels.txt")
    ap.add_argument("--limit", type=int, default=None, help="cap samples per source (for smoke tests)")
    args = ap.parse_args()
    return convert([Path(d) for d in args.lmdb_dir], Path(args.out), limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())