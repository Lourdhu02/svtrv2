"""Inspect an OpenOCR-style LMDB directory.

Keys used by OpenOCR/PaddleOCR-style recognition LMDSets:
    num-samples      -> int (1-based indexable count)
    image-%09d       -> raw encoded image bytes (JPEG/PNG)
    label-%09d       -> utf-8 ground-truth text

Usage:
    python tools/inspect_lmdb.py <lmdb_dir> [--head N]

Example:
    python tools/inspect_lmdb.py data/evaluation/IIIT5k --head 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import lmdb
import numpy as np


def inspect(lmdb_dir: Path, head: int = 3) -> int:
    if not (lmdb_dir / "data.mdb").exists():
        print(f"error: no data.mdb in {lmdb_dir}", file=sys.stderr)
        return 1
    env = lmdb.open(str(lmdb_dir), max_readers=32, readonly=True,
                    lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        raw_count = txn.get(b"num-samples")
        count = int(raw_count) if raw_count else None
        print(f"  dir          : {lmdb_dir}")
        print(f"  num-samples  : {count}")

        shown = 0
        with env.begin(write=False) as txn2:
            cursor = txn2.cursor()
            for key, value in cursor:
                if key.startswith(b"label-"):
                    try:
                        label = value.decode("utf-8")
                    except UnicodeDecodeError:
                        label = f"<undecodable {len(value)}B>"
                    if key.startswith(b"label-000000001") or shown == 0:
                        print(f"    {key.decode():<24} -> {label!r}")
                elif key.startswith(b"image-") and shown < head:
                    arr = np.frombuffer(value, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"    {key.decode():<24} -> <undecodable {len(value)}B>")
                    else:
                        h, w = img.shape[:2]
                        print(f"    {key.decode():<24} -> {w}x{h} RGB, {len(value)}B "
                              f"label={txn2.get((b'label-' + key[6:]))!r}")
                    shown += 1
    env.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lmdb_dir", help="directory containing data.mdb + lock.mdb")
    ap.add_argument("--head", type=int, default=3, help="how many image samples to show")
    args = ap.parse_args()
    return inspect(Path(args.lmdb_dir), head=args.head)


if __name__ == "__main__":
    sys.exit(main())