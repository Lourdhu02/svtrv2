"""Build the per-LMDB scan caches (.svtrv2_scan.npz) for a corpus root.

One-time cost that every later training/eval run reuses: the cache records
which samples carry charset-valid labels and which MSR bin each image header
belongs to.  Safe to re-run — existing valid caches are skipped.

Usage:
    python tools/scan_lmdb_caches.py --root data/Union14M-L-LMDB-Filtered
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svtrv2.dataset import _scan_lmdb_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/Union14M-L-LMDB-Filtered")
    args = ap.parse_args()

    root = Path(args.root)
    dirs = sorted(d for d in root.iterdir() if (d / "data.mdb").is_file())
    if not dirs:
        print(f"error: no LMDB dirs (data.mdb) under {root}", file=sys.stderr)
        return 1

    t0 = time.time()
    for d in dirs:
        cache = d / ".svtrv2_scan.npz"
        if cache.is_file():
            print(f"skip (cached): {d.name}", flush=True)
            continue
        print(f"scanning: {d.name} ...", flush=True)
        count, valid, _bins = _scan_lmdb_dir(d)
        print(f"done: {d.name}  {int(valid.sum()):,}/{count:,} usable "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"all caches ready in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
