"""Repair corrupted dataset files by re-downloading from HuggingFace.

For every manifest entry that is an LFS file (sha256 recorded), checks the local
copy under --root.  Files that are missing or hash-corrupt are re-downloaded
sequentially with curl (single stream -- the parallel/chunked path produced
corrupt assemblies), hash-verified in a temp file, then moved into place.

Usage:
    python tools/repair_dataset.py [--root data] [--manifest tools/dataset_manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://huggingface.co/datasets/topdu/OpenOCR-Data/resolve/main"


def sha256_of(path: Path, block: int = 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def curl(url: str, dest: Path) -> None:
    cmd = ["curl.exe", "-sS", "-L", "--retry", "5", "--retry-delay", "3",
           "-o", str(dest), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl exit {r.returncode}: {r.stderr.strip()[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data")
    ap.add_argument("--manifest", default="tools/dataset_manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = repo / root
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path

    with open(manifest_path, "r", encoding="utf-8") as fh:
        entries = json.load(fh).get("files", [])

    todo = []
    for e in entries:
        if not e.get("lfs") or not e.get("sha256"):
            continue
        p = root / e["path"]
        if p.exists() and p.stat().st_size == int(e["size"]) and sha256_of(p) == e["sha256"]:
            continue
        todo.append(e)

    print(f"{len(todo)} file(s) need repair "
          f"({sum(int(e['size']) for e in todo)/2**30:.2f} GiB)", flush=True)
    if args.dry_run:
        for e in todo:
            print("  ", e["path"])
        return 0

    failed = []
    for i, e in enumerate(todo, 1):
        rel, want_hash = e["path"], e["sha256"]
        dest = root / rel
        tmp = dest.with_suffix(dest.suffix + ".repair")
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ok = False
        for attempt in range(1, 4):
            try:
                print(f"[{i}/{len(todo)}] {rel} (attempt {attempt}) ...", flush=True)
                tmp.unlink(missing_ok=True)
                curl(f"{BASE}/{rel}?download=true", tmp)
                if tmp.stat().st_size != int(e["size"]):
                    raise RuntimeError(f"size {tmp.stat().st_size} != {e['size']}")
                got = sha256_of(tmp)
                if got != want_hash:
                    raise RuntimeError(f"sha256 {got[:12]}... != {want_hash[:12]}...")
                tmp.replace(dest)
                dt = time.time() - t0
                print(f"    OK  {int(e['size'])/2**20:.1f} MiB in {dt:.0f}s "
                      f"({int(e['size'])/2**20/max(dt,1e-9):.1f} MiB/s)", flush=True)
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"    FAIL: {exc}", flush=True)
        if not ok:
            failed.append(rel)

    print(flush=True)
    if failed:
        print(f"REPAIR FAILED for {len(failed)} file(s):", flush=True)
        for rel in failed:
            print(f"    {rel}", flush=True)
        return 1
    print("all files repaired and hash-verified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
