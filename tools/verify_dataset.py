"""Verify the downloaded OpenOCR-Data tree against the upstream HuggingFace manifest.

Usage:
    python tools/verify_dataset.py [--root data] [--manifest tools/dataset_manifest.json]

Checks every manifest file for presence, exact byte size, and (when a sha256 is
recorded) the content hash.  Exit code 0 means every manifest file exists with
the expected size and hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path, block: int = 2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data", help="local dataset root (relative to repo or absolute)")
    ap.add_argument("--manifest", default="tools/dataset_manifest.json")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 checks, size only")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = repo / root
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path

    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if isinstance(manifest, dict):  # tolerate {"files":[...]} wrappers
        manifest = manifest.get("files", [])

    missing, wrong, bad_hash, ok = [], [], [], []
    for entry in manifest:
        rel = entry.get("path", "")
        want = int(entry.get("size", 0))
        p = root / rel
        if not p.exists():
            missing.append((rel, want))
            continue
        if p.stat().st_size != want:
            wrong.append((rel, p.stat().st_size, want))
            continue
        if args.no_hash or not entry.get("sha256") or not entry.get("lfs"):
            ok.append(rel)  # non-LFS files: git-blob oid is not the content hash
            continue
        got = sha256_of(p)
        if got != entry["sha256"]:
            bad_hash.append((rel, got, entry["sha256"]))
        else:
            ok.append(rel)

    print(f"  files checked   : {len(ok) + len(missing) + len(wrong) + len(bad_hash)}")
    print(f"  ok              : {len(ok)}")
    print(f"  missing         : {len(missing)}")
    print(f"  size mismatch   : {len(wrong)}")
    print(f"  hash mismatch   : {len(bad_hash)}")
    for rel, want in missing[:200]:
        print(f"    MISSING {rel}  ({want:,} bytes)")
    for rel, got, want in wrong[:200]:
        print(f"    SIZEMISMATCH {rel}  ({got:,} != {want:,})")
    for rel, got, want in bad_hash[:200]:
        print(f"    HASHMISMATCH {rel}  (sha256 {got[:12]}... != {want[:12]}...)")

    return 0 if not (missing or wrong or bad_hash) else 1


if __name__ == "__main__":
    sys.exit(main())