"""Build a benchmark-manifest layout (images/ + labels.txt) from the LTB pack.

The LTB distribution ships as JPEGs in ``img/`` plus TSV lists
(``<fname>\\t<label>``) grouped by text length.  ``svtrv2 benchmark`` expects
one directory per benchmark set with ``images/`` + ``labels.txt``, so this
tool merges the lists, drops duplicates, filters labels to the charset
(counted and reported), and copies the referenced images.

Usage:
    python tools/build_ltb_manifest.py --ltb data/ltb --out data/ltb_benchmark
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svtrv2.text import is_valid_label  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ltb", default="data/ltb", help="extracted LTB directory")
    ap.add_argument("--out", default="data/ltb_benchmark", help="output benchmark dir")
    args = ap.parse_args()

    ltb, out = Path(args.ltb), Path(args.out)
    img_src = ltb / "img"
    if not img_src.is_dir():
        print(f"error: {img_src} not found", file=sys.stderr)
        return 1

    # Merge the length-grouped lists; first label wins on duplicates.
    entries: dict[str, str] = {}
    lists = sorted(ltb.glob("*_list.txt"))
    if not lists:
        print(f"error: no *_list.txt files in {ltb}", file=sys.stderr)
        return 1
    for lst in lists:
        for line in lst.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            fname, _, label = line.partition("\t")
            entries.setdefault(fname.strip(), label.strip())

    (out / "images").mkdir(parents=True, exist_ok=True)
    kept, dropped_label, missing_img = [], 0, 0
    for fname, label in sorted(entries.items()):
        if not is_valid_label(label):
            dropped_label += 1
            continue
        # List entries may carry an "img/" prefix; resolve against img_src.
        src = img_src / Path(fname).name
        if not src.is_file():
            missing_img += 1
            continue
        dst = out / "images" / Path(fname).name
        if not dst.exists():
            shutil.copy2(src, dst)
        kept.append(f"{Path(fname).name}\t{label}")

    (out / "labels.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"lists merged : {len(lists)} files, {len(entries):,} unique entries")
    print(f"kept         : {len(kept):,}  (charset-valid + image found)")
    print(f"dropped      : {dropped_label:,} labels outside charset "
          f"(non-ASCII), {missing_img:,} missing images")
    print(f"wrote        : {out / 'labels.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
