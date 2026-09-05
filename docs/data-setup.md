# Benchmark data setup

Everything under `data/` is **git-ignored** — it is downloaded material, not source.
This guide is the tracked copy of the local `data/SETUP.md`.

## Source

- HuggingFace dataset repo: `topdu/OpenOCR-Data` (main, revision `d59c1364`)
- Authoritative pack published by the SVTRv2 authors (OpenOCR).  Total ~12.9 GB,
  70 files.  Upstream dataset card is at `data/README.md` (from the repo).
- Mirrors for the same data (official Union14M repo, Mountchicken/Union14M):
  - Union14M-L + Union14M-Benchmark (12 GB): OneDrive `1drv.ms/u/s!AotJrudtBr-K7xAHjmr5qlHSr5Pa?e=LJRlKQ` / Baidu Yun `pan.baidu.com/s/1WiXfg9YjKiO1SzBfT14mmg?pwd=anxs`
  - 6 Common Benchmarks (17.6 MB): OneDrive `1drv.ms/u/s!AotJrudtBr-K7w8FSOI48iBI-du5?e=t8jSqN` / Baidu `pan.baidu.com/s/1XifQS0v-0YxEXkGTfWMDWQ?pwd=35cz`

Licensing note (from the dataset card): Union14M-L-Filter and Union14M-L-Benchmark
follow the Union14M-L copyright; Common benchmarks follow PARSeq's collection;
OST follows VisionLAN's.

## Layout (mirrors the OpenOCR structure)

```
data/
  Union14M-L-LMDB-Filtered/   training: filter_train_{easy,hard,medium,normal,challenging}/ (LMDB)
  evaluation/                 CUTE80, IC13_857, IC15_1811, IIIT5k, SVT, SVTP (LMDB)
  test/                       PARSeq common benchmark sets: ArT, COCOv1.4, IC13_*, IC15_*, IIIT5k, SVT, SVTP, Uber
  u14m/                       Union14M-Benchmark: curve, multi_oriented, artistic, contextless, salient, multi_words, general
  ltb.tar.xz                  Long Text Benchmark archive (extract with: tar -xf ltb.tar.xz)
  OST/                        Occluded Scene Text: heavy, weak
  wordart_test/               word-art test set
```

## Verify / repair after download

```bash
python tools/verify_dataset.py            # compares data/ against tools/dataset_manifest.json (path + size + sha256)
python tools/repair_dataset.py --dry-run  # list files that are missing/truncated/corrupt
python tools/repair_dataset.py            # re-download exactly those files (resumable, temp+.replace)
```

## Benchmark usage (charset: 94 printable ASCII + space, 96 classes)

The package charset covers every English benchmark label, and `val`/`benchmark`
read evaluation sets straight from the LMDBs -- no conversion step required.

```bash
# One checkpoint across every benchmark pack under a root (default: data/):
svtrv2 benchmark --ckpt runs/exp/best.pth --root data --csv runs/exp/benchmark.csv

# Single-set validation on an LMDB (e.g. the PARSeq common set for IIIT5k):
svtrv2 val --ckpt runs/exp/best.pth --data data/test/IIIT5k

# Union14M-L training directly from the LMDBs (no image extraction):
svtrv2 train --data data/Union14M-L-LMDB-Filtered --model svtrv2-s
```

`tools/lmdb_to_manifest.py` is still available when a plain `images/ + labels.txt`
copy is wanted (e.g. for dataset audits or `svtrv2 bins`):

```bash
python tools/lmdb_to_manifest.py data/evaluation/IIIT5k -o data/manifests/IIIT5k
```

Notes:

- LMDB keys are `num-samples`, `image-%09d`, `label-%09d` (labels are utf-8, typically
  uppercase). Images are stored as encoded JPEG/PNG bytes and are copied verbatim by
  the converter.
- Labels outside the charset are dropped and reported by both the manifest parser and
  the LMDB scanner (`_scan_lmdb_dir`), so a silent fraction-of-dataset failure cannot
  happen. LMDB scan results are cached to `.svtrv2_scan.npz` beside each LMDB and are
  keyed to the charset fingerprint.
- `tools/dataset_manifest.json` records the upstream path + byte size + sha256 of
  every file, generated from the HuggingFace tree API on 2026-09-05.