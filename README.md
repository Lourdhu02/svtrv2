# svtrv2

SVTRv2 research implementation for scene text recognition.

This repository is being refactored as a paper-first codebase:
- base paper: [SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](papers/SVTRv2_2411.15858v2.pdf)
- focus: reproducible training, evaluation, ablations, and benchmark reporting
- scope: no meter-OCR product framing, no deployment-specific branding

## What is here

- SVTRv2 visual backbone with local/global mixing
- FRM and train-only SGM
- MSR preprocessing and bucketed batching
- synthetic tests that exercise the full training/inference/export loop
- CLI for training, validation, prediction, export, and dataset inspection

## Quick Start

```bash
pip install -e .[dev]
python -m svtrv2 info --model s
python -m svtrv2 bins --data /path/to/benchmark
python -m svtrv2 train --model s --data /path/to/benchmark --device cuda
python -m svtrv2 val --ckpt runs/exp/best.pth --data /path/to/benchmark
```

## Variants

The repo uses the paper naming convention:

| variant | dims | depths | heads | permutation |
|---|---|---|---|---|
| `svtrv2-t` | 64 / 128 / 256 | 3, 6, 3 | 2, 4, 8 | `[L]6[G]6` |
| `svtrv2-s` | 96 / 192 / 384 | 3, 6, 3 | 3, 6, 12 | `[L]6[G]6` |
| `svtrv2-b` | 128 / 256 / 384 | 6, 6, 6 | 4, 8, 12 | `[L]8[G]10` |
| `svtrv2-xl` | 192 / 384 / 512 | 6, 9, 9 | 6, 12, 16 | `[L]8[G]16` |

`t`, `s`, `b`, and `xl` are accepted as short CLI aliases.

## Benchmark Direction

This repo is being reorganized around the paper benchmark workflow:
- benchmark-specific configs live in YAML or CLI overrides
- the default preprocessing is MSR-based
- evaluation should be done on fixed benchmark splits, not ad hoc random splits
- reproducibility artifacts are stored beside each run

The current code still supports the old synthetic test harness, but the intended
end state is a clean research workflow centered on paper benchmarks.

## CLI

```bash
python -m svtrv2 train   --model s --data DATA --device cuda --name run1
python -m svtrv2 train   --model s --data DATA --name run1 --resume
python -m svtrv2 val     --ckpt runs/run1/best.pth --data DATA
python -m svtrv2 predict --ckpt runs/run1/best.pth --source image_or_dir -o predictions.csv
python -m svtrv2 export  --ckpt runs/run1/best.pth --bin medium
python -m svtrv2 info    --model xl
python -m svtrv2 bins    --data DATA
```

`predict` writes `filename,text,confidence,flag` for directory inputs and flags
rows below `--min-conf` as `REVIEW`.

## Dataset layout

```text
<data_dir>/
  images/
  labels.txt      # "filename<TAB>text" per line
```

The manifest parser keeps related samples together when a shared group id is
present in the filename prefix before the first `-`.

## Paper PDF

The base paper is checked into the repo for local reference:

- [`papers/SVTRv2_2411.15858v2.pdf`](papers/SVTRv2_2411.15858v2.pdf)

## Tests

```bash
python -m pytest tests -q
```

The suite is CPU-only and synthetic, so it does not require a benchmark download
to validate the implementation.
