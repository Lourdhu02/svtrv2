# SVTRv2 benchmark project — plan

Goal: reproduce SVTRv2 (arXiv 2411.15858v2) on the Union14M benchmark suite with a
clean, tested implementation, then extend it toward the research direction in
`research/plan/proposed_plan.md` (adaptive MSR routing + SGM distillation that keeps
inference CTC-only).

Machine: OMEN laptop, RTX 5060 Laptop GPU 8 GB (Blackwell, sm_120), Ryzen 9 8940HX,
Python 3.12 venv at `.venv/`, torch 2.14.0+cu130.

## Phases

### Phase 0 — Environment (DONE)
- CUDA 13 torch + torchvision in `.venv` (CPU build replaced; sm_120 needs cu128+).
- onnxruntime-gpu, lmdb, huggingface_hub installed; project installed editable.
- Verified: CUDA available, model fwd (bf16/fp32), CTC backward + optimizer step on GPU,
  full test suite green inside the venv.

### Phase 1 — Benchmark data (DONE, verify after every re-download)
- Full pack from HF `topdu/OpenOCR-Data` (pinned revision) under `data/`:
  Union14M-L training LMDBs, `evaluation/`, `test/` (common benchmarks), `u14m/`
  (Union14M-B), `OST/`, `wordart_test/`, `ltb.tar.xz` (long-text).
- Integrity tooling: `tools/dataset_manifest.json` + `tools/verify_dataset.py`
  (path + size + sha256); `tools/repair_dataset.py` re-downloads only bad files.
- 25 of 70 files arrived corrupt and were re-downloaded + hash-verified.
- LTB archive still needs extraction: `tar -xf data/ltb.tar.xz` (check layout, then
  wire into `benchmark` if not auto-discovered).

### Phase 2 — Package readiness (DONE)
- Charset: 94 printable ASCII + space (96 classes with CTC blank).
- `LMDBTextDataset`: trains directly from OpenOCR-style LMDBs; same item contract,
  batch sampler and collate as the manifest dataset.
- CLI: `svtrv2 benchmark` evaluates every set under a root in one run (table + CSV);
  `val` works on LMDB dirs directly.
- Tests: 32/32 in `tests/test_engine.py` (standalone or pytest).

### Phase 3 — Baseline training (NEXT)
1. Fit MSR bins to the actual training distribution:
   `python -m svtrv2 bins --data data/Union14M-L-LMDB-Filtered` and adjust
   `MSR_BINS` in `svtrv2/config.py` if the terciles differ materially from
   (1.5, 2.5).
2. Start small to validate the pipeline end-to-end on GPU, then full runs:
   - smoke: `svtrv2 train --data data/Union14M-L-LMDB-Filtered --model svtrv2-t
     --epochs 2 --batch 256 --device cuda --name smoke`
   - baseline: `--model svtrv2-s --epochs 250 --batch 256 --device cuda --name s-base`
     (early stopping + EMA built in; resume with `--resume`).
3. 8 GB VRAM notes: keep `--batch 256` with bf16; if OOM, drop to 128 and keep the
   LR; `torch.compile` stays on for CUDA.
4. Track runs in `runs/<name>/` (history.json, best.pth, last.pth).

### Phase 4 — Benchmark evaluation (AFTER a trained checkpoint)
- `svtrv2 benchmark --ckpt runs/s-base/best.pth --root data --csv` then compare the
  per-set exact/CER table against the paper's numbers (Table 1/2 in the notes in
  `research/papers/svtrv2_notes.md`).
- Record protocol caveats in `research/datasets/tooling_notes.md` (charset case,
  EMA vs live weights, bin edges).

### Phase 5 — Research extension (AFTER baselines)
- Follow `research/plan/proposed_plan.md`: adaptive MSR routing + SGM distillation
  with a CTC-only student; target the documented gaps in
  `research/gaps/research_gaps.md` (long-text, occlusion, generalization).
- LTB (long text) is the differentiating benchmark — extract and include it early.

## Command reference
```
.venv\Scripts\Activate.ps1
python tools/verify_dataset.py                                  # data integrity
python tools/repair_dataset.py                                  # fix bad files
python -m svtrv2 info                                           # variant summary
python -m svtrv2 bins --data data/Union14M-L-LMDB-Filtered      # fit MSR bins
python -m svtrv2 train --data data/Union14M-L-LMDB-Filtered --model s --device cuda
python -m svtrv2 benchmark --ckpt runs/<name>/best.pth --root data
python tests/test_engine.py                                     # 32 tests
```
