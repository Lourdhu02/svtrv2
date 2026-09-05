# Status snapshot — 2026-09-05 (v1.1.0, commit `d3810db` + this commit)

Read `plan.md` for phases and commands. This file says what is actually done, with
evidence, and what to pick up next. Update it as you work.

## Completed ✅

| Item | Evidence |
|---|---|
| GPU environment (RTX 5060 8 GB, sm_120) | `.venv` with torch 2.14.0+cu130; CUDA available; model fwd (bf16/fp32), CTC backward + optimizer step on GPU; full test suite green in-venv |
| Benchmark data pack (~12.9 GB, 70 files) | `data/` from HF `topdu/OpenOCR-Data` (pinned rev); Union14M-L LMDBs, `evaluation/`, `test/`, `u14m/`, `OST/`, `wordart_test/`, `ltb.tar.xz` |
| Data integrity + repair tooling | `tools/dataset_manifest.json`, `tools/verify_dataset.py` (size+sha256), `tools/repair_dataset.py` (resumable, hash-verified atomic swap); 25 corrupt files re-downloaded; **final verification: 70/70 files OK, 0 missing, 0 size/hash mismatches** |
| LMDB inspector + converter | `tools/inspect_lmdb.py`, `tools/lmdb_to_manifest.py` |
| Charset widened to benchmark English | 94 printable ASCII + space → 96 classes; commit `84b7e79` |
| LMDB training data layer | `LMDBTextDataset` + scan cache + auto-detect in `build_loaders`; commit `07ae382` |
| `svtrv2 benchmark` command | per-set exact/char/CER table + CSV, macro average; commit `211560d` |
| Test suite | 32/32 in `tests/test_engine.py` (commit `b9a7cc0`); run: `python tests/test_engine.py` |
| Docs | `README.md` (benchmark workflow), `docs/data-setup.md`, `data/SETUP.md`, research notes updated; 13 commits pushed to `origin/main` (`d3810db`) |

## Left ⬜ (priority order)

1. **Extract LTB**: `tar -xf data/ltb.tar.xz`; check layout; wire into `benchmark`
   if not auto-discovered. (The long-text benchmark is the research differentiator.)
2. **Fit MSR bins to the real distribution**: `python -m svtrv2 bins --data
   data/Union14M-L-LMDB-Filtered`; adjust `MSR_BINS` in `config.py` if terciles
   differ materially from (1.5, 2.5).
3. **Smoke train on GPU** (validates the whole chain): `svtrv2 train --data
   data/Union14M-L-LMDB-Filtered --model t --epochs 2 --batch 256 --device cuda --name smoke`
4. **Baseline run**: `--model s --epochs 250 --batch 256 --device cuda --name s-base`
   (early stop + EMA built in; `--resume` supported; if OOM drop batch to 128, keep LR).
5. **Benchmark the checkpoint** and compare vs paper tables
   (`research/papers/svtrv2_notes.md`); record protocol caveats in
   `research/datasets/tooling_notes.md`.
6. **Research extension** per `research/plan/proposed_plan.md` (adaptive MSR routing +
   SGM distillation, CTC-only student).

## Known caveats

- HF download tooling was flaky on this network (proxy); `repair_dataset.py` with
  chunked fallback is the reliable path — reuse it for any re-download.
- `data/` is git-ignored; anyone cloning needs `tools/repair_dataset.py --all` (or
  equivalent) to rebuild it from the manifest.
- Union14M-L scan caches (`.svtrv2_scan.npz`) are per-LMDB-dir and charset-keyed;
  delete them if the charset changes again.
