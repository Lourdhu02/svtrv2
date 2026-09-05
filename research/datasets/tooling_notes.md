# Dataset and Tooling Notes

What the repo already has:

- `images/` + `labels.txt` manifest parsing
- group-aware split persistence
- MSR bin selection and cached bin lookup
- exact-match and CER evaluation
- fixed-width ONNX export per bin

Local tooling added for the benchmark pack (2026-09-05):

- `tools/dataset_manifest.json` — upstream file manifest (path + bytes) for `topdu/OpenOCR-Data`
- `tools/verify_dataset.py` — compare `data/` against the manifest (presence + exact size)
- `tools/inspect_lmdb.py` — LMDB key/label/image sanity check (`num-samples`, `image-%09d`, `label-%09d`)
- `tools/lmdb_to_manifest.py` — convert LMDB dir(s) into `images/` + `labels.txt` for this repo's engine

What the research workspace should track:

- benchmark download links and access restrictions
- preprocessing policy for each benchmark family
- split names and benchmark variants
- metrics and evaluation protocol
- checksums, release tags, and date of acquisition

What should be recorded per experiment:

- git commit
- dataset release or source commit
- exact benchmark pack
- exact evaluation command
- charset and normalization policy
- whether EMA, compile, or AMP affected eval
