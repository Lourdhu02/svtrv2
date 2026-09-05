# Dataset and Tooling Notes

What the repo already has:

- `images/` + `labels.txt` manifest parsing
- group-aware split persistence
- MSR bin selection and cached bin lookup
- exact-match and CER evaluation
- fixed-width ONNX export per bin

Local tooling added for the benchmark pack (2026-09-05):

- `tools/dataset_manifest.json` — upstream file manifest (path + bytes + sha256) for `topdu/OpenOCR-Data`
- `tools/verify_dataset.py` — compare `data/` against the manifest (presence + size + sha256)
- `tools/inspect_lmdb.py` — LMDB key/label/image sanity check (`num-samples`, `image-%09d`, `label-%09d`)
- `tools/lmdb_to_manifest.py` — convert LMDB dir(s) into `images/` + `labels.txt` for this repo's engine
- `tools/repair_dataset.py` — resumable re-download of exactly the corrupt/missing files

## 2026-09-05 acquisition run log

- Pack: `topdu/OpenOCR-Data` @ `d59c1364` (~12.9 GB, 70 files), downloaded via
  `huggingface_hub` (hf_xet) with aria2 fallback.
- First verification pass found 25 of 70 files (12.70 GiB) truncated/corrupt —
  every `.mdb` body was affected while `.mta` sidecars survived. Root cause was
  unstable long transfers; `repair_dataset.py` re-downloads only the flagged
  files, verifies sha256 before an atomic swap-in, and resumes cleanly.
- Repair progress at this commit: 2/25 rebuilt and sha256-verified
  (`filter_train_challenging` 1.49 GB, `filter_train_easy` 6.13 GB);
  remaining files re-downloading in the background.
- Live-data smoke on the repaired `filter_train_challenging`: 481,803 records
  scanned, 481,796 usable (7 labels outside the charset, counted), MSR bin
  distribution short/medium/long/xlong = 261,635/100,599/53,824/65,738, and a
  64-image batch collated at the expected canvas shapes.

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
