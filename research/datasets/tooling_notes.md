# Dataset and Tooling Notes

What the repo already has:

- `images/` + `labels.txt` manifest parsing
- group-aware split persistence
- MSR bin selection and cached bin lookup
- exact-match and CER evaluation
- fixed-width ONNX export per bin

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
