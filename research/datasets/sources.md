# Primary Sources

- SVTRv2 paper: https://arxiv.org/abs/2411.15858
- OpenOCR SVTRv2 docs: https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md
- OpenOCR benchmark/config docs: https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md
- IIIT5K official task page: https://tc11.cvc.uab.es/datasets/IIIT%205K-Word_1/task_1_1
- SVT official page: https://cvit.iiit.ac.in/research/projects/cvit-projects/scene-text-understanding
- ICDAR 2015/RRC overview: https://rrc.cvc.uab.es/?ch=4
- CUTE80 official page: https://cs-chan.com/downloads_CUTE80_dataset.html
- Union14M official repo: https://github.com/Mountchicken/Union14M
- PARSeq benchmark convention reference: https://github.com/baudm/parseq

## Data packs (verified 2026-09-05)

### Primary pack — HuggingFace `topdu/OpenOCR-Data` (12.9 GB, all that SVTRv2 releases)

- repo: https://huggingface.co/datasets/topdu/OpenOCR-Data
- revision: `d59c1364` (main)
- contents:
  - `Union14M-L-LMDB-Filtered/` — paper training set (filter_train_{easy,hard,medium,normal,challenging}, LMDB) ~9.9 GB
  - `evaluation/` — CUTE80, IC13_857, IC15_1811, IIIT5k, SVT, SVTP (LMDB)
  - `test/` — PARSeq common tests: ArT, COCOv1.4, IC13_857/1015/1095, IC15_1811/2077, IIIT5k, SVT, SVTP, Uber
  - `u14m/` — Union14M-Benchmark: curve, multi_oriented, artistic, contextless, salient, multi_words, general
  - `ltb.tar.xz` — Long Text Benchmark; `OST/` (heavy, weak) — Occluded Scene Text; `wordart_test/`
- local mirror of the exact file manifest (path + bytes): `tools/dataset_manifest.json`

### Official Union14M repo mirrors

- Union14M-L + Union14M-Benchmark (12 GB)
  - OneDrive: https://1drv.ms/u/s!AotJrudtBr-K7xAHjmr5qlHSr5Pa?e=LJRlKQ
  - Baidu Yun: https://pan.baidu.com/s/1WiXfg9YjKiO1SzBfT14mmg?pwd=anxs
- Union14M-U (36.63 GB, unlabeled)
  - OneDrive: https://1drv.ms/f/s!AotJrudtBr-K7xGbC7wFSU62-R9m?e=ywgQAx
  - Baidu Yun: https://pan.baidu.com/s/1yOUCYgjwSB8czmZyyX56PA?pwd=4c9v
- 6 Common Benchmarks (17.6 MB)
  - OneDrive: https://1drv.ms/u/s!AotJrudtBr-K7w8FSOI48iBI-du5?e=t8jSqN
  - Baidu Yun: https://pan.baidu.com/s/1XifQS0v-0YxEXkGTfWMDWQ?pwd=35cz

### Preprocessed/LMDB files for building the filter set

- Filtered image list (create Union14M-L-Filter from Union14M-L): https://drive.google.com/drive/folders/1x1LC8C_W-Frl3sGV9i9_i_OD-bqNdodJ
- Latency-measurement IIIT5K images: https://drive.google.com/drive/folders/1Po1LSBQb87DxGJuAgLNxhsJ-pdXxpIfS
