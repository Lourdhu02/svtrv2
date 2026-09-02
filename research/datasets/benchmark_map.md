# Benchmark Map

Recommended core shortlist for a follow-up paper:

- IIIT5K
- SVT
- ICDAR 2013
- ICDAR 2015
- SVTP
- CUTE80
- Union14M-Benchmark: Curve
- Union14M-Benchmark: Multi-Oriented
- Union14M-Benchmark: Artistic
- Union14M-Benchmark: Contextless
- Union14M-Benchmark: Salient
- Union14M-Benchmark: Multi-word
- Union14M-Benchmark: General
- LTB
- OST

What each family tests:

- IIIT5K: regular short scene words
- SVT: street-view text with low resolution and perspective noise
- ICDAR 2013: focused scene text
- ICDAR 2015: incidental scene text
- SVTP: perspective distortion
- CUTE80: curved text
- Curve: severe curvature
- Multi-Oriented: rotation and vertical text
- Artistic: stylized fonts and logos
- Contextless: weak or absent semantic priors
- Salient: adjacent / overlapping text
- Multi-word: multiple words in one crop
- General: broad hard real-world quality issues
- LTB: long text
- OST: occluded / partially erased text

Data hygiene notes:

- keep lexicon-free evaluation as the default
- track exact split variants for ICDAR 2013 and 2015
- track case and punctuation rules
- persist source URL, release tag, download date, and checksum
- preserve train/test overlap filters for any Union14M-derived data

Folder split suggestion:

- `research/datasets/common/`
- `research/datasets/union14m/`
- `research/datasets/special/`
- `research/datasets/multilingual/`
