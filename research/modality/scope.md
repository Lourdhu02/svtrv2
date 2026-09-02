# Modality Scope

SVTRv2 is fundamentally a scene-text recognizer.

Best-fit modalities for a follow-up paper:

1. Scene text
2. Long scene text
3. Multilingual scene text
4. Occluded / irregular scene text

Possible but higher-risk modalities:

- document line OCR
- handwritten OCR
- historical OCR

What the current repo already supports well:

- crop-level OCR
- variable aspect ratios
- short-to-long sequence handling
- strong local/global feature mixing

What remains underexplored:

- open-vocabulary text
- multilingual scripts
- line-level long text
- severe curved or perspective distortion
- domain shift beyond printed scene text

Recommended paper framing:

- keep the task in scene text recognition
- expand benchmark coverage
- add a new method for adaptive resizing and inference-preserving context

