# AUDIT-03: Dataset Correctness

Baseline:

```text
branch: audit/formicai-code-audit
immutable experiment baseline: ants-v4-domain-e01-final / 087c4cf8ce41790160de409fc0a1f0ab807ae42c
AUDIT-01: VERIFIED
AUDIT-02: VERIFIED
```

Problems addressed:

- `DatasetValidator` parsed `dataset.yaml` but validated hardcoded `images/train`, `labels/train`, `images/val`, and `labels/val`.
- `FrameExtractor` could write into a previously populated extraction directory, leaving stale images from earlier runs.
- `DatasetStatisticsCalculator` treated missing label files as zero-annotation images.
- Core YOLO boundary validation used effectively zero normalized tolerance, which rejected legitimate subpixel export rounding.

Implemented behavior:

- Dataset validation resolves `path`, `train`, and `val` from `dataset.yaml` relative to the YAML location, including relative and absolute dataset roots.
- Dataset statistics uses the same shared `dataset.yaml` path-resolution helper as validation; it no longer reads hardcoded local `images/train` or `images/val` paths.
- Label directories are derived from YOLO layout semantics by mapping an `images/...` path to the corresponding `labels/...` path; unsupported layouts fail clearly.
- Frame extraction refuses prior extractor-owned outputs by default. `--overwrite` explicitly deletes only supported extracted image files and `metadata.json` in the output directory, preserving unrelated files.
- Dataset statistics now reads image dimensions before parsing labels, distinguishes an intentionally empty label file from a missing label file, and applies the same `0.5` physical-pixel boundary policy as validation. Empty `.txt` labels remain valid zero-ant images; missing labels raise `ValueError`.
- Image-aware YOLO validation accepts boundary overshoot up to `0.5` physical pixels when image dimensions are known, using axis-specific normalized tolerances derived from image width/height.
- Segmentation polygons follow the same image-aware boundary policy for their source points when image dimensions are known; without image dimensions, normalized segmentation coordinates remain strictly bounded.
- The same `0.5` physical-pixel image-aware boundary policy is applied when validating external-test labels and loading ground truth for custom public diagnostics whenever real image dimensions are known.
- Validation never clamps, rewrites, or mutates annotation coordinates.

No training, real model inference, benchmark evaluation, dataset mutation, annotation mutation, model-weight mutation, final-test freeze, v5 design, tracking work, push, or merge occurred as part of AUDIT-03.
