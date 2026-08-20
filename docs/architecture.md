# FormicAI Vision Architecture

FormicAI Vision is a local-first computer-vision toolkit for ant-colony video analysis. The repository now contains two mature paths:

- motion analysis for activity/movement inspection
- ant detection dataset/evaluation tooling for YOLO-style single-class detectors

The current experimental detector champion is `ants_v3_mixedscale_e01`. `ants_v4_domain_e01` is closed as `MIXED` and was not promoted.

## System Map

```text
Video sources
  |
  +-- Motion analysis
  |     |
  |     +-- frame streaming
  |     +-- optional ROI
  |     +-- background subtraction
  |     +-- contour/activity metrics
  |     +-- heatmap and annotated video
  |
  +-- Dataset tooling
  |     |
  |     +-- deterministic frame extraction
  |     +-- Roboflow export inspection/preparation
  |     +-- dataset.yaml-aware validation/statistics
  |     +-- label/image/hash/source metadata
  |
  +-- Detection
        |
        +-- registered detector specs
        +-- SHA256 model identity checks
        +-- canonical inference defaults
        +-- run provenance metadata
        +-- external/reference evaluation
```

## Motion Analysis

The motion pipeline is intentionally independent from ant detection. It uses OpenCV video IO, optional ROI validation, background subtraction, contour filtering, heatmap accumulation, and JSON metrics. It produces exploratory motion artifacts, not biological ant counts.

Main components:

- CLI `analyze`
- `VideoAnalyzer`
- `MotionDetector`
- `MetricsAggregator`
- `HeatmapAccumulator`
- annotated video and optional motion-mask writers

## Dataset Tooling

Dataset tooling is built around explicit source roles and reproducible manifests. Frame extraction uses deterministic names and refuses stale extractor-owned outputs unless `--overwrite` is explicitly requested.

YOLO dataset validation and statistics resolve `path`, `train`, and `val` from `dataset.yaml`; they do not assume local hardcoded split directories. Empty label files are valid zero-ant images, while missing labels are errors. When real image dimensions are known, label validation allows only the frozen `0.5` physical-pixel boundary tolerance for export rounding and never clamps or mutates annotations.

Roboflow preparation inspects `train`, `valid`, and `test`, but only prepares `train` and `valid` by default. Using test records as a training source requires the explicit `--allow-test-as-training-source` override.

## Detector Champion And Provenance

Detector identity is explicit and SHA-based. `--champion` resolves the current champion spec:

```text
experiment: ants_v3_mixedscale_e01
status: CURRENT_CHAMPION
canonical inference: conf=0.25, iou=0.70, imgsz=640, end2end=false
class: 0 ant
```

`ants_v4_domain_e01` introduced real-camera own-domain exposure but was not promoted because its final outcome was mixed. Historical model cards remain the source of detailed experiment-specific results.

Video detector runs record:

- source path and SHA256
- model path, SHA256, and known detector identity if registered
- canonical inference parameters and overrides
- class map validation
- environment metadata from `formicai.utils.environment`
- output paths and detection counts

## Evaluation

Ultralytics P/R/mAP remains the primary standard detection metric when evaluating YOLO datasets.

Custom diagnostics in `public_ants.py` are explicitly secondary:

- center-based localization metrics for annotation-style mismatch inspection
- fixed-IoU50 one-to-one localization diagnostics
- density analysis over already-computed per-frame counts

AUDIT-04 replaced greedy diagnostic matching with deterministic maximum-cardinality bipartite matching. This fixes undercounting of valid one-to-one assignments without changing Ultralytics metrics or claiming a maximum-total-IoU optimizer.

External evaluation writes generated evaluation YAML and validation run outputs outside frozen external-test directories. Existing external-test `dataset.yaml` files are not overwritten.

## Experiment Integrity

The audit series adds layered protections:

- AUDIT-01: holdout/test protection and read-only external evaluation boundaries
- AUDIT-02: canonical detector selection, model SHA provenance, protected output paths
- AUDIT-03: dataset correctness, image-aware label validation, stale extraction protection
- AUDIT-04: evaluation matching correctness, shared inference validation, CI, environment metadata, and future experiment record policy

Small future experiment records belong in `experiments/`. They should include protocol, source/dataset/model hashes, selection metadata, environment metadata, dependency lock snapshots, and final summaries. Large data, videos, images, labels, model weights, and generated artifacts stay ignored.

## Future Work

Tracking remains future work after detector/generalization quality is stable. This repository does not currently implement Kalman filters, ByteTrack, DeepSORT, ReID, or colony-level identity tracking.
