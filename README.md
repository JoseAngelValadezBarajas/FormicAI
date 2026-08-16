# FormicAI

FormicAI is a personal project for observing, recording, and analyzing ant colony videos over time. The long-term vision includes computer vision, tracking, data analysis, cloud services, multimodal AI, scientific literature retrieval, and IoT telemetry.

This repository currently contains:

- **FormicAI Vision v0.1**: motion analysis, activity metrics, heatmap, annotated video, and optional motion-mask video.
- **FormicAI Vision v0.2**: dataset preparation, `ants_v1_baseline` object detection, frozen internal validation, and frozen external ANTS benchmark evaluation.
- **ants_v3_mixedscale_e01**: current experimental detector champion after mixed-scale queen/large-worker and AntsNet/ant2 relabeling.
- **ants_v4_domain_e01**: closed real-camera own-domain exposure experiment; final result `MIXED`, not promoted.

v0.1 detects moving regions, not ants. v0.2 adds a first experimental ant detector, but it is still a small baseline and not a general-purpose biological measurement system.

Current experimental detector champion:

```text
ants_v3_mixedscale_e01
```

## Requirements

- Python 3.11+
- A local video file, for example `samples/ants.mp4`

## Installation

```bash
git clone <repo-url>
cd formicai

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Unix/macOS:

```bash
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For future YOLO training/inference work:

```bash
python -m pip install -e ".[ml,dev]"
```

Ultralytics is an external dependency. Review its current licensing terms before commercial deployment.

## Add a Sample Video

Place an `.mp4` video inside `samples/`:

```text
samples/ants.mp4
```

Large video files in `samples/` are ignored by Git.

## FormicAI Vision v0.1: Motion Analysis

Using the project entry point:

```bash
python -m formicai analyze samples/ants.mp4
```

Using the convenience script:

```bash
python main.py samples/ants.mp4
```

The default motion filter scales the minimum contour area to the ROI size. For a 1920x1080 full-frame video, the default is about `207` pixels; for a 320x240 video, it is about `8` pixels.

With a region of interest:

```bash
python main.py samples/ants.mp4 --roi 100,200,900,700
```

The ROI format is:

```text
x,y,width,height
```

If no ROI is provided, the full frame is analyzed.

By default, the first `30` frames are used to warm up the background model. They count as processed frames, but they do not participate in activity statistics. You can change this:

```bash
python main.py samples/ants.mp4 --warmup-frames 30
```

To write a video showing the final binary motion mask used for scoring, heatmap accumulation, and detections:

```bash
python -m formicai analyze samples/ants.mp4 --write-motion-mask
```

## Outputs

By default, files are written to `output/`:

- `output/analyzed.mp4`: original video with moving regions boxed and basic frame metrics overlaid.
- `output/activity_heatmap.png`: accumulated motion heatmap for the full video.
- `output/metrics.json`: video metadata and aggregate activity metrics.
- `output/motion_mask.mp4`: optional final binary motion mask video, only written with `--write-motion-mask`.

`metrics.json` includes:

- `processedFrames`: frames read and processed, including warmup.
- `warmupFrames`: frames used to stabilize the background model.
- `scoredFrames`: frames included in activity statistics.

## Activity Score

The activity score is intentionally simple in v0.1:

```text
motion pixels after noise filtering and contour filtering / total ROI pixels
```

The value is clamped between `0.0` and `1.0`. Aggregate activity statistics are calculated only from `scoredFrames`, not warmup frames.

Warmup frames are processed but excluded from `averageActivityScore`, `maxActivityScore`, `peakActivityTimestampSeconds`, `averageActiveRegions`, and `maxActiveRegions`.

This metric is useful for rough activity comparison, but it is not a biological ant count or behavior classifier.

## Tuning Motion Detection

Small moving regions may require lower contour-area thresholds. Try explicit values while inspecting `analyzed.mp4`, `activity_heatmap.png`, and optionally `motion_mask.mp4`:

```bash
python -m formicai analyze samples/ants.mp4 --min-motion-area 30 --write-motion-mask
python -m formicai analyze samples/ants.mp4 --min-motion-area 75 --write-motion-mask
```

Smaller values are more sensitive and may add noise. Larger values reduce noise but may miss small motion. These are experimental parameters, not scientifically validated ant-detection thresholds.

If the video is noisy, increase filtering:

```bash
python -m formicai analyze samples/ants.mp4 --background-var-threshold 64 --min-motion-area 100
```

If subtle motion is being missed, lower those values.

## Tests

```bash
pytest
```

The tests focus on deterministic behavior and include a smoke test that generates a temporary video and runs the full `VideoAnalyzer` pipeline.

## FormicAI Vision v0.2: Ant Detection

v0.2 closes with dataset preparation tooling, a small proof-of-concept object detector, internal validation on `ant2`, and a frozen public ANTS benchmark. It does not create fake labels from motion regions.

Extract candidate frames from a real local video:

```bash
python -m formicai dataset extract-frames samples/ant2.mp4 --output datasets/raw/ant2 --every-seconds 0.25 --roi 60,180,600,700
```

The first detector uses one class:

```text
0 ant
```

Annotate the extracted images manually with CVAT, Label Studio, Roboflow, or Ultralytics Platform, then prepare the export. The preparer preserves YOLO detection rows and can convert valid YOLO segmentation polygons into tight bounding boxes. See [datasets/README.md](datasets/README.md).

After labels are prepared:

```bash
python -m formicai dataset prepare-roboflow exported_dataset --output datasets/prepared/ants_v1 --train-fraction 0.8 --gap-count 1
python -m formicai dataset validate datasets/prepared/ants_v1
python -m formicai dataset stats datasets/prepared/ants_v1
```

The initial baseline uses Ultralytics YOLO26 nano (`yolo26n.pt`) with transfer learning. This is a proof of concept from temporally separated frames of one source video, not evidence of generalization to new colonies, cameras, lighting, or videos.

Run video detection with global-frame JSONL boxes:

```bash
python -m formicai detect-video samples/ant2.mp4 --model artifacts/models/ants_v1_baseline/weights/best.pt --roi 60,180,600,700
```

### FormicAI Vision v0.2 Milestone

v0.2 is closed as an experimental baseline milestone:

- `ants_v1_baseline` is frozen for reporting.
- `best.pt` must not be modified in-place.
- Public ANTS `Seq0001` and `Seq0006` are frozen benchmarks and must not participate in `ants_v1` training.
- Dataset raw files, public images/labels, model weights, generated videos, virtual environments, API keys, secrets, and temporary files are intentionally kept out of Git.

If future models train on other ANTS sequences, `Seq0001` and `Seq0006` must no longer be described as a completely external-dataset benchmark. They become a held-out sequence benchmark from the same public dataset, assuming those two sequences remain excluded from training.

### ants_v1_baseline

Model card: [docs/model_cards/ants_v1_baseline.md](docs/model_cards/ants_v1_baseline.md)

Frozen model:

```text
artifacts/models/ants_v1_baseline/weights/best.pt
SHA256: 4C91C700BCE3EF56F8198B600A4828686CEA2A0937C30F794B3F1B3023662A70
```

Final inference configuration:

```text
Architecture: YOLO26n
end2end: false
NMS IoU: 0.70
confidence: 0.25
imgsz: 640
```

Internal `ant2` validation:

| Metric | Value |
| --- | ---: |
| Precision | 0.8120 |
| Recall | 0.8542 |
| mAP50 | 0.8104 |
| mAP50-95 | 0.4821 |

Frozen external ANTS benchmark:

| Sequence | Standard P | Standard R | mAP50 | mAP50-95 | Center Precision | Center Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seq0001 | 0.3631 | 0.2969 | 0.1074 | 0.0240 | 0.8547 | 0.6672 |
| Seq0006 | 0.5594 | 0.4576 | 0.2868 | 0.0717 | 0.7892 | 0.6217 |

Center-based metrics are diagnostic localization metrics only. They count whether a prediction center lands inside a ground-truth box with one-to-one matching. They do not replace standard mAP, precision, or recall.

Annotation-style mismatch:

```text
FormicAI training: tight body boxes
ANTS Seq0001: fixed 94x94 boxes
ANTS Seq0006: fixed 64x64 boxes
```

Public benchmark provenance remains documented under [datasets/public/ants_mendeley/README.md](datasets/public/ants_mendeley/README.md): ANTS--ant detection and tracking, DOI `10.17632/9ws98g4npw.4`, CC0 1.0.

### ants_v2_diverse_e01

Model card: [docs/model_cards/ants_v2_diverse_e01.md](docs/model_cards/ants_v2_diverse_e01.md)

`ants_v2_diverse_e01` is frozen as a dataset-diversity experiment, not as a universal replacement for `ants_v1_baseline`.

```text
selected checkpoint: epoch40.pt / selected.pt
selected SHA256: 4e37c15340b6ae1636eecc5b0d796fce9123fb1e987901a8b01bde46a76782e6
dataset content SHA256: 8b7406900b49329e552bd002f9ceb7c4ea3aa3b64a076005ef18b6f9daa8990f
```

It strongly improves public ANTS external results, but the final ant2 regression check is a `MATERIAL REGRESSION`. ant2 is therefore no longer an untouched final benchmark for future v3 work. It may be used as a training candidate or diagnostic/regression reference, but not as a new unbiased final test.

The next proposed dataset design is documented under [datasets/ants_v3_design](datasets/ants_v3_design): keep the public ants_v2 train base, add a small own-colony mixed-scale ant2 pack, add verified hard negatives, keep ant4 frozen as an own-colony final-test candidate, and collect a new own-colony validation video.

### ants_v3_mixedscale_e01

Model card: [docs/model_cards/ants_v3_mixedscale_e01.md](docs/model_cards/ants_v3_mixedscale_e01.md)

`ants_v3_mixedscale_e01` is the current experimental detector champion.

```text
selected checkpoint: best.pt / selected.pt
selected SHA256: 424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf
status: CURRENT CHAMPION
```

It added targeted mixed-scale adult-ant examples from AntsNet and ant2 while preserving the single-class FormicAI policy:

```text
0 ant
tight visible adult ant body box
```

### ants_v4_domain_e01

Model card: [docs/model_cards/ants_v4_domain_e01.md](docs/model_cards/ants_v4_domain_e01.md)

`ants_v4_domain_e01` tested real-camera own-domain exposure and is formally closed:

```text
selected SHA256: 689249476dc61912050a5106ac712c9e24dd494da892a2d113f72ba20198e0e9
final classification: MIXED
promoted to champion: false
current champion: ants_v3_mixedscale_e01
```

Detector lineage:

| Version | Role |
| --- | --- |
| v1 | Original own-domain baseline |
| v2 | Public/diverse generalization |
| v3 | Mixed-scale + queen/large + AntsNet + ant2 |
| v4 | Real-camera own-domain exposure |

ANT4 final-test result:

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.1092 | 0.0741 | 0.0098 | 0.0016 |
| v2 | 0.4396 | 0.3527 | 0.1738 | 0.0459 |
| v3 | 0.8664 | 0.8466 | 0.8200 | 0.4434 |
| v4 | 0.8680 | 0.7654 | 0.7511 | 0.3786 |

v4 improved clean `own_colony_val_002` versus v3 and preserved public development sources, but it regressed versus v3 on ANT4, still failed `own_colony_val_001`, and regressed in standard public-reference metrics versus v2. The result remains `MIXED`.

After observation, source roles are:

```text
Seq0001 / Seq0006: OBSERVED PUBLIC REFERENCES
own_colony_val_001: DEVELOPMENT / PATHOLOGICAL FAILURE SOURCE
own_colony_val_002: OBSERVED OWN-COLONY REFERENCE for future experiments
ant4: OBSERVED OWN-COLONY FINAL TEST, no longer unbiased for v5+
```

Weights, datasets, videos, labels, and analysis artifacts remain intentionally excluded from Git. Reproducibility is preserved through model cards, manifests, metadata, and SHA256 hashes.

## Current Limitations

- Motion analysis still detects moving regions, not ants.
- The first ant detector is trained on a very small single-video dataset and should be treated as experimental.
- The detector struggles with small ants, partial visibility, crowded groups, complex backgrounds, shadows, rocks, and nest entrances.
- External performance is affected by domain shift and annotation policy mismatch.
- Background subtraction can be sensitive to lighting changes, camera shake, reflections, and early video frames while the background model stabilizes.
- No tracking, behavior classification, backend, frontend, cloud, Gemini, RAG, or IoT features are included yet.

## Next Steps

- v0.3: tracking and trajectories.
- v0.4: behavioral metrics.
- v0.5+: backend, frontend, cloud, AI-assisted reports, research/RAG, and IoT telemetry.
