# FormicAI Datasets

This directory is for local dataset preparation. Generated images, labels, prepared datasets, and model artifacts are ignored by Git.

## Class Definition

FormicAI Vision v0.2 starts with one detection class:

```text
0 ant
```

Do not label larvae, eggs, pupae, food, cotton, reflections, or plastic as ants.

## Extract Candidate Frames

Use a sparse interval so the first dataset has visual diversity instead of hundreds of near-duplicates:

```bash
python -m formicai dataset extract-frames samples/ant2.mp4 --output datasets/raw/ant2 --every-seconds 0.25 --roi 60,180,600,700
```

For `ant2.mp4`, `0.25` seconds samples roughly 4 frames per second and should produce around 60-70 candidate images.

Each image name includes the source video stem, frame index, and timestamp. `metadata.json` records source video metadata and the ROI used.

## Manual Annotation

Use an external annotation tool instead of a custom GUI:

- CVAT
- Label Studio
- Roboflow
- Ultralytics Platform

Export annotations in YOLO detection format:

```text
class x_center y_center width height
```

Coordinates must be normalized between `0` and `1`. Images with no visible ants are valid hard negatives and should have an empty `.txt` label file.

Roboflow can export multiple YOLO variants. The preparer preserves YOLO detection rows and converts valid YOLO segmentation polygons to tight bounding boxes without padding. Invalid polygons still fail validation and are not silently fixed.

Inspect a Roboflow export before preparing it:

```bash
python -m formicai dataset inspect-export exported_dataset
```

Prepare a temporal train/val split from the export:

```bash
python -m formicai dataset prepare-roboflow exported_dataset --output datasets/prepared/ants_v1 --train-fraction 0.8 --gap-count 1
```

## Prepared YOLO Layout

After annotation, arrange/export the dataset like this:

```text
datasets/prepared/ants_v1/
  images/
    train/
    val/
  labels/
    train/
    val/
  dataset.yaml
```

`dataset.yaml`:

```yaml
path: .
train: images/train
val: images/val
names:
  0: ant
```

Because early frames come from the same source video, avoid random frame-by-frame splitting. Prefer a temporal block split, such as earlier timestamps for train and later timestamps for validation. A single video does not provide truly independent validation; later datasets should split by source video.

## Validate And Inspect

```bash
python -m formicai dataset validate datasets/prepared/ants_v1
python -m formicai dataset stats datasets/prepared/ants_v1
```

Training should wait until validation passes with real manual labels.

## External Test Frames

Use `datasets/external_test/` for videos held out from training. These frames are for manual annotation and later evaluation of an existing model, not for fitting new weights.

```bash
python -m formicai dataset extract-frames samples/ant4.mp4 --output datasets/external_test/ant4/images --count 20 --roi 40,122,400,472
python -m formicai dataset extract-frames samples/ant3.mp4 --output datasets/external_test/ant3/images --count 15 --roi 40,122,400,472
```

After Roboflow labels are placed in each matching `labels/` folder:

```bash
python -m formicai dataset evaluate-external datasets/external_test --model artifacts/models/ants_v1_baseline/weights/best.pt --conf 0.25 --iou 0.70 --imgsz 640 --device 0 --no-end2end
```

The external evaluation command runs validation only and reports precision, recall, mAP50, and mAP50-95 per video.
