# External Test Sets

These folders hold temporally distributed frames from videos that were not used to train the current baseline model.

The intent is evaluation only:

```text
datasets/external_test/
  ant3/
    images/
    labels/
    metadata.json
    dataset.yaml
  ant4/
    images/
    labels/
    metadata.json
    dataset.yaml
```

`images/` and `labels/` are local generated data and are ignored by Git.

## Current Frame Selection

The first external test batch uses cropped ROI images with the same scaled ROI used for ant3/ant4 inference:

```text
ROI: 40,122,400,472
Class: 0 ant
```

Extraction commands:

```bash
python -m formicai dataset extract-frames samples/ant4.mp4 --output datasets/external_test/ant4/images --count 20 --roi 40,122,400,472
python -m formicai dataset extract-frames samples/ant3.mp4 --output datasets/external_test/ant3/images --count 15 --roi 40,122,400,472
```

## Manual Roboflow Annotation

Upload the images from each `images/` folder to Roboflow and annotate only:

```text
0 ant
```

Do not label larvae, eggs, pupae, food, cotton, reflections, or plastic as ants.

After export, place YOLO `.txt` files back into the matching per-video `labels/` folder. Label filenames must match image stems exactly. Empty `.txt` files are valid for frames with no visible ants.

## Evaluate Without Training

After all labels exist:

```bash
python -m formicai dataset evaluate-external datasets/external_test \
  --model artifacts/models/ants_v1_baseline/weights/best.pt \
  --output artifacts/analysis/external_test_metrics.json \
  --conf 0.25 \
  --iou 0.70 \
  --imgsz 640 \
  --device 0 \
  --no-end2end
```

This runs validation only. It does not train or modify model weights.
