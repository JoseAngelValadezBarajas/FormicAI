# ANTS Mendeley External Benchmark

This directory is reserved for the public ANTS ant detection and tracking dataset used as external evaluation data for FormicAI Vision.

These sequences are external benchmark data. They MUST NOT be copied into `datasets/prepared/ants_v1`, included in training, used for fine-tuning, or used to create new model weights.

## Source

- Dataset name: ANTS--ant detection and tracking
- Mendeley Data ID: `9ws98g4npw`
- Version used: `4`
- DOI: `10.17632/9ws98g4npw.4`
- Dataset page: https://data.mendeley.com/datasets/9ws98g4npw/4
- License: CC0 1.0 Public Domain Dedication
- Contributor listed by Mendeley: xiaoyan Cao

Official metadata was fetched from:

```text
https://data.mendeley.com/public-api/datasets/9ws98g4npw/snapshot/4
https://data.mendeley.com/public-api/datasets/9ws98g4npw/files?folder_id=root&version=4
https://data.mendeley.com/public-api/datasets/9ws98g4npw/versions
https://data.mendeley.com/api/datasets-v2/datasets/9ws98g4npw/zip?version=4
```

## Downloaded File

Only one archive has been downloaded for the initial structure inspection:

```text
raw/Ant_dataset.zip
```

Mendeley file metadata:

```text
filename: Ant_dataset.zip
size: 751242824 bytes
sha256: 2f1df42e7faf4823723b30af876b4f79ade92a42f6fb5e2b850144cd989c375c
content type: application/x-zip-compressed
last modified: 2021-10-27T03:54:12.549Z
```

The local file size and SHA-256 were verified after download.

`OriginalSequences.zip` was discovered in the official file list but was not downloaded during this phase because `Ant_dataset.zip` contains the selected image sequences and ground truth annotations needed for inspection.

## Local Structure

```text
datasets/public/ants_mendeley/
  raw/
    Ant_dataset.zip
    Ant_dataset/
      IndoorDataset/
        Seq0001Object10Image94/
      OutdoorDataset/
        Seq0006Object21Image64/
  prepared/
    Seq0001/
      images/
      labels/
    Seq0006/
      images/
      labels/
  metadata/
```

Only `Seq0001Object10Image94` and `Seq0006Object21Image64` were extracted from the archive for this initial benchmark inspection.

## Structure Inspection

The real dataset structure uses MOT-style folders:

```text
det/det.txt
gt/gt.txt
img/*.jpg
seqinfo.ini
```

Images are JPEG files named with six-digit, 1-based frame numbers, such as `000001.jpg`.

The annotation rows are not YOLO. Each `gt.txt` and `det.txt` row has seven comma-separated values:

```text
frame,id,bbox_left,bbox_top,bbox_width,bbox_height,confidence_or_gt_flag
```

Coordinates are pixel coordinates. Bounding boxes are `[x, y, width, height]` with `x,y` at the top-left corner.

For `det.txt`, the identity column is `-1`. For `gt.txt`, the identity column is the ant trajectory ID. The seventh column is a confidence score for detections and acts as a consider/ignore flag for ground truth and results. In the inspected `Seq0001` and `Seq0006` ground truth files, all flag values are `1`.

Inspection output:

```text
metadata/inspection_seq0001_seq0006.json
```

## Selected External Evaluation Sequences

`Seq0001`:

```text
environment: indoor
raw folder: IndoorDataset/Seq0001Object10Image94
images: 351
resolution: 1920x1080
fps: 25
ground-truth rows: 3510
unique trajectory IDs: 10
box size: 94x94
flags: [1]
```

`Seq0006`:

```text
environment: outdoor
raw folder: OutdoorDataset/Seq0006Object21Image64
images: 600
resolution: 1280x720
fps: 30
ground-truth rows: 11178
unique trajectory IDs: 73
box size: 64x64
flags: [1]
```

Known inconsistency:

```text
Seq0006 folder name: Seq0006Object21Image64
Seq0006 seqinfo.ini name: Seq0006Object10Image64
```

The ground truth has 73 unique IDs, and the paper's table reports 73 ants for Seq0006.

## Conversion Status

Phase 2 performed deterministic conversion from valid `gt/gt.txt` rows to single-class YOLO detection labels:

```text
0 x_center y_center width height
```

All classes must map to:

```text
0 ant
```

Conversion must preserve full-frame evaluation. Do not apply the private-video ROI used for `ant2`, `ant3`, or `ant4`.

Generated files:

```text
prepared/Seq0001/images/
prepared/Seq0001/labels/
prepared/Seq0006/images/
prepared/Seq0006/labels/
metadata/seq0001.yaml
metadata/seq0006.yaml
metadata/tracking_seq0001.jsonl
metadata/tracking_seq0006.jsonl
metadata/conversion_report.json
```

Ground-truth visual sanity montages:

```text
artifacts/analysis/public_ants/Seq0001_ground_truth_montage.jpg
artifacts/analysis/public_ants/Seq0006_ground_truth_montage.jpg
```

No model inference or metric evaluation was performed in Phase 2.

## Phase 3 Frozen Baseline Evaluation

Phase 3 evaluates the frozen `ants_v1_baseline` detector on the public ANTS sequences without training, fine-tuning, ROI cropping, or ground-truth resizing.

```bash
python -m formicai dataset evaluate-ants-benchmark \
  --root datasets/public/ants_mendeley \
  --model artifacts/models/ants_v1_baseline/weights/best.pt \
  --output artifacts/analysis/public_ants/external_benchmark_metrics.json \
  --predictions-dir artifacts/analysis/public_ants/predictions \
  --video-dir output/public_ants \
  --conf 0.25 \
  --iou 0.70 \
  --imgsz 640 \
  --device 0 \
  --no-end2end \
  --sequences Seq0001 Seq0006
```

Outputs:

```text
artifacts/analysis/public_ants/external_benchmark_metrics.json
artifacts/analysis/public_ants/predictions/
output/public_ants/Seq0001_formicai.mp4
output/public_ants/Seq0006_formicai.mp4
```

The consolidated JSON keeps Ultralytics standard metrics separate from `center-based localization metrics`. Center-based matching counts a detection when the prediction center lies inside the fixed-size ANTS ground-truth box, with one-to-one matching.

Seq0006 contains one exact duplicate YOLO label row in `000268.txt`, representing two track IDs with the same fixed box. Ultralytics removes that duplicate during standard validation cache creation; the source ground-truth row count is preserved in the benchmark JSON.
