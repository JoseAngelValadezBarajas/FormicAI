# ants_v3_mixedscale_e01 Model Card

## Status

`ants_v3_mixedscale_e01` is the current FormicAI experimental detector champion.

```text
status: CURRENT CHAMPION
model: YOLO26n
selected checkpoint: best.pt / selected.pt
selected represented epoch: 38
weights: artifacts/models/ants_v3_mixedscale_e01/selected.pt
sha256: 424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf
```

The weight file is a local artifact and is intentionally excluded from Git.

## Lineage

`ants_v3_mixedscale_e01` extends the `ants_v2_diverse_e01` dataset with targeted mixed-scale adult-ant examples:

- AntsNet curated relabel pack.
- ant2 own-colony relabel pack with queen and large-worker scale.
- FormicAI tight-body boxes, single class `0: ant`.

It was selected using development validation only. No public reference benchmark or ant4 result was used for checkpoint selection.

## Dataset

```text
dataset: datasets/prepared/ants_v3_mixedscale_e01
train images: 222
train annotations: 3731
validation images: 90
validation annotations: 1503
total annotations: 5234
train SHA256: 3bbed9fc02c77d8f5247c9256abd6a3faadb553c3265c2201c4e50495c13e09b
validation SHA256: 71d3ad221967ee120a93333e5c7a3eb0a9392ed71788f6cba098fee052171adf
full dataset SHA256: 820b19f114e17220ec70ee43362d84da8c5a1e7b4c632e8b84f289b4cc7835a2
manifest SHA256: f5c262074bc6c69d977fb792d544ef1f3202a111594017fa1bfa0a309e434939
```

Training sources:

- Seq0002
- Seq0003
- Seq0004
- Seq0007
- Seq0008
- Seq0009
- ant2
- AntsNet

Development validation sources:

- Seq0005
- Seq0010
- own_colony_val_001

## Canonical Inference

```text
end2end: false
confidence: 0.25
NMS IoU: 0.70
imgsz: 640
```

## Development Validation

| Source | P | R | mAP50 | mAP50-95 | Predictions | GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seq0005 | 0.9374 | 0.9482 | 0.9641 | 0.5228 | 349 | 300 |
| Seq0010 | 0.8704 | 0.7575 | 0.7793 | 0.3671 | 1167 | 1068 |
| own_colony_val_001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 135 |
| Overall | 0.8911 | 0.7279 | 0.7400 | 0.3623 | 1516 | 1503 |

Macro-source metrics:

| Metric | Value |
| --- | ---: |
| Macro source precision | 0.6026 |
| Macro source recall | 0.5686 |
| Macro source mAP50 | 0.5811 |
| Macro source mAP50-95 | 0.2966 |

`own_colony_val_001` remains a development/pathological failure source. The model produced zero predictions there.

## Clean Own-Colony Reference

`own_colony_val_002` was evaluated only after checkpoint selection. It is now an observed own-colony reference for future experiments.

| Source | P | R | mAP50 | mAP50-95 | Predictions | GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| own_colony_val_002 | 0.9444 | 0.5862 | 0.5833 | 0.3226 | 19 | 29 |

## ANT4 Final Test

`ant4` is now an observed own-colony final test and is no longer unbiased for future v5+ experiments.

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| ants_v1_baseline | 0.1092 | 0.0741 | 0.0098 | 0.0016 |
| ants_v2_diverse_e01 | 0.4396 | 0.3527 | 0.1738 | 0.0459 |
| ants_v3_mixedscale_e01 | 0.8664 | 0.8466 | 0.8200 | 0.4434 |
| ants_v4_domain_e01 | 0.8680 | 0.7654 | 0.7511 | 0.3786 |

`ants_v3_mixedscale_e01` remains the current champion because it outperformed v4 on ANT4 final-test recall, mAP50, and mAP50-95.

## Public Reference Status

`Seq0001` and `Seq0006` are now observed public references, not unbiased model-selection data. No frozen v3 public-reference benchmark result was recorded before the v4 closure.

## Limitations

- Persistent zero-prediction failure on `own_colony_val_001`.
- Ant4 is now observed and cannot be reused as an unbiased final test.
- Public ANTS references are observed references, not fresh external validation.
- Performance remains sensitive to real-camera lighting, tube/plastic reflections, occlusion, and crowded small workers.
