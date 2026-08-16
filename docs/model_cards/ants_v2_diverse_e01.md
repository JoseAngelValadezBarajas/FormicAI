# ants_v2_diverse_e01 Model Card

## Status

`ants_v2_diverse_e01` is a frozen experiment checkpoint selected from the first diverse public ANTS training set. It must not be modified in place, threshold-tuned on external benchmarks, fine-tuned, or treated as a drop-in replacement for `ants_v1_baseline`.

```text
experiment: ants_v2_diverse_e01
model: YOLO26n
selected checkpoint: epoch40.pt / selected.pt
weights: artifacts/models/ants_v2_diverse_e01/weights/selected.pt
sha256: 4e37c15340b6ae1636eecc5b0d796fce9123fb1e987901a8b01bde46a76782e6
```

The weight file is a local artifact and is intentionally excluded from Git.

## Dataset

```text
train images: 155
train annotations: 3364
val images: 60
val annotations: 1368
total annotations: 4732
dataset content SHA256: 8b7406900b49329e552bd002f9ceb7c4ea3aa3b64a076005ef18b6f9daa8990f
```

Training sources:

- Seq0002
- Seq0003
- Seq0004
- Seq0007
- Seq0008
- Seq0009

Validation sources:

- Seq0005
- Seq0010

## Selection

The selected checkpoint is `epoch40.pt`, copied/referenced as `selected.pt`. Selection used only the approved validation set with the canonical FormicAI inference policy:

```text
end2end: false
confidence: 0.25
NMS IoU: 0.70
imgsz: 640
```

## External Benchmark

Standard mAP remains the benchmark metric. Center-based localization metrics are diagnostic only and do not replace mAP.

| Sequence | v1 mAP50 | v2 mAP50 | v1 Center Recall | v2 Center Recall |
| --- | ---: | ---: | ---: | ---: |
| Seq0001 | 0.1074 | 0.4923 | 0.6672 | 0.9997 |
| Seq0006 | 0.2868 | 0.7164 | 0.6217 | 0.9722 |

## ant2 Regression Check

The original ant2 own-colony domain was checked after frozen external evaluation. ant2 was not used for v2 model selection, threshold tuning, retraining, or fine-tuning.

| Model | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| ants_v1_baseline | 0.8120 | 0.8542 | 0.8104 | 0.4821 |
| ants_v2_diverse_e01 selected | 0.0225 | 0.0417 | 0.0014 | 0.0005 |

Regression classification:

```text
MATERIAL REGRESSION
```

## Conclusion

`ants_v2_diverse_e01` strongly improves external/public-domain generalization but is not a drop-in replacement for `ants_v1_baseline` because it materially regresses on the original ant2/own-colony domain, especially large queen scale and local visual distractors.

This regression should be carried forward explicitly into `ants_v3_mixedscale_e01` dataset design.

## v3 Implication

ant2 is no longer an untouched final benchmark for v3 because its results have been observed and will inform the next dataset. It may be used as a training candidate and diagnostic/regression reference, but not as a new unbiased final test.

The proposed next step is a mixed-scale dataset: keep the public ants_v2 TRAIN base, add a small own-colony ant2 positive pack for queen/large-worker scale, add verified hard negatives for nest materials/glare, and keep ant4 frozen as an own-colony final-test candidate.
