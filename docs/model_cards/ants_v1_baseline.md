# ants_v1_baseline Model Card

## Status

`ants_v1_baseline` is the frozen FormicAI Vision v0.2 proof-of-concept detector. Do not train it further, fine-tune it, or modify `best.pt` in place.

```text
weights: artifacts/models/ants_v1_baseline/weights/best.pt
sha256: 4C91C700BCE3EF56F8198B600A4828686CEA2A0937C30F794B3F1B3023662A70
```

The weight file is a local artifact and is intentionally excluded from Git.

## Intended Use

This model is intended for experimental ant localization on the early FormicAI sample videos. It is not a production detector and should not be treated as a scientific ant counter without additional validation.

## Training Data Boundary

The baseline was built from the internal `ant2` workflow with tight ant body boxes. Public ANTS `Seq0001` and `Seq0006` are frozen benchmark sequences and must not participate in `ants_v1` training.

If future models train on other ANTS sequences, `Seq0001` and `Seq0006` stop being a completely external-dataset benchmark and become a held-out sequence benchmark from the same public dataset, provided they remain excluded from training.

## Inference Configuration

```text
architecture: YOLO26n
end2end: false
NMS IoU: 0.70
confidence: 0.25
imgsz: 640
```

## Internal Validation

`ant2` internal validation:

| Metric | Value |
| --- | ---: |
| Precision | 0.8120 |
| Recall | 0.8542 |
| mAP50 | 0.8104 |
| mAP50-95 | 0.4821 |

## Frozen External Benchmark

Public ANTS benchmark provenance:

```text
dataset: ANTS--ant detection and tracking
DOI: 10.17632/9ws98g4npw.4
license: CC0 1.0
frozen sequences: Seq0001, Seq0006
```

| Sequence | Standard P | Standard R | mAP50 | mAP50-95 | Center Precision | Center Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seq0001 | 0.3631 | 0.2969 | 0.1074 | 0.0240 | 0.8547 | 0.6672 |
| Seq0006 | 0.5594 | 0.4576 | 0.2868 | 0.0717 | 0.7892 | 0.6217 |

Center-based metrics are diagnostic localization metrics only. They count a match when the prediction center lies inside the ground-truth box, using one-to-one matching. They do not replace standard precision, recall, mAP50, or mAP50-95.

## Annotation Policy Mismatch

```text
FormicAI training: tight body boxes
ANTS Seq0001: fixed 94x94 boxes
ANTS Seq0006: fixed 64x64 boxes
```

Low IoU on ANTS can reflect annotation-style mismatch rather than a complete missed ant. Standard mAP remains the canonical metric and is reported unchanged.

## Limitations

- Small ants.
- Partial visibility.
- Crowded groups.
- Complex backgrounds.
- Shadows, rocks, and nest entrances.
- Domain shift across cameras, lighting, species, and environments.
- Annotation policy mismatch between tight boxes and fixed-size ANTS boxes.

## Recommended Next Step

For `ants_v2`, expand training diversity before threshold tuning: include more backgrounds, lighting conditions, scales, crowded scenes, and tight body-box annotations consistent with FormicAI's intended box policy.
