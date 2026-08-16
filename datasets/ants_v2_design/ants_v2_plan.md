# ants_v2 Dataset Design Proposal

This is a design/audit phase only. No model was trained, no pseudo-labeling was performed, and `ants_v1_baseline` remains frozen.

## Conceptual Dataset Shape

```text
datasets/prepared/ants_v2/
  train/
    images/
    labels/
  val/
    images/
    labels/
  metadata/
  dataset.yaml
```

Do not materialize this dataset until the annotation policy and split are approved.

## Frozen Benchmark

`Seq0001` and `Seq0006` are frozen external benchmark sequences. They must not enter ants_v2 training, validation, hyperparameter tuning, confidence tuning, augmentation tuning, or model selection.

## Initial Recommendation

Use source-level split where possible. Reserve at least one indoor and one outdoor source for validation. Use selected non-frozen ANTS sequences only after resolving annotation policy. Local `ant3` and `ant4` are useful future external/stress tests after manual labels, but they are currently unlabeled.

## First Experiment Size

Target a manageable first experiment:

- train: 600-900 images
- val: 200-350 images
- approximate annotations: 5k-15k ants depending on outdoor sequence density

Prefer diversity per annotation over raw volume.

## Sampling

Use fixed temporal sampling per selected source, then manual review:

- indoor public candidates: 75-150 frames/source
- outdoor public candidates: 100-200 frames/source
- validation held-out sources: 100-200 frames/source

Avoid random frame splits. If a source must be split, use temporal blocks with a gap.

## Hard Negatives

Hard negatives are recommended later for rocks, shadows, nest entrances, food/substrate, and empty textured backgrounds. Do not generate them yet.
