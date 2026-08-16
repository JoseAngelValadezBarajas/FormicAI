# AUDIT-01 Experimental Integrity

## Baseline

```text
repository: JoseAngelValadezBarajas/FormicAI
baseline tag: ants-v4-domain-e01-final
baseline commit: 087c4cf8ce41790160de409fc0a1f0ab807ae42c
branch: audit/formicai-code-audit
```

AUDIT-01 is a code-safety and experimental-integrity change only. It did not train, evaluate, tune, or select any model.

## Problem 1: Roboflow Test Leakage Risk

Old behavior:

- `inspect_roboflow_export()` inspected Roboflow `train`, `valid`, and `test`.
- `RoboflowDatasetPreparer.prepare()` then sent all inspected records into `temporal_split()`.
- A Roboflow `test` image could therefore be reassigned into prepared TRAIN or VAL.

New safe behavior:

- Inspection still covers `train`, `valid`, and `test`.
- Preparation includes only Roboflow `train` and `valid` by default.
- Roboflow `test` records are excluded before temporal splitting.
- `split_metadata.json` records inspected, included, and excluded source splits, image counts, annotation counts, and whether holdout protection was overridden.
- If fewer than two eligible records remain after excluding protected splits, preparation fails clearly.

Backward compatibility:

- Historical reconstruction can opt in with `--allow-test-as-training-source`.
- The override is deliberately explicit and is recorded in `split_metadata.json` as holdout protection overridden.
- New FormicAI experiments should not use this override for true holdout data.

## Problem 2: External Evaluation Mutated Test Directories

Old behavior:

- `ExternalEvaluator` wrote `<external_test_video>/dataset.yaml` inside the evaluated test directory.
- Existing `dataset.yaml` files inside external test folders could be overwritten.

New safe behavior:

- External test image and label folders are read-only inputs.
- Evaluation YAML files are generated under the evaluation artifact tree:

```text
<output_path.parent>/external_test_dataset_yamls/<video>.yaml
```

- Generated YAML uses the absolute resolved video folder path and points Ultralytics to `images`.
- Existing `dataset.yaml` files inside external test folders are neither overwritten nor used.

## Historical Result Impact

This audit does not invalidate v1-v4 reported results because it changes future safety behavior in preparation/evaluation code only. AUDIT-01 did not regenerate frozen datasets, modify model weights, revise model-card conclusions, edit freeze manifests, or run frozen benchmarks.

## Explicit Non-Actions

During AUDIT-01:

- No training occurred.
- No model inference occurred.
- No benchmark evaluation occurred.
- No dataset regeneration occurred.
- No frozen model card conclusions were changed.
- No frozen result hashes were changed.
