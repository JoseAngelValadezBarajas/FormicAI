# ants_v3_mixedscale_e01 Plan

Status: proposed design only. No training, tuning, pseudo-labeling, annotation edits, commit or push were performed.

## Why v3 Exists

`ants_v2_diverse_e01` strongly improved public/external generalization, but it materially regressed on the original ant2 own-colony domain. The v3 dataset design should preserve the public-domain gains while reintroducing own-colony scale and distractor coverage.

## Starting Point

Keep the ants_v2 public TRAIN base:

- Seq0002
- Seq0003
- Seq0004
- Seq0007
- Seq0008
- Seq0009

Frozen base count before own-colony additions: 155 images and 3364 annotations.

Keep public validation provisionally:

- Seq0005
- Seq0010

Warning: these validation sources were already used for v2 model selection. Repeated experiments can overfit them.

## Own-Colony Addition

Add a small, diverse own-colony annotation pack from ant2, approximately 30-50 positive frames. Proposed e01 size: 42 positive ant2 frames. These frames target queen scale, large workers, workers near queen, brood, food, glare, substrate/container artifacts and pose variation.

Add 10-20 verified hard-negative frames where possible. Proposed e01 hard-negative size: 12 candidate frames from nest_test_001, only after full-resolution zero-ant verification.

## Validation Need

NEW OWN-COLONY VALIDATION VIDEO REQUIRED. `nest_test_001` is useful for hard negatives but does not appear to provide positive queen/worker validation diversity. `ant4` must remain frozen as final own-colony test candidate and cannot be used for train/val/model selection.

## Frozen/Stress Sources

- ant4: frozen own-colony final test candidate after manual GT.
- ant3: stress test only; not quantitative validation or train.
- Seq0001/Seq0006: observed frozen public benchmark/reference; never train/tune.
- ant2: training candidate and regression reference, not untouched final benchmark.

## Key Gap

Median normalized bbox area:

- ant2: 0.02842368
- ants_v2 TRAIN: 0.00208822

Median ant2 area is 13.61x the ants_v2 TRAIN median. This explains why mixed-scale data is the first dataset-design fix to try before threshold or architecture changes.
