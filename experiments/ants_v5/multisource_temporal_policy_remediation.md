# V5 Multisource Temporal Policy Remediation

Status: `PREREGISTERED_REMEDIATION`

Branch: `experiment/ants-v5`

Experiment: `ants_v5_crowdedscale_e01`

## Purpose

This document preregisters the minimum temporal-selection remediation needed after the public frame-selection constraint diagnosis showed:

1. the previous greedy center-nearest selector was not cardinality-optimal
2. a fixed two-second minimum spacing made nominal `12` image targets infeasible for short public sources

This remediation does not select or freeze frames, does not inspect or use ground truth, does not train, and does not run inference.

## Unchanged Design Constraints

- total annotation pool target: `72`
- public-family target: `36`
- own-domain target: `36`
- public allocation: `Seq0007=12`, `Seq0008=12`, `Seq0009=12`
- own-domain allocation: `Ants3=6`, `Ants4=3`, `Ants5=4`, `Ants6=3`, `Ants7=6`, `Ants8=6`, `Ants9=4`, `Ants10=4`
- source floors and caps remain unchanged
- source-family balance remains unchanged
- GT strata policy remains unchanged
- `source001` policy remains unchanged
- candidate-bin count remains `target * 3`

## Perceptual Redundancy Boundary

The remediated perceptual policy remains unchanged:

- exact image SHA256 match: hard reject
- same-source candidate with dHash distance `<=5` and pHash distance `<=8`: hard reject
- dHash-only or pHash-only threshold event: `PERCEPTUAL_SIMILARITY_FLAG`
- single-hash threshold events are review signals only, not automatic hard rejections

Perceptual thresholds must not be lowered to fill a target.

## Remediated Temporal Selection

For each source, generate the same center-nearest candidate set from `target * 3` equal temporal bins.

Then select using a deterministic maximum-cardinality valid subset algorithm. The optimization order is:

1. maximize selected count
2. maximize temporal spread
3. maximize minimum adjacent frame gap
4. lower candidate-bin index
5. lower frame index
6. lexicographically smaller source-relative filename
7. lexicographically smaller image SHA256

The preferred spacing is:

`round(fps * 2.0)`

The lower bound is:

`round(fps * 1.0)`

Choose the largest integer spacing `s` such that:

`round(fps * 1.0) <= s <= round(fps * 2.0)`

and the maximum-cardinality selector can select at least the target count.

If the source target is impossible at the one-second floor, record `SOURCE_SHORTFALL`, enter `REVIEW`, and stop.

## Public Dry-Run Feasibility

This label-blind dry run validates feasibility only. It does not freeze selected images.

| Source | Target | Candidate bins | Preferred spacing frames | Floor spacing frames | Selected spacing frames | Selected spacing seconds | Dry-run count | Shortfall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Seq0007` | 12 | 36 | 60 | 30 | 56 | 1.866667 | 12 | false |
| `Seq0008` | 12 | 36 | 60 | 30 | 46 | 1.533333 | 12 | false |
| `Seq0009` | 12 | 36 | 60 | 30 | 43 | 1.433333 | 12 | false |

Result: all public nominal targets are feasible under the remediated temporal policy.

## Explicit Non-Actions

- no image freeze
- no CVAT package creation
- no annotation
- no GT import or export
- no YOLO label export
- no training
- no inference
- no final-test access
