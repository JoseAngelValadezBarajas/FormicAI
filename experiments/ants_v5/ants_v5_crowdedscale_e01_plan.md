# ants_v5_crowdedscale_e01 Experiment Design

Status: DESIGN_ONLY / PRE-REGISTERED

Current champion: `ants_v3_mixedscale_e01`.
V4 status: CLOSED / MIXED / NOT PROMOTED.
Repository-wide audit: COMPLETE / VERIFIED.
V5 final test: FROZEN_UNOBSERVED and not accessed for this design.

## Objective

Preserve v3's broad/public and crowded performance, preserve useful own-domain gains from v4, and explicitly improve representation of small ants in locally crowded scenes without changing canonical inference.

## Testable Hypothesis

v4's sparse own-domain addition shifted training exposure toward scenes with relatively few ants and underrepresented small, locally crowded ants. A balanced addition containing substantially more small-ant and crowded examples, while preserving v3/v4 diversity, may recover crowded recall without sacrificing own-domain performance.

This is a TESTABLE HYPOTHESIS, not a conclusion.

## Alternative Hypotheses

- H1 confidence/calibration: v4 scores crowded ants lower, but confidence sweeps showed this is secondary.
- H2 localization: v4 boxes may drift under crowding, but center recall also dropped, so this is not sufficient alone.
- H3 small-object representation: v4 lost most in small GT bins; v5 primarily tests this.
- H4 local-crowding representation: v4 regressed in moderate/crowded-neighbor bins; v5 primarily tests this.
- H5 broader domain imbalance / forgetting: v4 own-domain sparse exposure may have changed representation balance.

## Evidence Base

Source: `experiments/ants_v5/ant4_v3_v4_autopsy.md`.

Key recorded findings:

- v4 lost true-positive coverage relative to v3 on observed ant4.
- Regression concentrated in small ants, high-density frames, and local crowding.
- Isolated ants did not regress.
- Confidence/calibration shift is secondary.
- NMS is not primary.
- Sparse-rebalancing hypothesis is PARTIALLY_SUPPORTED.

## Source Roles

Tracked in `experiments/ants_v5/source_role_registry.json`.

Development validation remains exactly:

- Seq0005
- Seq0010
- own_colony_val_001

Observed references, not optimization targets:

- Seq0001
- Seq0006
- own_colony_val_002
- ant4

Final test:

- v5 final test, FROZEN_UNOBSERVED, off limits until one-time post-freeze evaluation.

## Current Train Distribution

Tracked in `experiments/ants_v5/train_distribution_analysis.json`.

Summary:

- v3 train: 222 images, 3731 annotations, mean 16.81 ants/image, bbox area median 0.002197.
- v4 train: 293 images, 3886 annotations, mean 13.26 ants/image, bbox area median 0.002196.
- v4 own-domain addition: 71 images, 155 annotations, about 2.18 ants/image, including 2 intentional zero-adult negatives.

Primary imbalance: v4 added valuable real-camera domain exposure, but the addition was sparse relative to the crowded/small-ant failure mode diagnosed on ant4.

## Base Dataset Strategy

Recommended: full v4 base plus a compensating crowded/small addition.

Rationale:

- Do not remove v3 public/diverse and mixed-scale strengths.
- Do not discard v4 own-domain images merely because the v4 model regressed; the labels/images appear useful, and the issue may be weighting/distribution rather than bad data.
- A v3-base strategy risks losing the useful own-domain exposure from v4 unless it manually reconstructs it.
- Full v4 base plus a targeted addition keeps one main experimental variable: data distribution balance.

## V5 Train-Addition Strata

Numerical boundaries are derived from current train-eligible v4 train distribution, not final test data.

Density by annotations/image:

- sparse: <= 5
- medium: > 5 and <= 10
- crowded: > 10

Scale by image median bbox area:

- small-heavy: median area <= 0.0016905381
- mixed: > 0.0016905381 and <= 0.0028168401
- medium/large-heavy: > 0.0028168401

Local crowding proxy:

- radius: 0.095833335 normalized units, computed as 1.5 * median train max(width,height)
- isolated-dominant: median neighbor count <= 0
- moderate: > 0 and <= 2
- crowded-neighbor: > 2

Balance constraints before selection:

- no single density stratum should dominate the addition;
- crowded and medium-density frames must both be materially represented;
- small-heavy and crowded-neighbor examples are required;
- do not fill quotas with near-duplicates or low-quality ambiguity;
- if supply cannot satisfy these targets, record the shortfall instead of lowering criteria.

## Candidate Supply

Tracked in `experiments/ants_v5/candidate_supply_analysis.json`.

Current labeled train pool alone is insufficient as a new v5 addition because own-domain additions are sparse. Existing train-eligible source videos Ants3-Ants10 may still contain useful unused portions, but this must be audited deterministically without inference before frame selection. If they cannot supply crowded small-ant examples, create a new never-holdout training recording specifically for this purpose.

## Candidate Selection Policy

No images are selected in this design task.

Future selection must run in this order:

1. source-role audit;
2. candidate supply audit over train-eligible sources only;
3. predeclare source-specific quotas from available supply;
4. temporal stratification within eligible sources;
5. diversity pruning using only temporal distance, perceptual similarity, luminance/saturation/contrast/sharpness, and visual redundancy;
6. manual annotation with FormicAI single-class tight adult-ant policy;
7. QA, freeze manifests, and hashes.

Forbidden for candidate selection:

- ant4
- own_colony_val_002
- Seq0001
- Seq0006
- own_colony_val_001
- v5 final test

## Validation And Checkpoint Selection

Use development validation only:

- Seq0005
- Seq0010
- own_colony_val_001

Primary metric: `macro_source_mAP50_95` with equal source weighting.
Tie-breakers:

1. overall validation mAP50-95
2. macro source mAP50
3. overall validation recall
4. earlier epoch if effectively tied

## Canonical Inference

Keep canonical inference fixed:

- imgsz = 640
- conf = 0.25
- iou = 0.70
- end2end = false

Confidence and NMS sweeps from ant4 are diagnostic only and must not tune v5 production inference.

## Training Controls

Default controlled training setup:

- architecture: YOLO26n
- pretrained initialization family: same as v3/v4 controlled experiments
- imgsz: 640
- max epochs: 50
- seed: 42
- AMP: enabled where supported
- batch: comparable auto behavior

Do not train in this design phase.

## Success Criteria And Promotion Rules

Tracked in `experiments/ants_v5/success_criteria.json`.

High level:

- primary DEV gate: no material macro DEV regression relative to v3;
- own-domain development: investigate own_colony_val_001 without using final test;
- observed references opened only after selected.pt is frozen;
- final test evaluated exactly once after selected.pt freeze.

Promotion outcomes are predeclared as PROMOTE, MIXED, or REJECT.

## Experiment Sequence

1. source-role audit
2. current train distribution analysis
3. candidate supply analysis
4. predeclare strata/quotas from supply
5. deterministic candidate selection
6. manual annotation if new images are required
7. QA + hashes
8. materialize v5 dataset
9. freeze dataset manifest
10. train
11. select checkpoint using DEV only
12. freeze selected.pt SHA
13. evaluate observed references
14. evaluate frozen v5 final test exactly once

## Boundary

No v5 final-test content was accessed for this design. No image selection, Roboflow upload, annotation, training, inference, threshold tuning, or dataset mutation is performed here.
