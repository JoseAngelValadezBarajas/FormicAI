# V5 Multi-Source Redundancy-Policy Remediation

Status: PASS
Experiment: `ants_v5_crowdedscale_e01`
Policy being remediated: `experiments/ants_v5/multisource_annotation_pool_policy.json`
Previous policy commit: `c141951`

## Boundary

- Frames selected in this task: `false`
- Images frozen in this task: `false`
- GT accessed: `false`
- Public tracking GT used: `false`
- Model inference: `false`
- Pseudo-labeling: `false`
- Training: `false`
- Final-test semantic access: `false`

## Observed Policy Failure

The previous label-blind selection attempt did not freeze a pool. Under the prior same-source redundancy rule, public-family selection reached only `19 / 36` images:

- `Seq0007`: target `12`, feasible `6`, shortfall `6`
- `Seq0008`: target `12`, feasible `5`, shortfall `7`
- `Seq0009`: target `12`, feasible `8`, shortfall `4`

Observed near-duplicate rejection events:

- dHash: `4`
- pHash: `341`

No GT was inspected, no detector inference occurred, and no selected pool was frozen. The failed attempt is valid aggregate label-blind policy-validation evidence.

## Diagnosis

Diagnosis: `PERCEPTUAL_HASH_BACKGROUND_DOMINANCE`.

The prior perceptual-hash rule rejected same-source candidates when either dHash or pHash crossed its threshold:

`dHash <= 5 OR pHash <= 8`

For these relatively static-background sources, pHash dominated rejection. In this acquisition regime, ants can occupy a small fraction of the frame while the background remains nearly unchanged, so low pHash distance alone is not sufficient evidence that two frames are redundant for small-object detection.

This does not make pHash generally invalid. It only invalidates pHash as a standalone hard-rejection criterion for this pool policy.

## Revised Hard-Rejection Rule

The remediated same-source hard-rejection rule is:

`exact SHA duplicate OR (dHash <= 5 AND pHash <= 8)`

Single-hash threshold crossings are no longer automatic rejections:

- dHash `<=5` and pHash `>8`: record `PERCEPTUAL_SIMILARITY_FLAG`, keep eligible subject to all other rules.
- pHash `<=8` and dHash `>5`: record `PERCEPTUAL_SIMILARITY_FLAG`, keep eligible subject to all other rules.

Exact SHA duplicates always reject.

## Unchanged Controls

- Temporal policy changed: `false`
- Pool size changed: `false`
- Source quotas changed: `false`
- Source-family balance changed: `false`
- GT strata changed: `false`
- Source001 policy changed: `false`
- YOLO/training controls changed: `false`

The temporal policy remains the main protection against adjacent repeated video states:

`max(round(fps * 2.0), floor(totalEligibleFrames / (target * 4)))`

candidate bins remain:

`target * 3`

## Retry State

The next label-blind selection attempt may retry the frozen 72-image pool using the remediated redundancy rule. It must still avoid GT, public tracking boxes, detector predictions, learned embeddings, pseudo-labels, semantic cherry-picking, and final-test information.
