# V5 Multi-Source Annotation-Pool Preregistration

Status: PASS
Experiment: `ants_v5_crowdedscale_e01`
Policy type: annotation-pool construction only
Created: `2026-08-28T09:10:00Z`

## Purpose

This preregisters a deterministic multi-source annotation-pool policy before selecting concrete frames. It does not select final training images, does not annotate, and does not train.

The committed feasibility result is `MULTI_SOURCE_COMBINATION_REQUIRED`: existing labeled unused supply is not enough, existing unlabeled train-eligible material is promising as a combination, new manual annotation is required, and a new recording is not required before trying this policy.

## Boundaries

- Train-eligible sources only: `Seq0007`, `Seq0008`, `Seq0009`, `Ants3`, `Ants4`, `Ants5`, `Ants6`, `Ants7`, `Ants8`, `Ants9`, `Ants10`, plus bounded existing complement `training_source_001`.
- Absolute exclusions: `Seq0005`, `Seq0010`, `own_colony_val_001`, `Seq0001`, `Seq0006`, `own_colony_val_002`, `ant4`, `former_v5_final_test_001`, `v5_final_test_002`, `Ants11`, `Ants2`, and ambiguous/excluded sources.
- Final-test semantic access: `false`.
- GT used for frame selection: `false`.
- Model inference used for frame selection: `false`.
- Final train addition selected: `false`.
- Training: `false`.

Public `Seq0007`/`Seq0008`/`Seq0009` tracking-style GT must not be imported, converted, or used as V5 labels. Any selected public frames require new manual FormicAI tight-body annotation.

## Frozen Strata

The post-annotation GT strata remain unchanged:

- Density: sparse `<=5`, medium `>5 <=10`, crowded `>10` adult annotations/image.
- Scale: small-heavy median normalized bbox area `<=0.0016905381`; mixed `>0.0016905381 <=0.0028168401`; medium/large-heavy `>0.0028168401`.
- Local crowding: radius `0.095833335`; isolated-dominant median neighbors `<=0`; moderate `>0 <=2`; crowded-neighbor `>2`.

No frame may be chosen by adult count, bbox size, exact density, exact scale, or exact local crowding before manual GT exists.

## Pool Size

Target new manual annotation pool: `72` images.

Rationale: `72` is inside the preregistered `60-90` range, large enough that failed strata candidates should not collapse the experiment, and splits evenly into public and own-domain families while keeping per-source caps small.

This is a manual annotation pool only. It is not the future `FINAL_TRAIN_ADDITION`.

## Source-Family Allocation

Public/diverse family: `36` images.

- `Seq0007`: target `12`, floor `10`, cap `12`.
- `Seq0008`: target `12`, floor `10`, cap `12`.
- `Seq0009`: target `12`, floor `10`, cap `12`.

Own-domain family: `36` images.

- `Ants3`: target `6`, floor `3`, cap `6`.
- `Ants4`: target `3`, floor `3`, cap `6`.
- `Ants5`: target `4`, floor `3`, cap `6`.
- `Ants6`: target `3`, floor `3`, cap `6`.
- `Ants7`: target `6`, floor `3`, cap `6`.
- `Ants8`: target `6`, floor `3`, cap `6`.
- `Ants9`: target `4`, floor `3`, cap `6`.
- `Ants10`: target `4`, floor `3`, cap `6`.

No single source may exceed `12` images or `16.667%` of the new pool. If supply permits, all 11 new-frame sources must be represented.

Family bounds after redistribution:

- Public/diverse: minimum `30`, maximum `42`.
- Own-domain: minimum `30`, maximum `42`.

## Label-Blind Signals

Allowed for future frame selection:

- source identity
- frame index
- timestamp
- deterministic temporal stratification
- source-relative filename
- image SHA256
- technical readability/decode checks
- non-learned dHash/pHash similarity
- source-level feasibility category already frozen in the feasibility audit

Forbidden:

- manual adult count
- exact density classification
- bbox size measurement
- exact scale classification
- exact local-crowding classification
- GT boxes
- public tracking boxes as labels
- v1/v2/v3/v4 predictions
- pretrained YOLO predictions
- pseudo-labels
- learned embeddings
- final-test information

## Temporal Selection

For each source, future selection must enumerate unused eligible frames in ascending frame-index order after excluding frames already represented in the v4 base.

For each source, split the eligible temporal span into `target * 3` equal bins. Generate one center-nearest candidate per bin before hash pruning. Accepted frames from the same source must be at least:

`max(round(fps * 2.0), floor(totalEligibleFrames / (target * 4)))`

frame indexes apart. If fps is unknown, use only the frame-index term.

Tie-breakers:

1. center-nearest frame within temporal bin
2. lower absolute frame-index distance to bin center
3. lower frame index
4. lexicographically smaller source-relative filename
5. lexicographically smaller image SHA256

## Readability And Near-Duplicates

Readability gates are technical only:

- image/video frame must decode
- width `>=320`
- height `>=240`
- grayscale standard deviation `>=8.0`
- Laplacian variance `>=25.0`

Near-duplicate pruning:

- compute 64-bit dHash and 64-bit pHash
- exact image SHA256 matches always reject
- reject a same-source candidate if dHash Hamming distance is `<=5` and pHash Hamming distance is `<=8` against the same already accepted same-source frame
- if only dHash `<=5` or only pHash `<=8`, record `PERCEPTUAL_SIMILARITY_FLAG` and keep the candidate eligible subject to all other rules
- do not automatically reject cross-source similarities, but flag cross-source dHash `<=3` or pHash `<=6`
- do not lower hash or spacing thresholds to fill the target number

Remediation note: a label-blind policy-validation attempt under the prior `dHash <=5 OR pHash <=8` rule reached only `19 / 36` public-family images and produced `341` pHash-only hard rejections versus `4` dHash hard rejections. The diagnosis is `PERCEPTUAL_HASH_BACKGROUND_DOMINANCE`: pHash is retained as a useful audit signal, but no longer acts as a standalone hard-rejection criterion in this relatively static-background acquisition regime.

## Redistribution

If a source cannot meet its target after eligibility, temporal spacing, readability, and hash pruning, keep accepted frames if the source meets its floor; otherwise record the floor shortfall.

Redistribute deficits within the same family first, below cap, using feasibility tier then source order:

- public order: `Seq0007`, `Seq0008`, `Seq0009`
- own-domain order: `Ants3`, `Ants7`, `Ants8`, `Ants5`, `Ants9`, `Ants10`, `Ants4`, `Ants6`

If a whole family cannot reach `36`, transfer at most `6` images to the other family while preserving source caps and the donor family minimum of `30`.

If the pool still cannot reach `72` without violating caps, family bounds, temporal spacing, readability, or hash thresholds, report the shortfall. Do not fill with near-duplicates or forbidden sources.

## Source001 Policy

Policy choice: `B_BOUNDED_DOMAIN_COMPLEMENT_SUBSET`.

`training_source_001` is already manually annotated and is not part of the new 72-image manual annotation queue. It may later provide a bounded domain-complement subset only in a separate final-train-addition preregistration.

Limits:

- include all 48 automatically: `false`
- maximum images in any future final train addition: `12`
- maximum fraction of any future final train addition: `20%`
- role: domain-complement candidate only
- selected now: `false`

Future deterministic rule if used: after new pool GT strata are measured, source001 may be considered only as complement; select at most the lower of 12 images or 20% of the future addition, prioritizing preregistered medium-density/moderate-local-crowding complement and breaking ties by source001 image SHA256 then filename.

## Annotation Policy

All newly selected pool images will later be annotated in local CVAT as object detection.

Exactly one class:

- `0 = ant`

Manual policy: tight-body boxes around every visible adult ant.

Exclude brood, larvae, pupae, food, debris, shadows, and reflections.

No Auto Label, Label Assist, SAM, YOLO inference, pseudo-labeling, pretrained detector predictions, or model assistance.

## After Manual GT

After manual GT and QA, apply the frozen density, scale, and local-crowding strata. The future `FINAL_TRAIN_ADDITION` selection policy must be preregistered separately before selecting concrete final image identities.

## Training Controls

`ants_v5_crowdedscale_e01` remains:

- architecture: `YOLO26n`
- imgsz: `640`
- max epochs: `50`
- seed: `42`
- pretrained family: same as v3/v4 controlled experiments
- AMP: enabled
- canonical conf: `0.25`
- canonical IoU: `0.70`
- end2end: `false`

YOLO26s remains a future controlled ablation only.

## Output

Policy JSON: `experiments/ants_v5/multisource_annotation_pool_policy.json`

This plan: `experiments/ants_v5/multisource_annotation_pool_plan.md`
