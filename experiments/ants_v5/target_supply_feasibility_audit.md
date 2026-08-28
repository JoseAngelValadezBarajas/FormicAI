# V5 Target-Supply Feasibility Audit

Status: PASS
Experiment: `ants_v5_crowdedscale_e01`
Audit type: supply feasibility only
Created: `2026-08-28T08:41:00Z`

## Boundary

- Final-test semantic access: `false`
- Former final test 001 used for decisions: `false`
- V5 final test 002 used for decisions: `false`
- Training: `false`
- Final train addition selected: `false`
- YOLO architecture changed: `false`

The frozen strata definitions were not changed. Source `training_source_001` remains an annotated candidate / annotation pool, not a final train addition.

## Source-Role Result

Train-eligible sources audited: `Seq0002`, `Seq0003`, `Seq0004`, `Seq0007`, `Seq0008`, `Seq0009`, `ant2`, `AntsNet`, `Ants3`, `Ants4`, `Ants5`, `Ants6`, `Ants7`, `Ants8`, `Ants9`, `Ants10`, `training_source_001`.

Forbidden sources excluded: `Seq0001`, `Seq0006`, `Seq0005`, `Seq0010`, `own_colony_val_001`, `own_colony_val_002`, `ant4`, `former_v5_final_test_001`, `v5_final_test_002`, `Ants11`, `Ants2`, `ant3`, and ambiguous/excluded sources.

No train-eligibility role conflict was found. `Seq0005` and `Seq0010` are treated as forbidden development-validation sources; `Ants11` remains an own-colony validation alias and never-train.

## Available Material Categories

- Already in v4 base: 293 images / 3886 annotations across public Seq0002/3/4/7/8/9, AntsNet, ant2, and own-domain Ants3-10.
- Unused labeled supply: `training_source_001`, 48 images / 193 annotations, exact FormicAI manual GT.
- Unused unlabeled supply: public Seq0002/3/4/7/8/9 archive frames, raw AntsNet images, historical ant2/raw frames, own-domain Ants3-10 video frames, and remaining `training_source_001` video frames. Exact strata are not assigned to these without manual GT.

## Existing Labeled Target Supply

Already-in-v4 supply contains target examples but is not new v5 information:

- v4 train density: {'sparse': 118, 'medium': 89, 'crowded': 86}
- v4 train scale: {'small-heavy': 54, 'mixed': 101, 'medium-large-heavy': 136, 'zero-adult': 2}
- v4 train local crowding: {'isolated-dominant': 147, 'moderate': 140, 'crowded-neighbor': 6}
- v4 train target intersections: {'crowded': 86, 'small-heavy': 54, 'crowded-neighbor': 6, 'crowded_and_small_heavy': 33, 'crowded_and_crowded_neighbor': 5, 'small_heavy_and_crowded_neighbor': 5, 'crowded_small_heavy_and_crowded_neighbor': 5}

Unused labeled supply is insufficient for the missing H3/H4 intervention:

- `training_source_001` density: {'crowded': 0, 'medium': 10, 'sparse': 38}
- `training_source_001` scale: {'medium-large-heavy': 23, 'mixed': 25, 'small-heavy': 0, 'zero-adult': 0}
- `training_source_001` local crowding: {'crowded-neighbor': 0, 'isolated-dominant': 27, 'moderate': 21}
- `training_source_001` target intersections: {'crowded': 0, 'small-heavy': 0, 'crowded-neighbor': 0, 'crowded_and_small_heavy': 0, 'crowded_and_crowded_neighbor': 0, 'small_heavy_and_crowded_neighbor': 0, 'crowded_small_heavy_and_crowded_neighbor': 0}

## Seq0007 / Seq0008 / Seq0009

- `Seq0007`: 30 v4-train images / 1089 annotations; ~647 unused archive frames. Label-blind sample: `likely_useful_candidate`. Original public GT is fixed tracking-box style, so unused exact use requires manual FormicAI relabeling.
- `Seq0008`: 30 v4-train images / 541 annotations; ~547 unused archive frames. Label-blind sample: `likely_useful_candidate`. Original public GT is fixed tracking-box style, so unused exact use requires manual FormicAI relabeling.
- `Seq0009`: 20 v4-train images / 987 annotations; ~506 unused archive frames. Label-blind sample: `likely_useful_candidate`. Original public GT is fixed tracking-box style, so unused exact use requires manual FormicAI relabeling.

These sources are feasible for more small/crowded public examples only with manual tight-body relabeling. They should not be treated as direct labeled unused supply because their original full-sequence GT uses incompatible fixed tracking/object boxes. Adding more from the same source would increase temporal coverage more than domain diversity.

## Ants3-10

- `Ants3`: 7 v4-train images already used; ~1876 unused video frames. Label-blind target feasibility: `uncertain`; lateral chamber with a few medium/large ants and local group near tube/chamber edge; visually stable and not obviously small/crowded.
- `Ants4`: 12 v4-train images already used; ~3054 unused video frames. Label-blind target feasibility: `unlikely_useful_candidate`; block entry/top area mostly sparse; useful mainly for sparse own-domain context.
- `Ants5`: 8 v4-train images already used; ~3681 unused video frames. Label-blind target feasibility: `unlikely_useful_candidate`; top-down own-domain arena, sparse-to-low-medium with occasional close pairs; not a primary target source.
- `Ants6`: 12 v4-train images already used; ~4390 unused video frames. Label-blind target feasibility: `unlikely_useful_candidate`; block entry footage like Ants4, sparse and highly redundant.
- `Ants7`: 8 v4-train images already used; ~4371 unused video frames. Label-blind target feasibility: `uncertain`; lateral chamber with small local groups around larger ants; possible local-contact support but not broad crowded/small supply.
- `Ants8`: 8 v4-train images already used; ~3668 unused video frames. Label-blind target feasibility: `uncertain`; similar lateral chamber material with a few grouped ants; somewhat varied but still visually stable.
- `Ants9`: 8 v4-train images already used; ~4586 unused video frames. Label-blind target feasibility: `unlikely_useful_candidate`; top-down own-domain arena mostly sparse in full-span sample.
- `Ants10`: 8 v4-train images already used; ~5786 unused video frames. Label-blind target feasibility: `unlikely_useful_candidate`; top-down own-domain arena with occasional pairs near entrance/white object, but sparse overall.

Overall, unused own-domain video supply is useful for domain complement and occasional local-contact examples, but the label-blind sample does not show a single sufficient own-domain H3/H4 source.

## Global Density vs Local Crowding

The audit does not reinterpret medium-density frames as crowded. Exact labels show that >10 ants/frame is not logically required to see close-neighbor groups, but it is rare to meet the frozen image-level `crowded-neighbor` threshold without high global density.

- Already-in-v4 medium-density + crowded-neighbor images: 1
- Already-in-v4 medium-density images with any ant having >=3 neighbors: 25
- `training_source_001` medium-density frames: 10
- `training_source_001` medium-density + moderate local crowding: 8
- `training_source_001` medium-density + crowded-neighbor: 0

## Source 001 Role

`training_source_001` should not be discarded. It adds own-domain appearance diversity and 10 medium-density examples. It can later complement a separate crowded/small pack, but it is not the primary H3/H4 intervention under the frozen thresholds and must not be marked `FINAL_TRAIN_ADDITION` now.

## Decision

Overall feasibility classification: `MULTI_SOURCE_COMBINATION_REQUIRED`

Existing labeled unused supply is not sufficient. Existing unlabeled train-eligible material is promising only as a combination: public Seq0007/8/9 for target crowded/small phenomena plus own-domain/source001 material for domain complement. New manual annotation is required; a new recording is not required before a preregistered multi-source annotation/selection policy is attempted.

YOLO26n @ imgsz640 remains unchanged for `ants_v5_crowdedscale_e01`; larger YOLO is a future controlled ablation.

## Files

- `experiments/ants_v5/target_supply_inventory.json`
- `experiments/ants_v5/unused_train_candidate_supply.json`
- `experiments/ants_v5/target_supply_feasibility_audit.md`
