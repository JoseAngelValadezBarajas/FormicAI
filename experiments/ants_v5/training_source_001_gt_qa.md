# Training Source 001 GT QA

Status: PASS
Selection status: REVIEW / SELECTION_POLICY_PREREGISTRATION_REQUIRED

## Scope

- Experiment: `ants_v5_crowdedscale_e01`
- Source role: `ANNOTATION_POOL`
- Local GT source: `LOCAL_CVAT_YOLO_DETECTION_EXPORT`
- Export: `artifacts/annotation/ants_v5_training_source_001_cvat/formicai_v5_training_source_001_annotations_yolo.zip`
- Export SHA256: `8d27dded59aaa78faa2c29697fc87362a17180ef378f9d03d46129170c64b619`
- Authoritative images: `datasets/ants_v5_annotation_pack/training_source_001/images`
- Final-test semantic content accessed: `false`
- Former final test 001 used for decisions: `false`
- Sealed final-test commit/tag contents inspected: `false`

## Annotation Policy

- Task: object detection
- Class map: `0 = ant`
- Adult-ant boxes only: tight-body bounding boxes around every visible adult ant
- Excluded: brood, larvae, pupae, food, debris, reflections, shadows
- Model assistance: disabled
- Inference used: false
- Auto-label / pseudo-labeling used: false

## Closed-Set Checks

- Expected images: 48
- Actual images: 48
- Matched images: 48
- Extra images: 0
- Missing images: 0
- Expected labels: 48
- Actual labels: 48
- Matched labels: 48
- Extra labels: 0
- Missing labels: 0
- Export label members: 48
- `train.txt` entries matched to frozen filenames: 48

## Label QA

- Classes seen: `[0]`
- Malformed rows: 0
- Duplicate rows: 0
- Image hash mismatches: 0
- Unreadable images: 0
- Dimension mismatches: 0
- Boundary tolerance: 0.5 physical px
- Silent clamping / resizing / correction: false

## Counts

- Adult-ant annotations: 193
- Zero-adult frames: 0
- Annotations per image min/median/mean/max: 1 / 4.0 / 4.0208 / 7
- BBox area ratio median/mean: 0.002808009295 / 0.00536173280975
- Neighbor count median/mean/max: 0.0 / 0.5181 / 3

## Hashes

- Frozen image-content SHA256 expected: `0aa74dca929cf992beea89d8fd19168e5cc73e013ca7561b74d7f816b24323ca`
- Frozen image-content SHA256 actual: `0aa74dca929cf992beea89d8fd19168e5cc73e013ca7561b74d7f816b24323ca`
- Frozen image-content SHA unchanged: `true`
- Annotation-content SHA256: `6383cc1361efa9a243c9635840f03a05973a8222e09829982f46f12fef84e4ea`
- Combined annotation-pool GT SHA256: `4439a2a4dc3e9a58c96881f94096fd855eba67b1b90343554881f9b8142cfdff`

## Selection Gate

The pre-GT preregistration audit found no deterministic `FINAL_TRAIN_ADDITION` rule: no fixed subset size, numerical quotas/minimums/maximums, balancing priority, or GT-aware tie-breakers. The pool was therefore characterized only; no final train addition was selected and no source-role promotion was made.
