# V5 Training Source 001 Annotation Pool

Status: PASS

Experiment: `ants_v5_crowdedscale_e01`
Role: `ANNOTATION_POOL` derived from `TRAINING_SOURCE`

This is not the final v5 train addition. It is a frozen manual-annotation pool; final train selection waits for manual GT density/scale/local-crowding labels.

## Boundary

No detector inference, pretrained YOLO, pseudo-labeling, Label Assist, manual ant counts, or final-test information were used. The V5 final test remains `FROZEN_UNOBSERVED`; `samples/v2/videonuevoants.mp4` and `datasets/external_test/v5_final_test/` were not accessed.

## Source Verification

- source: `samples/v2/ants_v5_training_source_001.mp4`
- source SHA256: `53a659c615fb1ae5e6ab1ae96ba84531cb8e7d8aae6c396a19b8f7e7cf623e7b`
- source role: `TRAINING_SOURCE`
- alias result: PASS, inherited from verified training-source audit and guarded source-role registry

## Candidate Pool Verification

- candidates: 120
- unique candidate filenames/indexes/image hashes: 120
- extraction algorithm: unchanged 120 equal temporal bins / center frame
- model-derived metadata: none

## Redundancy

- near-duplicate rule: dHash Hamming distance <= 4
- adjacent candidate pairs: 119
- near-duplicate adjacent pairs: 104
- effectively distinct adjacent-run states: 16
- largest cluster size: 28

## Selection Algorithm

- split source duration into 12 equal temporal regions
- target 48 images, four per region
- first strict pass keeps candidates with >=1.0s temporal spacing and dHash distance >4 within the region
- second pass relaxes temporal spacing while preserving dHash uniqueness
- final fill maximizes minimum timestamp distance, then lower candidate index
- output manifest is sorted by timestamp

## Frozen Pool

- selected images: 48
- first timestamp: 0.250s
- last timestamp: 60.308s
- selected image-content SHA256: `0aa74dca929cf992beea89d8fd19168e5cc73e013ca7561b74d7f816b24323ca`
- manifest SHA256: `7500a713419d631910429b6d97b6300d77cd489119ac02a04906465b59dd04f9`
- image staging path: `datasets/ants_v5_annotation_pack/training_source_001/images/`

## Leakage / Role Check

- selected images derive only from: `training_source_001`
- overlap count against allowed existing local/recorded hashes: 0
- final-test protection: source-role guard and already-recorded registry/hash results only; forbidden final-test paths were excluded before inspection

## Roboflow

- upload plan: `experiments/ants_v5/training_source_001_roboflow_upload_plan.json`
- upload executed: false
- labels created: false
- missing labels mean: not yet annotated, not zero-adult

V5 final test content accessed: false.
