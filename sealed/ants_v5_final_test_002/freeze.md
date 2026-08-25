# Ants V5 Final Test 002 Freeze

- Final test ID: `v5_final_test_002`
- Target experiment: `ants_v5_crowdedscale_e01`
- Role: `FINAL_TEST`
- Status: `FROZEN_MODEL_UNOBSERVED`
- Source: `samples/v2/ants_v5_final_test_002.mp4`
- Source SHA256: `489f48cd4d4513d201c2afe19512f3e36ebc5b748823975190347350e276a13d`
- Frozen images: `datasets/external_test/v5_final_test_002/images/`
- Frozen image count: `30`
- Frozen image-content SHA256: `8fc4371e70864de5848617fd0668e7fd855b23ef2c690259205e74c0fc4bbe7c`
- Labels: `datasets/external_test/v5_final_test_002/labels/`
- Label file count: `30`
- Annotation-content SHA256: `88c282eb36deb8f061517678f17d67ea04ad3fd558a345330ded2a88eaaf157d`
- Combined final-test SHA256: `2dcfb490d2e70d422102efb862ebc2bb302d84e03b657cba7fe8451fac195332`
- Detailed manifest SHA256: `ce48b5ac39a61cf369dc31989707ba99ed26db477d759c4dda71bb0f7da23fdd`
- Model inference: `false`
- Models observed: `[]`
- Model assistance: `disabled`

## Annotation Policy

- Task type: object detection
- Class map: `0 = ant`
- Manual policy: tight-body bounding boxes around every visible adult ant
- Excluded: brood, larvae, pupae, food, debris, reflections, shadows
- Forbidden: Auto Label, Label Assist, model assist, SAM, YOLO inference, pseudo-labeling, pretrained detector predictions

## Closed Set QA

- Expected frozen images: `30`
- Matched annotation records: `30`
- Missing labels: `0`
- Unmatched/extras: `0`
- Duplicate annotation rows: `0`
- Malformed rows: `0`
- Class IDs observed: `0` only
- Boundary tolerance: `0.5` physical pixel
- Boundary violations: `0`
- Silent clamping: `false`
- All images readable: `true`

## Sealed GT Statistics

- Total annotations: `118`
- Verified-zero frames: `0`
- Min annotations/image: `2`
- Max annotations/image: `5`
- Mean annotations/image: `3.93333333333`
- Median annotations/image: `4`
- Median bbox width ratio: `0.044339`
- Median bbox height ratio: `0.0725645`
- Median bbox area ratio: `0.0026556796875`
- Bbox area ratio p25: `0.00194419993925`
- Bbox area ratio p75: `0.00649965696975`

## Files Committed On Seal Branch Only

- `sealed/ants_v5_final_test_002/freeze.md`
- `sealed/ants_v5_final_test_002/source_frame_manifest.json`
- `sealed/ants_v5_final_test_002/ground_truth_manifest.json`

The source video, frozen images, label files, CVAT ZIP, and model weights are not committed in the sealed records commit.
