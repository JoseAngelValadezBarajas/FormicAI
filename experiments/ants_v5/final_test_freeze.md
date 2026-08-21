# ants_v5 Final Test Freeze

Status: FROZEN_UNOBSERVED

This record freezes the source video, deterministic 30-frame image set, and manual FormicAI ground truth for the future `ants_v5` final test. It is not a training, development-validation, checkpoint-selection, tuning, or autopsy source.

## Source

- filename: `videonuevoants.mp4`
- path: `samples/v2/videonuevoants.mp4`
- SHA256: `97c8411cd7ea4bfe1f821fe2023258d0caef529327bdfe9c30576d9276a1baf2`
- resolution: 1920x1080
- fps: 60.0197881834
- frame count: 3664
- duration seconds: 61.046533
- codec: `h264`

Alias checks found no exact SHA256 match in 820 versioned historical hashes or 17 local known source videos.

## Role

- role: `FINAL_TEST`
- status: `FROZEN_UNOBSERVED`
- experiment target: `ants_v5`
- intended scope: v5 and later only after model freeze
- train eligible: false
- development validation eligible: false
- checkpoint selection eligible: false
- tuning eligible: false
- autopsy input eligible: false

## Deterministic Selection

Exactly 30 frames were selected by temporal stratification over the full frame interval:

```text
selected_frame_index = floor((bin_index + 0.5) * frame_count / 30)
```

Selection depends only on source frame count and bin index. It does not use ant count, appearance, perceived difficulty, crowding, model predictions, or known detector failures.

- selected frames: 30
- first selected frame: 61
- last selected frame: 3602

## Ground Truth Freeze

Manual annotations were refreshed from Roboflow using the official SDK and matched to the frozen frame set by exact filename. The Roboflow split was treated as transport metadata only; the scientific role is `FINAL_TEST`.

- images: 30
- label files: 30
- annotations: 141
- empty labels / verified-zero adult frames: 0
- malformed rows: 0
- class mapping: `0 = ant`
- boundary tolerance used for derived pixel-boundary QA: 0.5 px

## Hashes

- image-content SHA256: `2ad51b67334de6ea0310d534b79f0b9510b0f6b9da9ff5512245d4cae584cca4`
- annotation-content SHA256: `9c92753691a16327271ef72c2caef9139009cdab32d7238cd3fa7cedb4c96360`
- combined final-test SHA256: `10b7b81894d43d5ae1974ca7ea165631f156ab4b854bc9c65feda283b6289e5b`
- final manifest SHA256: `a3ceefa2ad649ecce1b4f56c27c77e64ff3b3951a12e7b46ad11e1ff8fbb41cb`

## Leakage Guard

- selected image hashes unique: true
- local existing images checked outside this set: 5802
- selected image hash overlaps: 0
- training/dev YAML mentions outside this v5 record: 0
- detector observation before freeze: false

## Model Observation

```json
{
  "modelInferencePerformed": false,
  "modelsObserved": [],
  "detectorPredictionsGenerated": false,
  "labelAssistUsed": false,
  "pseudoLabelingUsed": false
}
```

No detector inference, pseudo-labeling, label assist, benchmark, or prediction artifact generation was performed during this freeze.

## Artifacts

- local frame set: `datasets/external_test/v5_final_test/images/`
- labels directory: `datasets/external_test/v5_final_test/labels/`
- root metadata: `datasets/external_test/v5_final_test/metadata.json`
- final local manifest: `datasets/external_test/v5_final_test/metadata/final_gt_manifest.json`
- tracked manifest: `experiments/ants_v5/final_test_source_frame_manifest.json`
- tracked GT manifest: `experiments/ants_v5/final_test_ground_truth_manifest.json`
