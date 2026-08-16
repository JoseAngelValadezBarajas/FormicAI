# AUDIT-02: Canonical Detector And Provenance

Baseline:

```text
branch: audit/formicai-code-audit
immutable experiment baseline: ants-v4-domain-e01-final / 087c4cf8ce41790160de409fc0a1f0ab807ae42c
AUDIT-01 commits: 43fe77ac7c453bf4567c9c7de9e31cd69f80aafb, 116a55aaea961532f11f8dceea3f0026813e5142
```

Problem addressed:

- `detect-video` did not guarantee canonical inference by default.
- Detection outputs did not prove which model generated them.
- Model and source video SHA256 values were not recorded.
- Detector class semantics were not validated before inference.
- Run environment and configuration provenance were incomplete.

Canonical current champion:

```text
experiment: ants_v3_mixedscale_e01
status: CURRENT_CHAMPION
path: artifacts/models/ants_v3_mixedscale_e01/selected.pt
SHA256: 424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf
classes: {0: ant}
```

Canonical inference:

```text
confidence: 0.25
NMS IoU: 0.70
imgsz: 640
end2end: false
```

Implementation notes:

- `--champion` resolves the current champion path from the centralized detector spec.
- Champion mode verifies the model SHA256 before importing/loading YOLO.
- `--champion` and `--model` are mutually exclusive.
- Custom `--model` paths remain supported and always have their SHA256 recorded.
- Model identity is claimed by SHA256, not by filename. A custom path with the champion SHA is identified as `ants_v3_mixedscale_e01`; otherwise it is recorded as `CUSTOM / UNREGISTERED`.
- Loaded YOLO class maps must exactly match the FormicAI one-class policy: class id `0`, semantic name `ant`.
- `detect-video` writes frame JSONL as before and also writes a separate human-readable run metadata JSON artifact.
- The metadata records source SHA256, model SHA256, model identity, inference configuration, canonical/noncanonical status, overrides, class policy, environment versions, and result counts.
- `detect-video` rejects colliding output destinations after path resolution, so the annotated video, JSONL detections, and run provenance metadata cannot overwrite one another.

No real training, real model inference, benchmark evaluation, dataset mutation, annotation mutation, champion change, v5 design, tracking, push, or merge occurred as part of AUDIT-02.
