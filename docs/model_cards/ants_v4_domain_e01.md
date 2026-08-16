# ants_v4_domain_e01 Model Card

## Status

`ants_v4_domain_e01` is closed as `MIXED` and was not promoted. The current champion remains `ants_v3_mixedscale_e01`.

```text
status: CLOSED / MIXED / NOT PROMOTED
model: YOLO26n
selected checkpoint: best.pt / selected.pt
selected represented epoch: 37
weights: artifacts/models/ants_v4_domain_e01/selected.pt
sha256: 689249476dc61912050a5106ac712c9e24dd494da892a2d113f72ba20198e0e9
```

The weight file is a local artifact and is intentionally excluded from Git.

## Lineage

Detector lineage:

- v1: original own-domain baseline.
- v2: public/diverse generalization.
- v3: mixed-scale, queen/large-worker, AntsNet, and ant2 targeted relabeling.
- v4: real-camera own-domain exposure from additional camera/tube videos.

v4 improved clean `own_colony_val_002` compared with v3 and preserved the public development sources, but it regressed against v3 on ANT4 final test, still failed `own_colony_val_001`, and regressed in standard public-reference metrics compared with v2. Overall result: `MIXED`.

## Dataset

```text
dataset: datasets/prepared/ants_v4_domain_e01
train images: 293
train annotations: 3886
train intentional zero-adult labels: 2
validation images: 90
validation annotations: 1503
total images: 383
total annotations: 5389
full train/dev dataset SHA256: 7af79305b19b25579d99bdb48e9da668cb31651f9924d7f2a02b34f0c74c4bb7
dataset manifest SHA256: 064ba8f2773cf1bb6a47be73ae19253d981d17c55de7c7b64a547736e4ad6ae4
train combined SHA256: 63030020c02725af3023698e27381615f645574ebba4ddd4cb0522b1c6e0a557
validation combined SHA256: 946125b5b5f9810d45f015d00ad36f1374e0435865e9e4487864f052bdd126b5
```

Training added real-camera own-domain sources from `Ants3` through `Ants10`, excluding `Ants11` because it is a byte-identical alias of `own_colony_val_001`, and excluding `Ants2` as an auxiliary short source.

## Canonical Inference

```text
end2end: false
confidence: 0.25
NMS IoU: 0.70
imgsz: 640
```

## Development Validation

Checkpoint selection used only `Seq0005`, `Seq0010`, and `own_colony_val_001`.

| Source | P | R | mAP50 | mAP50-95 | Predictions | GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seq0005 | 0.9819 | 0.9200 | 0.9698 | 0.5172 | 365 | 300 |
| Seq0010 | 0.8885 | 0.7390 | 0.7813 | 0.3753 | 1202 | 1068 |
| own_colony_val_001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 135 |
| Overall | 0.9048 | 0.7142 | 0.7507 | 0.3687 | 1567 | 1503 |

Macro-source metrics:

| Metric | Value |
| --- | ---: |
| Macro source precision | 0.6235 |
| Macro source recall | 0.5530 |
| Macro source mAP50 | 0.5837 |
| Macro source mAP50-95 | 0.2975 |

`own_colony_val_001` remains a development/pathological failure source, with zero predictions for both v3 and v4.

## Clean Own-Colony Reference

`own_colony_val_002` was held out from checkpoint selection and evaluated only after `selected.pt` was frozen. It is now an observed own-colony reference for future experiments.

| Model | P | R | mAP50 | mAP50-95 | Predictions | GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ants_v3_mixedscale_e01 | 0.9444 | 0.5862 | 0.5833 | 0.3226 | 19 | 29 |
| ants_v4_domain_e01 | 1.0000 | 0.6897 | 0.6850 | 0.3560 | 19 | 29 |

This is a positive v4 signal, but it did not override the ANT4 final-test regression.

## Observed Public References

`Seq0001` and `Seq0006` were already observed before this closure and are not unbiased model-selection data.

| Source | v2 P | v2 R | v2 mAP50 | v2 mAP50-95 | v4 P | v4 R | v4 mAP50 | v4 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seq0001 | 0.5638 | 0.6726 | 0.4923 | 0.1139 | 0.4663 | 0.4960 | 0.3442 | 0.0807 |
| Seq0006 | 0.7821 | 0.7225 | 0.7164 | 0.2212 | 0.6845 | 0.5861 | 0.5605 | 0.1773 |
| Macro | 0.6729 | 0.6976 | 0.6044 | 0.1676 | 0.5754 | 0.5411 | 0.4524 | 0.1290 |

v4 regressed in standard public-reference metrics compared with v2. Center-based metrics improved, but center metrics are diagnostic only and do not replace standard mAP.

## ANT4 Final Test

`ant4` is now an observed own-colony final test and is no longer unbiased for future v5+ experiments.

| Model | P | R | mAP50 | mAP50-95 | Predictions | GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ants_v1_baseline | 0.1092 | 0.0741 | 0.0098 | 0.0016 | 397 | 567 |
| ants_v2_diverse_e01 | 0.4396 | 0.3527 | 0.1738 | 0.0459 | 466 | 567 |
| ants_v3_mixedscale_e01 | 0.8664 | 0.8466 | 0.8200 | 0.4434 | 579 | 567 |
| ants_v4_domain_e01 | 0.8680 | 0.7654 | 0.7511 | 0.3786 | 559 | 567 |

v4 improved precision by only `+0.0016` over v3, but recall dropped by `-0.0812`, mAP50 by `-0.0689`, and mAP50-95 by `-0.0648`.

## Test Role Changes

```text
Seq0001 / Seq0006: OBSERVED PUBLIC REFERENCES
own_colony_val_001: DEVELOPMENT / PATHOLOGICAL FAILURE SOURCE
own_colony_val_002: OBSERVED OWN-COLONY REFERENCE for future experiments
ant4: OBSERVED OWN-COLONY FINAL TEST, no longer unbiased for v5+
```

## Reproducibility References

```text
v3 selected model SHA256: 424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf
v4 selected model SHA256: 689249476dc61912050a5106ac712c9e24dd494da892a2d113f72ba20198e0e9

own_colony_val_002 source video SHA256: 02ecfaf7a26865b6629f49756de3f3e558e0419f97b1404b8f094577dccc731e
own_colony_val_002 image-content SHA256: 9de3ea752bc889b90b6bb9528f4727daa551857c7f86486af80be91535c7af37
own_colony_val_002 annotation-content SHA256: 57f9e49ca0e7b65192985236e75ae3135c0c201cfa7f8af0ea8a63f277a6a4f0
own_colony_val_002 combined SHA256: 46a954cf00e21f196af3a33d97d34b91df953ee5eac015778e5ffaad8d327173

ant4 source video SHA256: 1f550cde40f1bfbb0cb3b513a43dd033d029e8a187fabc303e88a7c69e650d6f
ant4 image-content SHA256: 213848a0cad11a00f12bd97a4b8cd3b44104336fdf3bf9f7b3a29dcdec50a36a
ant4 annotation-content SHA256: 2029980d1c87429da3546fbbe0e44e1cbbeca3bddc7175411759732f04b94b3e
ant4 combined final-test SHA256: 79cef95f320a9796c87aa1be1c5f3a0b3b262ef68332ae1f1eec18fc0792550c
ant4 final GT manifest SHA256: 598b4962977cf230694c6fd84e414aba87a6c56fb5876ea58665ff4fcbe6e60b
```

Weights, prepared image data, labels, videos, and analysis artifacts are intentionally excluded from Git. Their paths and hashes are recorded for reproducibility.

## Conclusion

Final classification: `MIXED`.

`ants_v4_domain_e01` is closed and not promoted. The current FormicAI experimental detector champion remains `ants_v3_mixedscale_e01`.
