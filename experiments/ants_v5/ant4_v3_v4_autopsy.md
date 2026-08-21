# ANT4 Crowded-Density Autopsy: v3 vs v4

Status: PASS

This autopsy investigates the already-observed ant4 regression for `ants_v4_domain_e01` relative to `ants_v3_mixedscale_e01`.

Important boundary: the new v5 final test was NOT accessed. No files under `datasets/external_test/v5_final_test/` and no `samples/v2/videonuevoants.mp4` content were opened, decoded, inspected, evaluated, or used.

## Inputs

- ant4 role: OBSERVED_REFERENCE
- ant4 frames: 20
- ant4 GT ants: 567
- ant4 source SHA256: `1f550cde40f1bfbb0cb3b513a43dd033d029e8a187fabc303e88a7c69e650d6f`
- ant4 image-content SHA256: `213848a0cad11a00f12bd97a4b8cd3b44104336fdf3bf9f7b3a29dcdec50a36a`
- ant4 annotation-content SHA256: `2029980d1c87429da3546fbbe0e44e1cbbeca3bddc7175411759732f04b94b3e`
- ant4 combined final-test SHA256: `79cef95f320a9796c87aa1be1c5f3a0b3b262ef68332ae1f1eec18fc0792550c`
- v3 selected SHA256: `424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf`
- v4 selected SHA256: `689249476dc61912050a5106ac712c9e24dd494da892a2d113f72ba20198e0e9`

## Canonical Params

- imgsz: 640
- conf: 0.25
- NMS IoU: 0.70
- end2end: false
- requested device: 0
- runtime device: cpu (local CPU fallback; CUDA unavailable)

## Canonical Reproduction

| model | P | R | mAP50 | mAP50-95 | predictions | IoU50 matched | center recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 | 0.8667 | 0.8483 | 0.8201 | 0.4443 | 579 | 485 | 0.9065 |
| v4 | 0.8680 | 0.7653 | 0.7511 | 0.3787 | 558 | 450 | 0.8695 |

Observed v4-v3 deltas reproduced within tolerance:

- precision: 0.0013
- recall: -0.0830
- mAP50: -0.0690
- mAP50-95: -0.0656

## Key Findings

- Primary explanation: v4 loses mostly true-positive coverage in crowded ant4 frames: canonical prediction count is lower, IoU50 matches drop, and center recall also drops. This points more to detection/confidence/suppression under crowding than to pure box-style localization.
- Sparse-rebalancing hypothesis: PARTIALLY_SUPPORTED
- Hypothesis note: v4 added sparse own-domain exposure and improves sparse clean own-domain validation, while ant4 is very crowded and v4 loses recall/AP. The autopsy supports a crowded-behavior failure mode but cannot prove training causality.
- Density caution: Only 20 frames; correlations are descriptive, not statistical proof.

## Density Relationship

- GT count vs recall delta Pearson: -0.27179145422997514
- GT count vs missed-ant delta Pearson: 0.32358317437172407
- High-density group v4-v3 matched delta: -22
- High-density group v4-v3 missed delta: 22

## Scale And Crowding

- Small-object degradation is the largest scale-specific effect: small GT bin v3 matched 149/189, v4 matched 122/189, delta -27, recall delta -0.1429.
- Medium GT bin: v3 matched 165/189, v4 matched 164/189, delta -1.
- Large GT bin: v3 matched 171/189, v4 matched 164/189, delta -7.
- Local-crowding degradation is present: crowded 3+ neighbor bin v3 matched 221/274, v4 matched 201/274, delta -20.
- Moderate 1-2 neighbor bin: v3 matched 207/232, v4 matched 191/232, delta -16.
- Isolated 0 neighbor bin did not regress: v3 matched 57/61, v4 matched 58/61.

## Threshold And NMS Diagnostics

- Confidence sweep is diagnostic only; canonical conf remains 0.25.
- Lowering confidence from 0.25 to 0.05 recovers v4 IoU50 matches from 450 to 497, but v3 also recovers from 485 to 527, so threshold/calibration shift is only a partial explanation.
- NMS sweep is diagnostic only; canonical NMS IoU remains 0.70.
- Raising NMS IoU from 0.70 to 0.90 recovers v4 IoU50 matches only from 450 to 454 while predictions rise from 558 to 1033, so NMS suppression is secondary and not a practical explanation for the regression.

## Miss Classification

- v3 canonical IoU50 misses: 82 total; 53 confidence/calibration-like below 0.25 and 29 localization-like nearby but poor.
- v4 canonical IoU50 misses: 117 total; 71 confidence/calibration-like below 0.25, 43 localization-like nearby but poor, and 3 with no useful candidate at conf 0.05.
- v4 increases every miss category, with most of the additional loss concentrated in true-positive coverage under crowding/small-object conditions.

## Diagnostic Artifacts

- metrics: `artifacts/analysis/ants_v3_v4_ant4_autopsy/autopsy_metrics.json`
- per-frame CSV: `artifacts/analysis/ants_v3_v4_ant4_autopsy/per_frame_breakdown.csv`
- confidence sweep CSV: `artifacts/analysis/ants_v3_v4_ant4_autopsy/confidence_sweep.csv`
- NMS sweep CSV: `artifacts/analysis/ants_v3_v4_ant4_autopsy/nms_sweep.csv`
- scale/crowding/overlap JSON: `artifacts/analysis/ants_v3_v4_ant4_autopsy/scale_crowding_overlap_analysis.json`
- miss classification JSON: `artifacts/analysis/ants_v3_v4_ant4_autopsy/miss_classification.json`
- plots: `artifacts/analysis/ants_v3_v4_ant4_autopsy/plots/`
- montages: `artifacts/analysis/ants_v3_v4_ant4_autopsy/montages/`

Representative visual frames were selected deterministically from largest v4 regression, similar v3/v4 performance, highest density, lower-confidence recovery, and NMS-recovery criteria:

- `ant4_frame_000289_t0009.629.jpg`
- `ant4_frame_000273_t0009.096.jpg`
- `ant4_frame_000305_t0010.162.jpg`
- `ant4_frame_000080_t0002.666.jpg`
- `ant4_frame_000096_t0003.199.jpg`
- `ant4_frame_000032_t0001.066.jpg`
- `ant4_frame_000209_t0006.964.jpg`
- `ant4_frame_000112_t0003.732.jpg`
- `ant4_frame_000128_t0004.265.jpg`
- `ant4_frame_000144_t0004.798.jpg`

## Scientific Boundary

This is failure analysis only. It does not select thresholds, tune NMS, select checkpoints, propose v5 data, or alter model status.

V5 final test accessed: false.
