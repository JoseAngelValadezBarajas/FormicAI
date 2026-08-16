# ants_v2 Annotation Policy Analysis

## Current Mismatch

FormicAI `ants_v1_baseline` was trained primarily with tight ant body boxes. The public ANTS dataset uses tracking boxes that are fixed or approximately fixed by sequence: `Seq0001`-`Seq0005` use 94x94 boxes and `Seq0006`-`Seq0010` use 64x64 boxes.

## Option A: Train Directly With Original ANTS Boxes

Advantages: fastest path to more labels, exact public provenance, immediate indoor/outdoor/density diversity.

Risks: teaches fixed tracking boxes instead of tight body boxes, may degrade FormicAI local-video localization style, and may improve ANTS mAP while worsening practical tight localization.

## Option B: Manually Relabel A Subset With FormicAI Tight-Box Policy

Advantages: consistent detector target, cleaner small-ant/crowding signal, less annotation-style noise.

Risks: more manual work and a smaller first dataset.

## Option C: Use ANTS Only For Validation/Testing

Advantages: avoids fixed-box training contamination and keeps ANTS diagnostic.

Risks: does not solve the lack of diverse training data.

## Option D: Mixed Strategy

Use manually relabeled tight-box subsets for training and keep original ANTS boxes as a separate diagnostic/evaluation layer. Do not silently mix fixed tracking boxes and tight body boxes as the same target.

## Recommendation

Start ants_v2 with a mixed design, but not mixed training labels yet:

1. Keep `Seq0001` and `Seq0006` frozen as external benchmark only.
2. Select non-frozen ANTS sequences for diversity.
3. Manually relabel a manageable subset with FormicAI tight-body boxes.
4. Use source-held-out validation with the same tight-box policy.
5. Keep original ANTS fixed boxes as diagnostic/evaluation metadata until an explicit policy experiment compares them.
