# FormicAI Ants v3 Annotation Guide

Experiment: `ants_v3_mixedscale_e01`

## Class

- `0 = ant`

## Positive Adult Ants

Annotate these as `ant`:

- queen
- worker
- major
- minor
- replete

## Not Ant

Do not annotate these as ant:

- brood
- larvae
- pupae
- cocoons
- food
- cotton
- substrate
- reflection/glare

## Box Policy

- One physical adult ant = one box.
- Use a tight visible body box.
- Include head, thorax, and abdomen.
- Do not intentionally enlarge boxes to include antenna tips or leg tips.
- Overlap between boxes is allowed when physical ants overlap.
- For hard-negative candidates, create empty labels only after full-resolution human review confirms zero identifiable ants.

## Source Notes

- AntsNet taxonomy is metadata only; do not remap its original 70 classes automatically.
- ant2 historical labels are reference metadata only; do not copy them as new FormicAI v3 labels.
