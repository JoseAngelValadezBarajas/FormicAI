# ants_v3_mixedscale_e01 Hard Negative Plan

Status: proposal only. No hard-negative images have been materialized or labeled in this step.

## Rule

A hard-negative image may have zero labels only if a human verifies at full resolution that there are no identifiable ants in the frame. Never leave visible ants unlabeled just to manufacture a negative.

## Candidate Source

`nest_test_001` is the primary candidate because sampled frames show soil, rocks, cotton, plastic tube, container edges and glare/reflections without identifiable ants. It is not suitable as own-colony validation because it appears to lack positive queen/worker examples.

Proposed initial target: 10-12 verified hard-negative frames from `nest_test_001`. Expand to 10-20 only if manual review confirms zero-ant frames with useful distractor diversity.

## Distractors To Cover

- substrate and rocks
- cotton
- plastic tube
- container walls and edges
- glare/reflections
- nest entrance-like shapes

## Exclusions

- Do not use `ant4` for hard negatives; keep it frozen as own-colony final test candidate.
- Do not use `ant3` for train negatives in e01; keep it as stress test only.
- Do not use any frame with an identifiable ant as a zero-label negative.
