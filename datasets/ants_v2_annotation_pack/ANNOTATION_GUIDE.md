# ants_v2_diverse_e01 Annotation Guide

## Class Policy

Use exactly one class:

```text
0 ant
```

Queen, worker, male, species, size, and color differences are ignored for this detector. If it is an adult ant, label it as `ant`.

## Box Policy

- Draw a tight bounding box around the visible ant body.
- Include head, thorax, and abdomen.
- Do not enlarge the box just to include long antennae or legs.
- One box per physical ant.
- For a partially visible ant, annotate it if its identity as an ant is clear.
- For a heavily occluded or ambiguous object, do not guess.

## Do Not Label

- Rocks.
- Shadows.
- Holes or nest entrances.
- Food or substrate.
- Brood, larvae, eggs, or pupae unless the visible object is actually an adult ant.
- Reflections, stains, debris, or background texture.

## Notes

The original ANTS tracking boxes were used only to choose frames and estimate annotation workload. They are fixed-size tracking boxes and must not be copied as training labels. New labels for this pack must be manually drawn with FormicAI tight-body boxes.
