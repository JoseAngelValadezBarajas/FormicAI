# AUDIT-04: Evaluation Correctness + Engineering Hardening

Baseline:

```text
branch: audit/formicai-code-audit
immutable experiment baseline: ants-v4-domain-e01-final / 087c4cf8ce41790160de409fc0a1f0ab807ae42c
AUDIT-01: VERIFIED
AUDIT-02: VERIFIED
AUDIT-03: VERIFIED
```

Problems addressed:

- Custom center and fixed-IoU diagnostics used greedy one-to-one matching, which can undercount valid matches.
- `VideoDetector` and `ExternalEvaluator` did not share complete inference-parameter validation.
- Detector run provenance kept environment fields locally instead of using a reusable metadata collector.
- The repo had no simple CI workflow for the self-contained test suite.
- Future experiments needed a tracked place for small reproducibility records while keeping weights, videos, datasets, and generated artifacts ignored.

Implemented behavior:

- Custom diagnostic matching now uses deterministic maximum-cardinality bipartite matching over valid prediction/GT pairs.
- Candidate quality still orders edge exploration, but the algorithm guarantees maximum match count, not globally maximum total IoU.
- Existing diagnostic keys such as `standardIou50...` remain compatible; newly generated metadata identifies fixed-IoU50 as a custom one-to-one localization diagnostic.
- Shared inference validation rejects boolean values and enforces finite `0.0 <= confidence <= 1.0`, finite `0.0 <= iou <= 1.0`, and finite `image_size > 0` before model construction.
- `formicai.utils.environment.collect_environment_metadata()` records Python, platform, NumPy, OpenCV, Ultralytics, PyTorch, CUDA availability, and PyTorch CUDA version when available. Missing optional ML packages are recorded as `null`.
- Detector run metadata now uses the shared environment collector.
- GitHub Actions CI runs `python -m pytest -q` on push and pull requests with Python 3.11 and no GPU requirements.
- `experiments/` is reserved for future small, tracked experiment protocols, hashes, environment metadata, lock snapshots, selection metadata, and result summaries.

Reproducibility policy for future frozen experiments:

1. Capture detector/data/source SHA256 hashes before training or evaluation.
2. Capture `collect_environment_metadata()` output.
3. Capture and hash a dependency lock such as `pip freeze` from the actual run environment.
4. Store small protocol, selection, environment, and final summary records in a tracked experiment record.
5. Keep model weights, raw media, extracted images, prepared datasets, and large generated artifacts out of Git.

No training, real model inference, frozen benchmark rerun, dataset/image/label mutation, model-weight mutation, tracking work, v5 design, push, or merge occurred as part of AUDIT-04.
