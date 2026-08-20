from __future__ import annotations

from math import isfinite


def validate_inference_parameters(
    *,
    confidence: float,
    iou: float,
    image_size: int | float,
) -> None:
    if isinstance(confidence, bool):
        raise ValueError("confidence must be numeric, not boolean.")
    if isinstance(iou, bool):
        raise ValueError("iou must be numeric, not boolean.")
    if isinstance(image_size, bool):
        raise ValueError("image_size must be numeric, not boolean.")
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if not isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("iou must be between 0 and 1.")
    if not isfinite(image_size) or image_size <= 0:
        raise ValueError("image_size must be finite and greater than 0.")
