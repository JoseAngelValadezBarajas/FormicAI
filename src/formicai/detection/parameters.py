from __future__ import annotations


def validate_inference_parameters(
    *,
    confidence: float,
    iou: float,
    image_size: int,
) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("iou must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("image_size must be greater than 0.")
