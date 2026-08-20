from __future__ import annotations

import math

import pytest

from formicai.detection.parameters import validate_inference_parameters


@pytest.mark.parametrize("image_size", [math.nan, math.inf, -math.inf, 0, -1])
def test_validate_inference_parameters_rejects_invalid_image_size(image_size: float | int) -> None:
    with pytest.raises(ValueError, match="image_size must be finite and greater than 0"):
        validate_inference_parameters(confidence=0.25, iou=0.70, image_size=image_size)


@pytest.mark.parametrize("image_size", [640, 1280, 640.0])
def test_validate_inference_parameters_accepts_positive_finite_image_size(image_size: float | int) -> None:
    validate_inference_parameters(confidence=0.25, iou=0.70, image_size=image_size)


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -math.inf])
def test_validate_inference_parameters_rejects_non_finite_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        validate_inference_parameters(confidence=confidence, iou=0.70, image_size=640)


@pytest.mark.parametrize("iou", [math.nan, math.inf, -math.inf])
def test_validate_inference_parameters_rejects_non_finite_iou(iou: float) -> None:
    with pytest.raises(ValueError, match="iou must be between 0 and 1"):
        validate_inference_parameters(confidence=0.25, iou=iou, image_size=640)
