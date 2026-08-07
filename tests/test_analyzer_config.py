from __future__ import annotations

import pytest

from formicai.utils.video import ROI
from formicai.vision.analyzer import resolve_min_motion_area


def test_resolve_min_motion_area_uses_explicit_value() -> None:
    roi = ROI(x=0, y=0, width=1920, height=1080)

    assert (
        resolve_min_motion_area(
            roi=roi,
            min_motion_area=250.0,
            min_motion_area_ratio=0.0005,
        )
        == 250.0
    )


def test_resolve_min_motion_area_scales_with_roi() -> None:
    roi = ROI(x=0, y=0, width=1920, height=1080)

    assert (
        resolve_min_motion_area(
            roi=roi,
            min_motion_area=None,
            min_motion_area_ratio=0.0005,
        )
        == 1036.8
    )


@pytest.mark.parametrize(
    ("min_motion_area", "min_motion_area_ratio"),
    [(0.0, 0.0005), (None, 0.0)],
)
def test_resolve_min_motion_area_rejects_non_positive_values(
    min_motion_area: float | None,
    min_motion_area_ratio: float,
) -> None:
    with pytest.raises(ValueError):
        resolve_min_motion_area(
            roi=ROI(x=0, y=0, width=100, height=100),
            min_motion_area=min_motion_area,
            min_motion_area_ratio=min_motion_area_ratio,
        )
