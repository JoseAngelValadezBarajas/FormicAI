from __future__ import annotations

import argparse

import pytest

from formicai.utils.video import ROI, full_frame_roi, parse_roi


def test_parse_roi_accepts_x_y_width_height() -> None:
    assert parse_roi("10,20,300,400") == ROI(x=10, y=20, width=300, height=400)


@pytest.mark.parametrize(
    "value",
    ["10,20,30", "10,20,30,40,50", "10,a,30,40", "-1,0,10,10", "0,0,0,10"],
)
def test_parse_roi_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_roi(value)


def test_roi_validation_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        ROI(x=90, y=0, width=20, height=20).validate_within(
            frame_width=100,
            frame_height=100,
        )


def test_full_frame_roi_matches_dimensions() -> None:
    assert full_frame_roi(frame_width=1920, frame_height=1080) == ROI(
        x=0,
        y=0,
        width=1920,
        height=1080,
    )
