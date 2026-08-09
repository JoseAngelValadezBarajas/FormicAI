from __future__ import annotations

from pathlib import Path

import pytest

from formicai.dataset.ants_mendeley import (
    AntsGroundTruthRow,
    ants_row_to_yolo_box,
    collect_duplicate_track_id_errors,
    frame_to_image_name,
    parse_ants_ground_truth_line,
)
from formicai.dataset.yolo import YoloBox


def test_parse_ants_ground_truth_line_accepts_valid_row() -> None:
    row = parse_ants_ground_truth_line("1,11,740.603,48.563,94,94,1", Path("gt.txt"), 1)

    assert row == AntsGroundTruthRow(
        frame=1,
        track_id=11,
        x=740.603,
        y=48.563,
        width=94.0,
        height=94.0,
        flag=1,
    )


def test_parse_ants_ground_truth_line_rejects_invalid_column_count() -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        parse_ants_ground_truth_line("1,11,740.603,48.563,94,94")


def test_parse_ants_ground_truth_line_rejects_invalid_numeric_value() -> None:
    with pytest.raises(ValueError, match="x must be numeric"):
        parse_ants_ground_truth_line("1,11,nope,48.563,94,94,1")


def test_parse_ants_ground_truth_line_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError, match="frame must be greater than or equal to 1"):
        parse_ants_ground_truth_line("0,11,740.603,48.563,94,94,1")


def test_parse_ants_ground_truth_line_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="width must be greater than 0"):
        parse_ants_ground_truth_line("1,11,740.603,48.563,0,94,1")


def test_parse_ants_ground_truth_line_rejects_zero_height() -> None:
    with pytest.raises(ValueError, match="height must be greater than 0"):
        parse_ants_ground_truth_line("1,11,740.603,48.563,94,0,1")


def test_ants_row_to_yolo_box_normalizes_known_example() -> None:
    row = AntsGroundTruthRow(
        frame=1,
        track_id=7,
        x=100,
        y=50,
        width=200,
        height=100,
        flag=1,
    )

    box = ants_row_to_yolo_box(row, image_width=1000, image_height=500)

    assert box == YoloBox(
        class_id=0,
        x_center=pytest.approx(0.20),
        y_center=pytest.approx(0.20),
        width=pytest.approx(0.20),
        height=pytest.approx(0.20),
    )


def test_ants_row_to_yolo_box_rejects_box_outside_image() -> None:
    row = AntsGroundTruthRow(
        frame=1,
        track_id=7,
        x=900,
        y=50,
        width=200,
        height=100,
        flag=1,
    )

    with pytest.raises(ValueError, match="extends outside image bounds"):
        ants_row_to_yolo_box(row, image_width=1000, image_height=500)


def test_frame_to_image_name_maps_one_based_frames_to_six_digit_jpegs() -> None:
    assert frame_to_image_name(1) == "000001.jpg"
    assert frame_to_image_name(351) == "000351.jpg"


def test_collect_duplicate_track_id_errors_detects_same_frame_identity() -> None:
    rows = [
        AntsGroundTruthRow(1, 10, 0, 0, 10, 10, 1),
        AntsGroundTruthRow(1, 10, 20, 20, 10, 10, 1),
        AntsGroundTruthRow(2, 10, 30, 30, 10, 10, 1),
    ]

    errors = collect_duplicate_track_id_errors(rows)

    assert errors == ["frame 1 track_id 10 appears 2 times"]
