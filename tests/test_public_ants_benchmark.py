from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from formicai.dataset.yolo import parse_yolo_label_file
from formicai.detection.public_ants import (
    FrameEvaluation,
    PixelBox,
    PredictionBox,
    _collect_duplicate_yolo_label_rows,
    _load_ground_truth,
    density_analysis,
    match_center_based,
    match_iou_threshold,
)


def write_image(path: Path, *, width: int = 400, height: int = 472) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def yolo_label_from_edges(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    return (
        f"0 {(x_min + x_max) / 2:.12g} {(y_min + y_max) / 2:.12g} "
        f"{x_max - x_min:.12g} {y_max - y_min:.12g}\n"
    )


def test_center_based_match_accepts_tight_prediction_inside_fixed_gt() -> None:
    gt = [PixelBox(100, 100, 200, 200)]
    prediction = [PredictionBox(PixelBox(135, 135, 165, 165), confidence=0.8)]

    matches = match_center_based(gt, prediction)
    iou_matches = match_iou_threshold(gt, prediction, threshold=0.50)

    assert matches == [(0, 0)]
    assert iou_matches == []


def test_center_based_match_keeps_one_prediction_per_gt_by_confidence() -> None:
    gt = [PixelBox(100, 100, 200, 200)]
    predictions = [
        PredictionBox(PixelBox(130, 130, 150, 150), confidence=0.4),
        PredictionBox(PixelBox(145, 145, 165, 165), confidence=0.9),
    ]

    matches = match_center_based(gt, predictions)

    assert matches == [(1, 0)]


def test_iou_match_uses_maximum_cardinality_for_greedy_failure_counterexample() -> None:
    gt = [
        PixelBox(0, 0, 3, 1),
        PixelBox(0, 0, 2, 1),
    ]
    predictions = [
        PredictionBox(PixelBox(0, 0, 3, 1), confidence=0.9),
        PredictionBox(PixelBox(0, 0, 5, 1), confidence=0.8),
    ]

    matches = match_iou_threshold(gt, predictions, threshold=0.50)

    assert len(matches) == 2
    assert set(matches) == {(0, 1), (1, 0)}


def test_center_match_uses_maximum_cardinality_for_overlapping_gt_counterexample() -> None:
    gt = [
        PixelBox(0, 0, 10, 10),
        PixelBox(4, 0, 10, 10),
    ]
    predictions = [
        PredictionBox(PixelBox(5.0, 4.9, 5.2, 5.1), confidence=0.9),
        PredictionBox(PixelBox(1.9, 4.9, 2.1, 5.1), confidence=0.8),
    ]

    matches = match_center_based(gt, predictions)

    assert len(matches) == 2
    assert set(matches) == {(0, 1), (1, 0)}


def test_matchers_return_zero_when_no_valid_edges_exist() -> None:
    gt = [PixelBox(0, 0, 1, 1)]
    predictions = [PredictionBox(PixelBox(10, 10, 11, 11), confidence=0.9)]

    assert match_center_based(gt, predictions) == []
    assert match_iou_threshold(gt, predictions, threshold=0.50) == []


def test_one_gt_with_multiple_predictions_returns_one_match() -> None:
    gt = [PixelBox(0, 0, 10, 10)]
    predictions = [
        PredictionBox(PixelBox(1, 1, 2, 2), confidence=0.7),
        PredictionBox(PixelBox(3, 3, 4, 4), confidence=0.9),
    ]

    matches = match_center_based(gt, predictions)

    assert len(matches) == 1
    assert matches == [(1, 0)]


def test_multiple_gt_with_one_prediction_returns_one_match() -> None:
    gt = [
        PixelBox(0, 0, 10, 10),
        PixelBox(4, 0, 10, 10),
    ]
    predictions = [PredictionBox(PixelBox(5.0, 4.9, 5.2, 5.1), confidence=0.9)]

    matches = match_center_based(gt, predictions)

    assert len(matches) == 1
    assert matches == [(0, 0)]


def test_perfect_independent_matches_preserve_expected_pairs() -> None:
    gt = [
        PixelBox(0, 0, 2, 2),
        PixelBox(10, 10, 12, 12),
    ]
    predictions = [
        PredictionBox(PixelBox(0, 0, 2, 2), confidence=0.8),
        PredictionBox(PixelBox(10, 10, 12, 12), confidence=0.7),
    ]

    assert match_iou_threshold(gt, predictions, threshold=0.50) == [(0, 0), (1, 1)]
    assert match_center_based(gt, predictions) == [(0, 0), (1, 1)]


def test_matching_results_are_deterministic() -> None:
    gt = [
        PixelBox(0, 0, 10, 10),
        PixelBox(4, 0, 10, 10),
    ]
    predictions = [
        PredictionBox(PixelBox(5.0, 4.9, 5.2, 5.1), confidence=0.9),
        PredictionBox(PixelBox(1.9, 4.9, 2.1, 5.1), confidence=0.8),
    ]

    first = match_center_based(gt, predictions)

    for _ in range(10):
        assert match_center_based(gt, predictions) == first


def test_density_analysis_splits_ranked_frames_into_terciles() -> None:
    frames = [
        FrameEvaluation(
            image=f"{index:06d}.jpg",
            gt_count=index,
            prediction_count=index + 1,
            center_matched=index - 1,
            center_missed=1,
            center_unmatched_predictions=2,
            iou50_matched=max(0, index - 2),
            iou50_missed=2,
            iou50_unmatched_predictions=3,
        )
        for index in range(1, 10)
    ]

    result = density_analysis(frames)

    assert result["lowDensity"]["frames"] == 3
    assert result["mediumDensity"]["frames"] == 3
    assert result["highDensity"]["frames"] == 3
    assert result["lowDensity"]["gtAnts"] == 6
    assert result["highDensity"]["gtAnts"] == 24


def test_density_analysis_reports_corrected_match_counts() -> None:
    frames = [
        FrameEvaluation(
            image="crowded.jpg",
            gt_count=2,
            prediction_count=2,
            center_matched=2,
            center_missed=0,
            center_unmatched_predictions=0,
            iou50_matched=2,
            iou50_missed=0,
            iou50_unmatched_predictions=0,
        )
    ]

    result = density_analysis(frames)

    assert result["highDensity"]["centerMatchedAnts"] == 2
    assert result["highDensity"]["standardIou50MatchedAnts"] == 2
    assert result["highDensity"]["centerRecall"] == 1.0
    assert result["highDensity"]["standardIou50Recall"] == 1.0


def test_collect_duplicate_yolo_label_rows_reports_extra_rows(tmp_path: Path) -> None:
    label_path = tmp_path / "000268.txt"
    label_path.write_text(
        "\n".join(
            [
                "0 0.1 0.2 0.3 0.4",
                "0 0.1 0.2 0.3 0.4",
                "0 0.5 0.6 0.7 0.8",
            ]
        ),
        encoding="utf-8",
    )

    duplicates = _collect_duplicate_yolo_label_rows(tmp_path)

    assert duplicates == [
        {
            "image": "000268.jpg",
            "label": str(label_path),
            "yoloRow": "0 0.1 0.2 0.3 0.4",
            "count": 2,
            "extraRows": 1,
            "note": "Ultralytics removes exact duplicate label rows during standard validation cache creation.",
        }
    ]


def test_load_ground_truth_accepts_half_pixel_boundary_rounding(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "ant4_frame_000016_t0000.533.jpg"
    label_path = labels_dir / "ant4_frame_000016_t0000.533.txt"
    write_image(image_path, width=400, height=472)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(yolo_label_from_edges(0.2, 0.2, 1.0 + 0.49 / 400, 0.8), encoding="utf-8")

    ground_truth = _load_ground_truth(images_dir, labels_dir)

    assert len(ground_truth[image_path]) == 1


def test_load_ground_truth_rejects_boundary_overshoot_over_half_pixel(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "ant4_frame_000016_t0000.533.jpg"
    label_path = labels_dir / "ant4_frame_000016_t0000.533.txt"
    write_image(image_path, width=400, height=472)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(yolo_label_from_edges(0.2, 0.2, 1.0 + 0.51 / 400, 0.8), encoding="utf-8")

    with pytest.raises(ValueError, match="more than 0.5 pixels"):
        _load_ground_truth(images_dir, labels_dir)


def test_load_ground_truth_preserves_pixel_coordinates_without_clamping(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "ant4_frame_000016_t0000.533.jpg"
    label_path = labels_dir / "ant4_frame_000016_t0000.533.txt"
    write_image(image_path, width=400, height=472)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(yolo_label_from_edges(0.2, 0.2, 1.0 + 0.49 / 400, 0.8), encoding="utf-8")

    box = _load_ground_truth(images_dir, labels_dir)[image_path][0]

    assert box.x1 == pytest.approx(80.0)
    assert box.y1 == pytest.approx(94.4)
    assert box.x2 == pytest.approx(400.49)
    assert box.y2 == pytest.approx(377.6)


def test_strict_low_level_label_parse_without_image_dimensions_remains_strict(tmp_path: Path) -> None:
    label_path = tmp_path / "label.txt"
    label_path.write_text(yolo_label_from_edges(0.2, 0.2, 1.0 + 0.49 / 400, 0.8), encoding="utf-8")

    with pytest.raises(ValueError, match="outside normalized image bounds"):
        parse_yolo_label_file(label_path)
