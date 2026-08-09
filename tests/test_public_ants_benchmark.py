from __future__ import annotations

from pathlib import Path

from formicai.detection.public_ants import (
    FrameEvaluation,
    PixelBox,
    PredictionBox,
    _collect_duplicate_yolo_label_rows,
    density_analysis,
    match_center_based,
    match_iou_threshold,
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
