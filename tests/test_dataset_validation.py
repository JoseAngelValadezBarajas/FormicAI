from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from formicai.dataset.statistics import DatasetStatisticsCalculator
from formicai.dataset.validation import DatasetValidator
from formicai.dataset.yolo import (
    YoloBox,
    collect_yolo_label_errors,
    parse_yolo_label_file,
    parse_yolo_detection_or_segmentation_line,
    segmentation_points_to_box,
    validate_yolo_box,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def create_prepared_dataset(root: Path) -> None:
    for split in ["train", "val"]:
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    write_image(root / "images" / "train" / "train_001.jpg")
    write_image(root / "images" / "train" / "train_002.jpg")
    write_image(root / "images" / "val" / "val_001.jpg")

    (root / "labels" / "train" / "train_001.txt").write_text(
        "0 0.500000 0.500000 0.250000 0.250000\n",
        encoding="utf-8",
    )
    (root / "labels" / "train" / "train_002.txt").write_text("", encoding="utf-8")
    (root / "labels" / "val" / "val_001.txt").write_text(
        "0 0.400000 0.400000 0.200000 0.100000\n",
        encoding="utf-8",
    )
    (root / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n",
        encoding="utf-8",
    )


def test_parse_yolo_label_file() -> None:
    path = Path("label.txt")
    tmp = Path.cwd() / path
    try:
        tmp.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        boxes = parse_yolo_label_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    assert boxes == [YoloBox(0, 0.5, 0.5, 0.25, 0.25)]


def test_collect_yolo_label_errors_reports_all_bad_lines(tmp_path: Path) -> None:
    label_path = tmp_path / "label.txt"
    label_path.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.25 0.25",
                "0 0.1 0.1 0.2 0.2 0.3 0.3",
                "1 0.5 0.5 0.2 0.2",
            ]
        ),
        encoding="utf-8",
    )

    boxes, errors = collect_yolo_label_errors(label_path)

    assert boxes == [YoloBox(0, 0.5, 0.5, 0.25, 0.25)]
    assert len(errors) == 2
    assert "must contain 5 values; found 7" in errors[0]
    assert "class id must be 0" in errors[1]


def test_validate_yolo_box_rejects_out_of_bounds() -> None:
    try:
        validate_yolo_box(YoloBox(0, 0.95, 0.5, 0.2, 0.2))
    except ValueError as exc:
        assert "outside normalized image bounds" in str(exc)
    else:
        raise AssertionError("Expected invalid box to raise ValueError.")


def test_parse_yolo_segmentation_polygon_converts_to_bounding_box() -> None:
    annotation = parse_yolo_detection_or_segmentation_line(
        "0 0.10 0.20 0.40 0.20 0.40 0.60 0.10 0.60"
    )

    assert annotation.annotation_format == "segmentation"
    assert annotation.box.class_id == 0
    assert annotation.box.x_center == pytest.approx(0.25)
    assert annotation.box.y_center == pytest.approx(0.4)
    assert annotation.box.width == pytest.approx(0.3)
    assert annotation.box.height == pytest.approx(0.4)


def test_segmentation_points_to_box_conversion() -> None:
    box = segmentation_points_to_box(
        0,
        [
            (0.2, 0.3),
            (0.5, 0.4),
            (0.4, 0.7),
        ],
    )

    assert box == YoloBox(0, 0.35, 0.5, 0.3, 0.39999999999999997)


def test_parse_yolo_segmentation_rejects_odd_coordinate_count() -> None:
    with pytest.raises(ValueError, match="coordinate count must be even"):
        parse_yolo_detection_or_segmentation_line("0 0.1 0.1 0.2 0.2 0.3")


def test_parse_yolo_segmentation_rejects_out_of_range_coordinate() -> None:
    with pytest.raises(ValueError, match="normalized between 0 and 1"):
        parse_yolo_detection_or_segmentation_line("0 0.1 0.1 1.2 0.2 0.3 0.3")


def test_parse_yolo_segmentation_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        parse_yolo_detection_or_segmentation_line("0 0.1 0.1 0.2 0.2 0.3 0.3")


def test_dataset_validator_accepts_valid_prepared_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ants_v1"
    create_prepared_dataset(dataset_dir)

    result = DatasetValidator(dataset_dir).validate()

    assert result.ok
    assert result.train.images == 2
    assert result.val.images == 1
    assert result.train.annotations == 1
    assert result.val.annotations == 1
    assert result.train.images_without_annotations == 1


def test_dataset_validator_rejects_missing_label(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ants_v1"
    create_prepared_dataset(dataset_dir)
    (dataset_dir / "labels" / "val" / "val_001.txt").unlink()

    result = DatasetValidator(dataset_dir).validate()

    assert not result.ok
    assert any("Missing label" in error for error in result.errors)


def test_dataset_validator_requires_train_and_val_in_yaml(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ants_v1"
    create_prepared_dataset(dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        "path: .\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    result = DatasetValidator(dataset_dir).validate()

    assert not result.ok
    assert "dataset.yaml must define 'train'." in result.errors
    assert "dataset.yaml must define 'val'." in result.errors


def test_dataset_statistics_calculates_box_ratios(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ants_v1"
    create_prepared_dataset(dataset_dir)

    stats = DatasetStatisticsCalculator(dataset_dir).calculate()

    assert stats.images == 3
    assert stats.train_images == 2
    assert stats.val_images == 1
    assert stats.annotations == 2
    assert stats.train_annotations == 1
    assert stats.val_annotations == 1
    assert stats.images_without_annotations == 1
    assert stats.median_box_width_ratio == 0.225
    assert stats.median_box_height_ratio == 0.175
    assert stats.median_box_area_ratio == 0.04125
    assert stats.p25_box_area_ratio == pytest.approx(0.030625)
    assert stats.p75_box_area_ratio == pytest.approx(0.051875)
