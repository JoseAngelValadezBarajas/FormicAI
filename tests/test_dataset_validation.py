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


def rewrite_dataset_yaml(dataset_dir: Path, root: str | Path, train: str = "images/train", val: str = "images/val") -> None:
    (dataset_dir / "dataset.yaml").write_text(
        f"path: {root}\ntrain: {train}\nval: {val}\nnames:\n  0: ant\n",
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


def test_dataset_validator_uses_yaml_target_instead_of_local_hardcoded_dirs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "wrapper"
    target_dir = tmp_path / "target_invalid"
    create_prepared_dataset(dataset_dir)
    create_prepared_dataset(target_dir)
    (target_dir / "labels" / "train" / "train_001.txt").unlink()
    rewrite_dataset_yaml(dataset_dir, "../target_invalid")

    result = DatasetValidator(dataset_dir).validate()

    assert not result.ok
    assert any(str(target_dir / "images" / "train" / "train_001.jpg") in error for error in result.errors)


def test_dataset_validator_accepts_yaml_target_when_local_hardcoded_dirs_are_invalid(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "wrapper"
    target_dir = tmp_path / "target_valid"
    create_prepared_dataset(dataset_dir)
    create_prepared_dataset(target_dir)
    (dataset_dir / "labels" / "train" / "train_001.txt").unlink()
    rewrite_dataset_yaml(dataset_dir, "../target_valid")

    result = DatasetValidator(dataset_dir).validate()

    assert result.ok
    assert result.train.images == 2
    assert result.val.images == 1


def test_dataset_validator_resolves_relative_yaml_path_from_yaml_location(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "configs" / "dataset"
    target_dir = tmp_path / "actual_dataset"
    dataset_dir.mkdir(parents=True)
    create_prepared_dataset(target_dir)
    rewrite_dataset_yaml(dataset_dir, "../../actual_dataset")

    result = DatasetValidator(dataset_dir).validate()

    assert result.ok
    assert result.train.annotations == 1


def test_dataset_validator_supports_absolute_yaml_path_root(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "wrapper"
    target_dir = tmp_path / "absolute_target"
    dataset_dir.mkdir()
    create_prepared_dataset(target_dir)
    rewrite_dataset_yaml(dataset_dir, target_dir.resolve())

    result = DatasetValidator(dataset_dir).validate()

    assert result.ok
    assert result.val.annotations == 1


def test_dataset_validator_rejects_unsupported_image_to_label_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "bad_layout"
    dataset_dir.mkdir()
    write_image(dataset_dir / "train_images" / "sample.jpg")
    write_image(dataset_dir / "val_images" / "sample.jpg")
    rewrite_dataset_yaml(dataset_dir, ".", train="train_images", val="val_images")

    result = DatasetValidator(dataset_dir).validate()

    assert not result.ok
    assert any("Cannot map YOLO image directory to labels directory" in error for error in result.errors)


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


def test_dataset_statistics_counts_existing_empty_label_as_zero_annotations(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ants_v1"
    create_prepared_dataset(dataset_dir)

    stats = DatasetStatisticsCalculator(dataset_dir).calculate()

    assert stats.images_without_annotations == 1
    assert stats.min_annotations_per_image == 0


def test_dataset_statistics_rejects_missing_label_file(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ants_v1"
    create_prepared_dataset(dataset_dir)
    missing_label = dataset_dir / "labels" / "train" / "train_001.txt"
    missing_label.unlink()

    with pytest.raises(ValueError) as exc_info:
        DatasetStatisticsCalculator(dataset_dir).calculate()

    message = str(exc_info.value)
    assert "Missing label for image" in message
    assert str(dataset_dir / "images" / "train" / "train_001.jpg") in message
    assert str(missing_label) in message


def _box_from_edges(x_min: float, y_min: float, x_max: float, y_max: float) -> YoloBox:
    return YoloBox(
        class_id=0,
        x_center=(x_min + x_max) / 2,
        y_center=(y_min + y_max) / 2,
        width=x_max - x_min,
        height=y_max - y_min,
    )


def test_validate_yolo_box_with_image_size_accepts_in_bounds_box() -> None:
    validate_yolo_box(YoloBox(0, 0.5, 0.5, 0.25, 0.25), image_width=400, image_height=472)


def test_validate_yolo_box_with_image_size_accepts_subpixel_right_rounding() -> None:
    validate_yolo_box(_box_from_edges(0.2, 0.2, 1.0 + 0.005 / 400, 0.8), image_width=400, image_height=472)


def test_validate_yolo_box_with_image_size_accepts_edge_overshoot_under_half_pixel() -> None:
    validate_yolo_box(_box_from_edges(0.2, 0.2, 1.0 + 0.49 / 400, 0.8), image_width=400, image_height=472)


def test_validate_yolo_box_with_image_size_rejects_edge_overshoot_over_half_pixel() -> None:
    with pytest.raises(ValueError, match="more than 0.5 pixels"):
        validate_yolo_box(_box_from_edges(0.2, 0.2, 1.0 + 0.51 / 400, 0.8), image_width=400, image_height=472)


def test_validate_yolo_box_with_image_size_applies_left_top_bottom_tolerance() -> None:
    validate_yolo_box(_box_from_edges(-0.49 / 400, 0.2, 0.8, 0.9), image_width=400, image_height=472)
    validate_yolo_box(_box_from_edges(0.2, -0.49 / 472, 0.8, 0.9), image_width=400, image_height=472)
    validate_yolo_box(_box_from_edges(0.2, 0.2, 0.8, 1.0 + 0.49 / 472), image_width=400, image_height=472)
    with pytest.raises(ValueError, match="more than 0.5 pixels"):
        validate_yolo_box(_box_from_edges(-0.51 / 400, 0.2, 0.8, 0.9), image_width=400, image_height=472)


def test_image_aware_yolo_validation_does_not_mutate_coordinates() -> None:
    box = _box_from_edges(0.2, 0.2, 1.0 + 0.49 / 400, 0.8)
    original = YoloBox(box.class_id, box.x_center, box.y_center, box.width, box.height)

    validate_yolo_box(box, image_width=400, image_height=472)

    assert box == original


def test_strict_yolo_validation_without_image_size_still_rejects_out_of_bounds_box() -> None:
    with pytest.raises(ValueError, match="outside normalized image bounds"):
        validate_yolo_box(_box_from_edges(0.2, 0.2, 1.0002, 0.8))


def test_image_aware_yolo_tolerance_scales_by_physical_pixels() -> None:
    box = _box_from_edges(0.2, 0.2, 1.0002, 0.8)

    validate_yolo_box(box, image_width=400, image_height=472)
    with pytest.raises(ValueError, match="more than 0.5 pixels"):
        validate_yolo_box(box, image_width=4000, image_height=472)
