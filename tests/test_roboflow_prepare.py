from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from formicai.dataset.roboflow import (
    DEFAULT_PREPARATION_SOURCE_SPLITS,
    ROBOFLOW_SPLITS,
    RoboflowDatasetPreparer,
    inspect_roboflow_export,
)
from formicai.dataset.validation import DatasetValidator


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def write_roboflow_record(root: Path, split: str, frame: int, timestamp: str, label: str) -> None:
    stem = f"ant2_frame_{frame:06d}_t{timestamp}_jpg.rf.hash{frame:06d}"
    write_image(root / split / "images" / f"{stem}.jpg")
    label_path = root / split / "labels" / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label, encoding="utf-8")


def create_roboflow_export(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['Ant']\n",
        encoding="utf-8",
    )
    for split in ["train", "valid", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    label = "0 0.500000 0.500000 0.250000 0.250000\n"
    write_roboflow_record(root, "valid", 21, "0000-701", label)
    write_roboflow_record(root, "train", 0, "0000-000", label)
    write_roboflow_record(root, "test", 35, "0001-168", label)
    write_roboflow_record(root, "train", 7, "0000-234", label)
    write_roboflow_record(root, "valid", 28, "0000-934", label)
    write_roboflow_record(root, "train", 14, "0000-467", label)


def create_test_only_roboflow_export(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['Ant']\n",
        encoding="utf-8",
    )
    for split in ROBOFLOW_SPLITS:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    label = "0 0.500000 0.500000 0.250000 0.250000\n"
    write_roboflow_record(root, "test", 0, "0000-000", label)


def test_inspect_roboflow_export_counts_detection_annotations(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    create_roboflow_export(export_dir)

    result = inspect_roboflow_export(export_dir)

    assert result.ok
    assert result.split_counts["train"].images == 3
    assert result.split_counts["valid"].images == 2
    assert result.split_counts["test"].images == 1
    assert result.total_images == 6
    assert result.images_with_labels == 6
    assert result.valid_annotations == 6
    assert result.source_detection_annotations == 6
    assert result.converted_segmentation_annotations == 0
    assert result.total_prepared_annotations == 6
    assert result.classes == {0: "Ant"}
    assert result.label_class_ids == {0}


def test_prepare_roboflow_temporal_split_and_normalized_dataset_yaml(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    output_dir = tmp_path / "datasets" / "prepared" / "ants_v1"
    create_roboflow_export(export_dir)

    result = RoboflowDatasetPreparer(
        export_dir=export_dir,
        output_dir=output_dir,
        train_fraction=0.67,
        gap_count=1,
    ).prepare()

    assert result.inspected_source_splits == ROBOFLOW_SPLITS
    assert result.included_source_splits == DEFAULT_PREPARATION_SOURCE_SPLITS
    assert result.excluded_source_splits == ("test",)
    assert not result.allow_test_as_training_source
    assert result.inspected_image_count == 6
    assert result.eligible_image_count == 5
    assert result.excluded_image_count == 1
    assert result.inspected_annotation_count == 6
    assert result.eligible_annotation_count == 5
    assert result.excluded_annotation_count == 1
    assert result.train_images == 3
    assert result.gap_images == 1
    assert result.val_images == 1
    assert result.train_annotations == 3
    assert result.val_annotations == 1
    assert result.gap_annotations == 1
    assert result.source_total_annotations == 5
    assert result.total_prepared_annotations == 4
    assert result.train_last_frame == 14
    assert result.val_first_frame == 28
    prepared_images = {
        image_path.name
        for split in ("train", "val")
        for image_path in (output_dir / "images" / split).glob("*.jpg")
    }
    assert not any("frame_000035" in image_name for image_name in prepared_images)
    metadata = json.loads((output_dir / "split_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sourceSplitPolicy"]["inspectedSourceSplits"] == ["train", "valid", "test"]
    assert metadata["sourceSplitPolicy"]["includedSourceSplits"] == ["train", "valid"]
    assert metadata["sourceSplitPolicy"]["excludedSourceSplits"] == ["test"]
    assert metadata["sourceSplitPolicy"]["allowTestAsTrainingSource"] is False
    assert metadata["sourceSplitPolicy"]["holdoutProtection"] == "enabled"
    assert metadata["sourceSplitPolicy"]["eligibleImageCount"] == 5
    assert metadata["sourceSplitPolicy"]["excludedImageCount"] == 1
    assert metadata["sourceSplitPolicy"]["excludedAnnotationCount"] == 1
    for split_name in ("train", "gap", "val"):
        assert all(record["sourceSplit"] != "test" for record in metadata[split_name])
    assert (output_dir / "dataset.yaml").read_text(encoding="utf-8") == (
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    assert DatasetValidator(output_dir).validate().ok


def test_prepare_roboflow_can_explicitly_override_test_holdout_protection(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    output_dir = tmp_path / "datasets" / "prepared" / "legacy"
    create_roboflow_export(export_dir)

    result = RoboflowDatasetPreparer(
        export_dir=export_dir,
        output_dir=output_dir,
        train_fraction=0.67,
        gap_count=1,
        allow_test_as_training_source=True,
    ).prepare()

    assert result.allow_test_as_training_source
    assert result.included_source_splits == ROBOFLOW_SPLITS
    assert result.excluded_source_splits == ()
    assert result.eligible_image_count == 6
    assert result.excluded_image_count == 0
    assert result.eligible_annotation_count == 6
    assert result.excluded_annotation_count == 0
    metadata = json.loads((output_dir / "split_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sourceSplitPolicy"]["allowTestAsTrainingSource"] is True
    assert metadata["sourceSplitPolicy"]["holdoutProtection"] == "overridden"
    assert metadata["sourceSplitPolicy"]["includedSourceSplits"] == ["train", "valid", "test"]
    assert any(record["sourceSplit"] == "test" for record in metadata["train"] + metadata["gap"] + metadata["val"])


def test_prepare_roboflow_fails_when_test_filtering_leaves_too_few_records(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    create_test_only_roboflow_export(export_dir)

    with pytest.raises(ValueError, match="at least two eligible timestamped images"):
        RoboflowDatasetPreparer(export_dir=export_dir, output_dir=tmp_path / "prepared").prepare()


def test_prepare_roboflow_converts_polygon_labels(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    output_dir = tmp_path / "prepared"
    create_roboflow_export(export_dir)
    polygon_label = next((export_dir / "train" / "labels").glob("*.txt"))
    polygon_label.write_text("0 0.10 0.20 0.40 0.20 0.40 0.60 0.10 0.60\n", encoding="utf-8")

    result = RoboflowDatasetPreparer(export_dir, output_dir, train_fraction=0.67, gap_count=1).prepare()

    assert result.source_detection_annotations == 4
    assert result.converted_segmentation_annotations == 1
    assert result.source_total_annotations == 5
    assert result.total_prepared_annotations == 4
    prepared_label = output_dir / "labels" / "train" / polygon_label.name
    assert prepared_label.read_text(encoding="utf-8") == "0 0.25 0.4 0.3 0.4\n"
    assert DatasetValidator(output_dir).validate().ok


def test_prepare_roboflow_rejects_invalid_polygon_labels(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    create_roboflow_export(export_dir)
    polygon_label = next((export_dir / "train" / "labels").glob("*.txt"))
    polygon_label.write_text("0 0.1 0.1 0.2 0.2 0.3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coordinate count must be even"):
        RoboflowDatasetPreparer(export_dir, tmp_path / "prepared").prepare()


def test_inspect_roboflow_export_accepts_subpixel_segmentation_boundary_rounding(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    create_roboflow_export(export_dir)
    polygon_label = next((export_dir / "train" / "labels").glob("*.txt"))
    x_edge = 1.0 + 0.49 / 32
    polygon_label.write_text(f"0 0.2 0.2 {x_edge} 0.2 {x_edge} 0.8 0.2 0.8\n", encoding="utf-8")

    result = inspect_roboflow_export(export_dir)

    assert result.ok
    assert result.converted_segmentation_annotations == 1


def test_inspect_roboflow_export_rejects_segmentation_boundary_overshoot_over_half_pixel(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    create_roboflow_export(export_dir)
    polygon_label = next((export_dir / "train" / "labels").glob("*.txt"))
    x_edge = 1.0 + 0.51 / 32
    polygon_label.write_text(f"0 0.2 0.2 {x_edge} 0.2 {x_edge} 0.8 0.2 0.8\n", encoding="utf-8")

    result = inspect_roboflow_export(export_dir)

    assert not result.ok
    assert result.invalid_annotation_lines == 1
    assert any("segmentation point" in error and "normalized between 0 and 1" in error for error in result.errors)
