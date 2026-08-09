from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from formicai.dataset.roboflow import RoboflowDatasetPreparer, inspect_roboflow_export
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


def test_inspect_roboflow_export_counts_detection_annotations(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    create_roboflow_export(export_dir)

    result = inspect_roboflow_export(export_dir)

    assert result.ok
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

    assert result.train_images == 4
    assert result.gap_images == 1
    assert result.val_images == 1
    assert result.train_annotations == 4
    assert result.val_annotations == 1
    assert result.gap_annotations == 1
    assert result.source_total_annotations == 6
    assert result.total_prepared_annotations == 5
    assert result.train_last_frame == 21
    assert result.val_first_frame == 35
    assert (output_dir / "dataset.yaml").read_text(encoding="utf-8") == (
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    assert DatasetValidator(output_dir).validate().ok


def test_prepare_roboflow_converts_polygon_labels(tmp_path: Path) -> None:
    export_dir = tmp_path / "exported_dataset"
    output_dir = tmp_path / "prepared"
    create_roboflow_export(export_dir)
    polygon_label = next((export_dir / "train" / "labels").glob("*.txt"))
    polygon_label.write_text("0 0.10 0.20 0.40 0.20 0.40 0.60 0.10 0.60\n", encoding="utf-8")

    result = RoboflowDatasetPreparer(export_dir, output_dir, train_fraction=0.67, gap_count=1).prepare()

    assert result.source_detection_annotations == 5
    assert result.converted_segmentation_annotations == 1
    assert result.source_total_annotations == 6
    assert result.total_prepared_annotations == 5
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
