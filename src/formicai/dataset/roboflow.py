from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from formicai.dataset.split import temporal_split
from formicai.dataset.yolo import (
    IMAGE_EXTENSIONS,
    ParsedYoloAnnotation,
    collect_yolo_detection_or_segmentation_errors,
    find_images,
    format_yolo_box,
    matching_label_path,
    parse_dataset_yaml,
)


ROBOFLOW_SPLITS = ("train", "valid", "test")
DEFAULT_PREPARATION_SOURCE_SPLITS = ("train", "valid")
FRAME_PATTERN = re.compile(r"_frame_(?P<frame>\d+)_t(?P<seconds>\d+)[-.](?P<fraction>\d+)")


@dataclass(frozen=True)
class ExportSplitCount:
    images: int
    labels: int


@dataclass(frozen=True)
class ExportRecord:
    source_split: str
    image_path: Path
    label_path: Path
    frame_index: int
    timestamp_seconds: float
    annotations: tuple[ParsedYoloAnnotation, ...]


@dataclass(frozen=True)
class RoboflowExportInspection:
    export_dir: Path
    split_counts: dict[str, ExportSplitCount]
    classes: dict[int, str]
    label_class_ids: set[int]
    records: list[ExportRecord]
    images_with_labels: int
    images_without_annotations: int
    source_detection_annotations: int
    converted_segmentation_annotations: int
    valid_annotations: int
    invalid_annotation_lines: int
    missing_label_files: list[Path]
    extra_label_files: list[Path]
    errors: list[str]

    @property
    def total_images(self) -> int:
        return sum(count.images for count in self.split_counts.values())

    @property
    def total_labels(self) -> int:
        return sum(count.labels for count in self.split_counts.values())

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total_prepared_annotations(self) -> int:
        return self.source_detection_annotations + self.converted_segmentation_annotations

    def to_text(self) -> str:
        lines = [f"Export: {self.export_dir}", ""]
        for split in ROBOFLOW_SPLITS:
            count = self.split_counts.get(split, ExportSplitCount(images=0, labels=0))
            lines.append(f"{split} images: {count.images}")
            lines.append(f"{split} labels: {count.labels}")
        lines.extend(
            [
                "",
                f"Total images: {self.total_images}",
                f"Total labels: {self.total_labels}",
                f"Images with non-empty labels: {self.images_with_labels}",
                f"Images without annotations: {self.images_without_annotations}",
                f"Detection annotations preserved: {self.source_detection_annotations}",
                f"Segmentation annotations converted: {self.converted_segmentation_annotations}",
                f"Total valid annotations after conversion: {self.total_prepared_annotations}",
                f"Invalid annotation lines: {self.invalid_annotation_lines}",
                f"Missing label files: {len(self.missing_label_files)}",
                f"Extra label files: {len(self.extra_label_files)}",
                "",
                "Classes from data.yaml:",
            ]
        )
        if self.classes:
            lines.extend(f"{class_id} {name}" for class_id, name in sorted(self.classes.items()))
        else:
            lines.append("(none)")
        lines.extend(
            [
                "",
                "Class IDs found in labels:",
                ", ".join(str(class_id) for class_id in sorted(self.label_class_ids)) or "(none)",
                "",
                f"Validation: {'OK' if self.ok else 'FAILED'}",
            ]
        )
        if self.errors:
            lines.append("")
            lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)


@dataclass(frozen=True)
class PreparedDatasetResult:
    output_dir: Path
    dataset_yaml: Path
    split_metadata: Path
    train_images: int
    val_images: int
    gap_images: int
    train_annotations: int
    val_annotations: int
    gap_annotations: int
    source_detection_annotations: int
    converted_segmentation_annotations: int
    source_total_annotations: int
    total_prepared_annotations: int
    inspected_source_splits: tuple[str, ...]
    included_source_splits: tuple[str, ...]
    excluded_source_splits: tuple[str, ...]
    allow_test_as_training_source: bool
    inspected_image_count: int
    eligible_image_count: int
    excluded_image_count: int
    inspected_annotation_count: int
    eligible_annotation_count: int
    excluded_annotation_count: int
    train_first_frame: int | None
    train_last_frame: int | None
    val_first_frame: int | None
    val_last_frame: int | None
    train_first_timestamp_seconds: float | None
    train_last_timestamp_seconds: float | None
    val_first_timestamp_seconds: float | None
    val_last_timestamp_seconds: float | None
    gap_frames: tuple[int, ...]

    def to_text(self) -> str:
        return "\n".join(
            [
                f"Prepared dataset: {self.output_dir}",
                f"dataset.yaml: {self.dataset_yaml}",
                f"split metadata: {self.split_metadata}",
                f"Train images: {self.train_images}",
                f"Val images: {self.val_images}",
                f"Gap images omitted: {self.gap_images}",
                f"Inspected source splits: {', '.join(self.inspected_source_splits)}",
                f"Included source splits: {', '.join(self.included_source_splits)}",
                f"Excluded source splits: {', '.join(self.excluded_source_splits) or '(none)'}",
                f"Allow test as training source: {self.allow_test_as_training_source}",
                f"Inspected images: {self.inspected_image_count}",
                f"Eligible images: {self.eligible_image_count}",
                f"Excluded images: {self.excluded_image_count}",
                f"Train annotations: {self.train_annotations}",
                f"Val annotations: {self.val_annotations}",
                f"Gap annotations omitted: {self.gap_annotations}",
                f"Detection annotations preserved: {self.source_detection_annotations}",
                f"Segmentation annotations converted: {self.converted_segmentation_annotations}",
                f"Inspected annotations after conversion: {self.inspected_annotation_count}",
                f"Eligible annotations after conversion: {self.eligible_annotation_count}",
                f"Excluded annotations after conversion: {self.excluded_annotation_count}",
                f"Source annotations after conversion: {self.source_total_annotations}",
                f"Prepared train+val annotations: {self.total_prepared_annotations}",
                f"Train first frame: {self.train_first_frame}",
                f"Train last frame: {self.train_last_frame}",
                f"Val first frame: {self.val_first_frame}",
                f"Val last frame: {self.val_last_frame}",
                f"Gap frames: {', '.join(str(frame) for frame in self.gap_frames) or '(none)'}",
            ]
        )


class RoboflowDatasetPreparer:
    def __init__(
        self,
        export_dir: Path,
        output_dir: Path,
        train_fraction: float = 0.8,
        gap_count: int = 1,
        allow_test_as_training_source: bool = False,
    ) -> None:
        self._export_dir = export_dir
        self._output_dir = output_dir
        self._train_fraction = train_fraction
        self._gap_count = gap_count
        self._allow_test_as_training_source = allow_test_as_training_source

    def prepare(self) -> PreparedDatasetResult:
        inspection = inspect_roboflow_export(self._export_dir)
        included_source_splits = _included_preparation_source_splits(self._allow_test_as_training_source)
        excluded_source_splits = tuple(split for split in ROBOFLOW_SPLITS if split not in included_source_splits)
        eligible_records = [
            record for record in inspection.records if record.source_split in included_source_splits
        ]
        excluded_records = [
            record for record in inspection.records if record.source_split not in included_source_splits
        ]
        eligible_detection_annotations = _annotation_format_count(eligible_records, "detection")
        eligible_converted_annotations = _annotation_format_count(eligible_records, "segmentation")
        eligible_annotations = _annotation_count(eligible_records)
        excluded_annotations = _annotation_count(excluded_records)

        errors = list(inspection.errors)
        if _normalized_classes(inspection.classes) != {0: "ant"}:
            errors.append("data.yaml must contain exactly one class, 0 = ant.")
        if len(eligible_records) < 2:
            excluded = ", ".join(excluded_source_splits) or "(none)"
            included = ", ".join(included_source_splits)
            errors.append(
                "Roboflow export must contain at least two eligible timestamped images "
                f"after applying source split protection. Included source splits: {included}. "
                f"Excluded source splits: {excluded}."
            )
        if errors:
            message = "Roboflow export is not a valid YOLO detection dataset."
            details = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"{message}\n{details}")

        split = temporal_split(
            eligible_records,
            train_fraction=self._train_fraction,
            gap_count=self._gap_count,
            timestamp_key="timestamp_seconds",
        )
        train_records = split["train"]
        val_records = split["val"]
        gap_records = split["gap"]

        _clear_prepared_files(self._output_dir)
        for split_name, records in [("train", train_records), ("val", val_records)]:
            for record in records:
                shutil.copy2(record.image_path, self._output_dir / "images" / split_name / record.image_path.name)
                _write_prepared_label(record, self._output_dir / "labels" / split_name / record.label_path.name)

        dataset_yaml = self._output_dir / "dataset.yaml"
        dataset_yaml.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n",
            encoding="utf-8",
        )
        train_annotations = _annotation_count(train_records)
        val_annotations = _annotation_count(val_records)
        gap_annotations = _annotation_count(gap_records)
        prepared_annotations = train_annotations + val_annotations
        split_metadata = self._output_dir / "split_metadata.json"
        split_metadata.write_text(
            json.dumps(
                {
                    "sourceExport": str(self._export_dir),
                    "strategy": (
                        f"Merged eligible Roboflow source splits ({', '.join(included_source_splits)}), "
                        "sorted by timestamp from filename, "
                        f"used first {self._train_fraction:.0%} for train and last "
                        f"{1.0 - self._train_fraction:.0%} for val with {len(gap_records)} "
                        "boundary frame(s) omitted as temporal gap."
                    ),
                    "sourceSplitPolicy": _source_split_policy_json(
                        inspection=inspection,
                        eligible_records=eligible_records,
                        excluded_records=excluded_records,
                        included_source_splits=included_source_splits,
                        excluded_source_splits=excluded_source_splits,
                        allow_test_as_training_source=self._allow_test_as_training_source,
                    ),
                    "trainFraction": self._train_fraction,
                    "gapCountRequested": self._gap_count,
                    "sourceDetectionAnnotations": eligible_detection_annotations,
                    "convertedSegmentationAnnotations": eligible_converted_annotations,
                    "sourceTotalAnnotations": eligible_annotations,
                    "inspectedAnnotations": inspection.total_prepared_annotations,
                    "excludedAnnotations": excluded_annotations,
                    "gapAnnotations": gap_annotations,
                    "totalPreparedAnnotations": prepared_annotations,
                    "invalidAnnotations": inspection.invalid_annotation_lines,
                    "trainImages": len(train_records),
                    "valImages": len(val_records),
                    "gapImages": len(gap_records),
                    "trainAnnotations": train_annotations,
                    "valAnnotations": val_annotations,
                    "temporalRanges": {
                        "train": _temporal_range(train_records),
                        "gap": _temporal_range(gap_records),
                        "val": _temporal_range(val_records),
                    },
                    "train": [_record_to_json(record) for record in train_records],
                    "gap": [_record_to_json(record) for record in gap_records],
                    "val": [_record_to_json(record) for record in val_records],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return PreparedDatasetResult(
            output_dir=self._output_dir,
            dataset_yaml=dataset_yaml,
            split_metadata=split_metadata,
            train_images=len(train_records),
            val_images=len(val_records),
            gap_images=len(gap_records),
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            gap_annotations=gap_annotations,
            source_detection_annotations=eligible_detection_annotations,
            converted_segmentation_annotations=eligible_converted_annotations,
            source_total_annotations=eligible_annotations,
            total_prepared_annotations=prepared_annotations,
            inspected_source_splits=ROBOFLOW_SPLITS,
            included_source_splits=included_source_splits,
            excluded_source_splits=excluded_source_splits,
            allow_test_as_training_source=self._allow_test_as_training_source,
            inspected_image_count=inspection.total_images,
            eligible_image_count=len(eligible_records),
            excluded_image_count=len(excluded_records),
            inspected_annotation_count=inspection.total_prepared_annotations,
            eligible_annotation_count=eligible_annotations,
            excluded_annotation_count=excluded_annotations,
            train_first_frame=train_records[0].frame_index if train_records else None,
            train_last_frame=train_records[-1].frame_index if train_records else None,
            val_first_frame=val_records[0].frame_index if val_records else None,
            val_last_frame=val_records[-1].frame_index if val_records else None,
            train_first_timestamp_seconds=train_records[0].timestamp_seconds if train_records else None,
            train_last_timestamp_seconds=train_records[-1].timestamp_seconds if train_records else None,
            val_first_timestamp_seconds=val_records[0].timestamp_seconds if val_records else None,
            val_last_timestamp_seconds=val_records[-1].timestamp_seconds if val_records else None,
            gap_frames=tuple(record.frame_index for record in gap_records),
        )


def inspect_roboflow_export(export_dir: Path) -> RoboflowExportInspection:
    errors: list[str] = []
    split_counts: dict[str, ExportSplitCount] = {}
    records: list[ExportRecord] = []
    classes: dict[int, str] = {}
    label_class_ids: set[int] = set()
    images_with_labels = 0
    images_without_annotations = 0
    source_detection_annotations = 0
    converted_segmentation_annotations = 0
    valid_annotations = 0
    invalid_annotation_lines = 0
    missing_label_files: list[Path] = []
    extra_label_files: list[Path] = []

    if not export_dir.exists():
        raise FileNotFoundError(f"Roboflow export does not exist: {export_dir}")

    data_yaml = export_dir / "data.yaml"
    try:
        classes = _read_classes(data_yaml)
    except ValueError as exc:
        errors.append(str(exc))

    for split in ROBOFLOW_SPLITS:
        images_dir = export_dir / split / "images"
        labels_dir = export_dir / split / "labels"
        if not images_dir.exists():
            errors.append(f"Missing directory: {images_dir}")
        if not labels_dir.exists():
            errors.append(f"Missing directory: {labels_dir}")

        images = find_images(images_dir)
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        split_counts[split] = ExportSplitCount(images=len(images), labels=len(labels))

        image_stems = {image_path.stem for image_path in images}

        for image_path in images:
            frame_data = _parse_frame_data(image_path)
            if frame_data is None:
                errors.append(f"Could not parse frame index/timestamp from filename: {image_path}")
                continue

            label_path = matching_label_path(image_path, labels_dir)
            if not label_path.exists():
                missing_label_files.append(label_path)
                errors.append(f"Missing label for image: {image_path}")
                continue

            annotations, label_errors = collect_yolo_detection_or_segmentation_errors(label_path)
            for class_id in _class_ids_from_label_file(label_path):
                label_class_ids.add(class_id)
            source_detection_annotations += sum(
                1 for annotation in annotations if annotation.annotation_format == "detection"
            )
            converted_segmentation_annotations += sum(
                1 for annotation in annotations if annotation.annotation_format == "segmentation"
            )
            valid_annotations += len(annotations)
            invalid_annotation_lines += len(label_errors)
            errors.extend(label_errors)
            if _has_nonempty_lines(label_path):
                images_with_labels += 1
            else:
                images_without_annotations += 1

            records.append(
                ExportRecord(
                    source_split=split,
                    image_path=image_path,
                    label_path=label_path,
                    frame_index=frame_data[0],
                    timestamp_seconds=frame_data[1],
                    annotations=tuple(annotations),
                )
            )

        for label_path in labels:
            if label_path.stem not in image_stems:
                extra_label_files.append(label_path)
                errors.append(f"Label without matching image: {label_path}")

    records = sorted(records, key=lambda record: (record.timestamp_seconds, record.frame_index, record.image_path.name))
    return RoboflowExportInspection(
        export_dir=export_dir,
        split_counts=split_counts,
        classes=classes,
        label_class_ids=label_class_ids,
        records=records,
        images_with_labels=images_with_labels,
        images_without_annotations=images_without_annotations,
        source_detection_annotations=source_detection_annotations,
        converted_segmentation_annotations=converted_segmentation_annotations,
        valid_annotations=valid_annotations,
        invalid_annotation_lines=invalid_annotation_lines,
        missing_label_files=missing_label_files,
        extra_label_files=extra_label_files,
        errors=errors,
    )


def _read_classes(data_yaml: Path) -> dict[int, str]:
    yaml_data = parse_dataset_yaml(data_yaml)
    names = yaml_data.get("names")
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, str):
        try:
            parsed_names = ast.literal_eval(names)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Unable to parse class names from {data_yaml}") from exc
        if isinstance(parsed_names, list):
            return {class_id: str(name) for class_id, name in enumerate(parsed_names)}
        if isinstance(parsed_names, dict):
            return {int(class_id): str(name) for class_id, name in parsed_names.items()}
    raise ValueError(f"Unable to parse class names from {data_yaml}")


def _normalized_classes(classes: dict[int, str]) -> dict[int, str]:
    return {class_id: name.strip().lower() for class_id, name in classes.items()}


def _parse_frame_data(image_path: Path) -> tuple[int, float] | None:
    match = FRAME_PATTERN.search(image_path.name)
    if match is None:
        return None
    seconds = int(match.group("seconds"))
    fraction = match.group("fraction")
    timestamp_seconds = float(f"{seconds}.{fraction}")
    return int(match.group("frame")), timestamp_seconds


def _class_ids_from_label_file(label_path: Path) -> set[int]:
    class_ids: set[int] = set()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        try:
            class_ids.add(int(parts[0]))
        except ValueError:
            continue
    return class_ids


def _has_nonempty_lines(label_path: Path) -> bool:
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def _clear_prepared_files(output_dir: Path) -> None:
    for split in ["train", "val"]:
        for subdir, extensions in [("images", IMAGE_EXTENSIONS), ("labels", {".txt"})]:
            target = output_dir / subdir / split
            target.mkdir(parents=True, exist_ok=True)
            for file_path in target.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    file_path.unlink()


def _write_prepared_label(record: ExportRecord, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_yolo_box(annotation.box) for annotation in record.annotations]
    target_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _annotation_count(records: list[ExportRecord]) -> int:
    return sum(len(record.annotations) for record in records)


def _annotation_format_count(records: list[ExportRecord], annotation_format: str) -> int:
    return sum(
        1
        for record in records
        for annotation in record.annotations
        if annotation.annotation_format == annotation_format
    )


def _included_preparation_source_splits(allow_test_as_training_source: bool) -> tuple[str, ...]:
    if allow_test_as_training_source:
        return ROBOFLOW_SPLITS
    return DEFAULT_PREPARATION_SOURCE_SPLITS


def _source_split_policy_json(
    *,
    inspection: RoboflowExportInspection,
    eligible_records: list[ExportRecord],
    excluded_records: list[ExportRecord],
    included_source_splits: tuple[str, ...],
    excluded_source_splits: tuple[str, ...],
    allow_test_as_training_source: bool,
) -> dict[str, object]:
    return {
        "inspectedSourceSplits": list(ROBOFLOW_SPLITS),
        "includedSourceSplits": list(included_source_splits),
        "excludedSourceSplits": list(excluded_source_splits),
        "allowTestAsTrainingSource": allow_test_as_training_source,
        "holdoutProtection": "overridden" if allow_test_as_training_source else "enabled",
        "inspectedImageCount": inspection.total_images,
        "inspectedRecordCount": len(inspection.records),
        "eligibleImageCount": len(eligible_records),
        "excludedImageCount": len(excluded_records),
        "inspectedAnnotationCount": inspection.total_prepared_annotations,
        "eligibleAnnotationCount": _annotation_count(eligible_records),
        "excludedAnnotationCount": _annotation_count(excluded_records),
        "sourceSplitCounts": {
            split: {
                "images": inspection.split_counts.get(split, ExportSplitCount(images=0, labels=0)).images,
                "labels": inspection.split_counts.get(split, ExportSplitCount(images=0, labels=0)).labels,
            }
            for split in ROBOFLOW_SPLITS
        },
        "excludedRecords": [_record_to_json(record) for record in excluded_records],
    }


def _temporal_range(records: list[ExportRecord]) -> dict[str, object] | None:
    if not records:
        return None
    return {
        "firstFrame": records[0].frame_index,
        "lastFrame": records[-1].frame_index,
        "firstTimestampSeconds": records[0].timestamp_seconds,
        "lastTimestampSeconds": records[-1].timestamp_seconds,
    }


def _record_to_json(record: ExportRecord) -> dict[str, object]:
    return {
        "sourceSplit": record.source_split,
        "image": str(record.image_path),
        "label": str(record.label_path),
        "frameIndex": record.frame_index,
        "timestampSeconds": record.timestamp_seconds,
        "detectionAnnotations": sum(
            1 for annotation in record.annotations if annotation.annotation_format == "detection"
        ),
        "convertedSegmentationAnnotations": sum(
            1 for annotation in record.annotations if annotation.annotation_format == "segmentation"
        ),
        "totalAnnotations": len(record.annotations),
    }
