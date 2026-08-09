from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from formicai.dataset.yolo import (
    find_images,
    matching_label_path,
    parse_dataset_yaml,
    parse_yolo_label_file,
)


@dataclass(frozen=True)
class SplitSummary:
    images: int
    annotations: int
    images_without_annotations: int


@dataclass(frozen=True)
class DatasetValidationResult:
    dataset_dir: Path
    train: SplitSummary
    val: SplitSummary
    classes: dict[int, str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_text(self) -> str:
        lines = [
            f"Dataset: {self.dataset_dir.name}",
            "",
            f"Train images: {self.train.images}",
            f"Val images:   {self.val.images}",
            "",
            f"Train annotations: {self.train.annotations}",
            f"Val annotations:   {self.val.annotations}",
            "",
            "Classes:",
        ]
        lines.extend(f"{class_id} {name}" for class_id, name in sorted(self.classes.items()))
        lines.extend(
            [
                "",
                "Images without ants: "
                f"{self.train.images_without_annotations + self.val.images_without_annotations}",
                "",
                f"Validation: {'OK' if self.ok else 'FAILED'}",
            ]
        )
        if self.errors:
            lines.append("")
            lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)


class DatasetValidator:
    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir

    def validate(self) -> DatasetValidationResult:
        errors: list[str] = []
        dataset_yaml = self._dataset_dir / "dataset.yaml"
        classes: dict[int, str] = {}

        try:
            yaml_data = parse_dataset_yaml(dataset_yaml)
            classes = self._validate_yaml(yaml_data, errors)
        except ValueError as exc:
            errors.append(str(exc))

        train = self._validate_split("train", errors)
        val = self._validate_split("val", errors)

        if train.images == 0:
            errors.append("train split must contain at least one image.")
        if val.images == 0:
            errors.append("val split must contain at least one image.")

        return DatasetValidationResult(
            dataset_dir=self._dataset_dir,
            train=train,
            val=val,
            classes=classes,
            errors=errors,
        )

    def _validate_yaml(self, yaml_data: dict[str, object], errors: list[str]) -> dict[int, str]:
        for key in ["train", "val"]:
            if not yaml_data.get(key):
                errors.append(f"dataset.yaml must define '{key}'.")

        names = yaml_data.get("names")
        if names != {0: "ant"}:
            errors.append("dataset.yaml must define exactly names: {0: ant}.")
            return {}
        return names

    def _validate_split(self, split: str, errors: list[str]) -> SplitSummary:
        images_dir = self._dataset_dir / "images" / split
        labels_dir = self._dataset_dir / "labels" / split
        if not images_dir.exists():
            errors.append(f"Missing images/{split} directory.")
        if not labels_dir.exists():
            errors.append(f"Missing labels/{split} directory.")

        images = find_images(images_dir)
        annotations = 0
        images_without_annotations = 0
        seen_label_paths: set[Path] = set()

        for image_path in images:
            label_path = matching_label_path(image_path, labels_dir)
            seen_label_paths.add(label_path)
            if not label_path.exists():
                errors.append(f"Missing label for image: {image_path}")
                continue
            try:
                boxes = parse_yolo_label_file(label_path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            annotations += len(boxes)
            if not boxes:
                images_without_annotations += 1

        if labels_dir.exists():
            for label_path in sorted(labels_dir.glob("*.txt")):
                if label_path not in seen_label_paths:
                    errors.append(f"Label without matching image: {label_path}")

        return SplitSummary(
            images=len(images),
            annotations=annotations,
            images_without_annotations=images_without_annotations,
        )
