from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from formicai.dataset.yolo import (
    find_images,
    matching_label_path,
    parse_dataset_yaml,
    parse_yolo_label_file,
    yolo_labels_dir_for_images_dir,
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


@dataclass(frozen=True)
class ResolvedSplitPaths:
    name: str
    images_dir: Path
    labels_dir: Path


class DatasetValidator:
    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir

    def validate(self) -> DatasetValidationResult:
        errors: list[str] = []
        dataset_yaml = self._dataset_dir / "dataset.yaml"
        classes: dict[int, str] = {}
        split_paths: dict[str, ResolvedSplitPaths] = {}

        try:
            yaml_data = parse_dataset_yaml(dataset_yaml)
            classes = self._validate_yaml(yaml_data, errors)
            split_paths = self._resolve_split_paths(dataset_yaml, yaml_data, errors)
        except ValueError as exc:
            errors.append(str(exc))

        train = self._validate_split(split_paths.get("train"), errors)
        val = self._validate_split(split_paths.get("val"), errors)

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

    def _resolve_split_paths(
        self,
        dataset_yaml: Path,
        yaml_data: dict[str, object],
        errors: list[str],
    ) -> dict[str, ResolvedSplitPaths]:
        dataset_root = _resolve_dataset_root(dataset_yaml, yaml_data.get("path", "."))
        split_paths: dict[str, ResolvedSplitPaths] = {}
        for split_name in ["train", "val"]:
            split_value = yaml_data.get(split_name)
            if not split_value:
                continue
            images_dir = _resolve_dataset_path(dataset_root, split_value)
            try:
                labels_dir = yolo_labels_dir_for_images_dir(images_dir)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            split_paths[split_name] = ResolvedSplitPaths(
                name=split_name,
                images_dir=images_dir,
                labels_dir=labels_dir,
            )
        return split_paths

    def _validate_split(self, split_paths: ResolvedSplitPaths | None, errors: list[str]) -> SplitSummary:
        if split_paths is None:
            return SplitSummary(images=0, annotations=0, images_without_annotations=0)

        images_dir = split_paths.images_dir
        labels_dir = split_paths.labels_dir
        if not images_dir.exists():
            errors.append(f"Missing image directory: {images_dir}")
        if not labels_dir.exists():
            errors.append(f"Missing label directory: {labels_dir}")

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
            image_size = _read_image_size(image_path, errors)
            if image_size is None:
                continue
            try:
                boxes = parse_yolo_label_file(
                    label_path,
                    image_width=image_size[0],
                    image_height=image_size[1],
                )
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


def _resolve_dataset_root(dataset_yaml: Path, raw_root: object) -> Path:
    root = Path(str(raw_root))
    if root.is_absolute():
        return root.resolve()
    return (dataset_yaml.parent / root).resolve()


def _resolve_dataset_path(dataset_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    return (dataset_root / path).resolve()


def _read_image_size(image_path: Path, errors: list[str]) -> tuple[int, int] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        errors.append(f"Could not read image dimensions: {image_path}")
        return None
    height, width = image.shape[:2]
    return width, height
