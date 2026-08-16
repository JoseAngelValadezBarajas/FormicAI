from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from formicai.dataset.yolo import parse_dataset_yaml, yolo_labels_dir_for_images_dir


@dataclass(frozen=True)
class ResolvedSplitPaths:
    name: str
    images_dir: Path
    labels_dir: Path


@dataclass(frozen=True)
class ResolvedDatasetPaths:
    dataset_yaml: Path
    dataset_root: Path
    yaml_data: dict[str, object]
    splits: dict[str, ResolvedSplitPaths]


def resolve_dataset_split_paths(
    dataset_yaml: Path,
    *,
    required_splits: tuple[str, ...] = ("train", "val"),
) -> ResolvedDatasetPaths:
    yaml_data = parse_dataset_yaml(dataset_yaml)
    dataset_root = _resolve_dataset_root(dataset_yaml, yaml_data.get("path", "."))
    split_paths: dict[str, ResolvedSplitPaths] = {}

    for split_name in required_splits:
        split_value = yaml_data.get(split_name)
        if not split_value:
            raise ValueError(f"dataset.yaml must define '{split_name}'.")
        images_dir = _resolve_dataset_path(dataset_root, split_value)
        labels_dir = yolo_labels_dir_for_images_dir(images_dir)
        split_paths[split_name] = ResolvedSplitPaths(
            name=split_name,
            images_dir=images_dir,
            labels_dir=labels_dir,
        )

    return ResolvedDatasetPaths(
        dataset_yaml=dataset_yaml,
        dataset_root=dataset_root,
        yaml_data=yaml_data,
        splits=split_paths,
    )


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image dimensions: {image_path}")
    height, width = image.shape[:2]
    return width, height


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
