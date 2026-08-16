from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from statistics import median

from formicai.dataset.paths import read_image_size, resolve_dataset_split_paths
from formicai.dataset.yolo import find_images, matching_label_path, parse_yolo_label_file


@dataclass(frozen=True)
class SplitStatistics:
    images: int
    annotations: int
    images_without_annotations: int


@dataclass(frozen=True)
class DatasetStatistics:
    images: int
    train_images: int
    val_images: int
    annotations: int
    train_annotations: int
    val_annotations: int
    average_annotations_per_image: float
    min_annotations_per_image: int
    max_annotations_per_image: int
    images_without_annotations: int
    median_box_width_ratio: float
    median_box_height_ratio: float
    median_box_area_ratio: float
    p25_box_area_ratio: float
    p75_box_area_ratio: float

    def to_text(self) -> str:
        return "\n".join(
            [
                f"Total images: {self.images}",
                f"Train images: {self.train_images}",
                f"Val images: {self.val_images}",
                f"Total annotations: {self.annotations}",
                f"Train annotations: {self.train_annotations}",
                f"Val annotations: {self.val_annotations}",
                f"Average ants per image: {self.average_annotations_per_image:.3f}",
                f"Min ants per image: {self.min_annotations_per_image}",
                f"Max ants per image: {self.max_annotations_per_image}",
                f"Images without annotations: {self.images_without_annotations}",
                f"Median box width ratio: {self.median_box_width_ratio:.6f}",
                f"Median box height ratio: {self.median_box_height_ratio:.6f}",
                f"Median box area ratio: {self.median_box_area_ratio:.6f}",
                f"P25 box area ratio: {self.p25_box_area_ratio:.6f}",
                f"P75 box area ratio: {self.p75_box_area_ratio:.6f}",
            ]
        )


class DatasetStatisticsCalculator:
    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir

    def calculate(self) -> DatasetStatistics:
        annotation_counts: list[int] = []
        widths: list[float] = []
        heights: list[float] = []
        areas: list[float] = []
        split_statistics: dict[str, SplitStatistics] = {}
        resolved_paths = resolve_dataset_split_paths(self._dataset_dir / "dataset.yaml")

        for split in ["train", "val"]:
            split_annotation_counts: list[int] = []
            split_paths = resolved_paths.splits[split]
            images_dir = split_paths.images_dir
            labels_dir = split_paths.labels_dir
            for image_path in find_images(images_dir):
                label_path = matching_label_path(image_path, labels_dir)
                if not label_path.exists():
                    raise ValueError(f"Missing label for image: {image_path}; expected label: {label_path}")
                image_width, image_height = read_image_size(image_path)
                boxes = parse_yolo_label_file(
                    label_path,
                    image_width=image_width,
                    image_height=image_height,
                )
                annotation_counts.append(len(boxes))
                split_annotation_counts.append(len(boxes))
                widths.extend(box.width for box in boxes)
                heights.extend(box.height for box in boxes)
                areas.extend(box.area_ratio for box in boxes)
            split_statistics[split] = SplitStatistics(
                images=len(split_annotation_counts),
                annotations=sum(split_annotation_counts),
                images_without_annotations=sum(1 for count in split_annotation_counts if count == 0),
            )

        image_count = len(annotation_counts)
        annotation_count = sum(annotation_counts)
        train = split_statistics["train"]
        val = split_statistics["val"]
        return DatasetStatistics(
            images=image_count,
            train_images=train.images,
            val_images=val.images,
            annotations=annotation_count,
            train_annotations=train.annotations,
            val_annotations=val.annotations,
            average_annotations_per_image=annotation_count / image_count if image_count else 0.0,
            min_annotations_per_image=min(annotation_counts, default=0),
            max_annotations_per_image=max(annotation_counts, default=0),
            images_without_annotations=sum(1 for count in annotation_counts if count == 0),
            median_box_width_ratio=median(widths) if widths else 0.0,
            median_box_height_ratio=median(heights) if heights else 0.0,
            median_box_area_ratio=median(areas) if areas else 0.0,
            p25_box_area_ratio=_percentile(areas, 0.25),
            p75_box_area_ratio=_percentile(areas, 0.75),
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1.")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile
    lower_index = floor(position)
    upper_index = ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight
