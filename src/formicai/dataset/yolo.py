from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_BOUNDS_TOLERANCE = 1e-9
YOLO_POLYGON_AREA_TOLERANCE = 1e-12


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area_ratio(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class ParsedYoloAnnotation:
    box: YoloBox
    annotation_format: Literal["detection", "segmentation"]


def parse_dataset_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"dataset.yaml does not exist: {path}")

    parsed: dict[str, object] = {}
    names: dict[int, str] = {}
    in_names = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if line.startswith("names:"):
            in_names = True
            value = line.split(":", 1)[1].strip()
            if value:
                parsed["names"] = value
            continue

        if in_names and line.startswith("  "):
            key, value = _split_key_value(line.strip(), path)
            names[int(key)] = _clean_yaml_scalar(value)
            continue

        in_names = False
        key, value = _split_key_value(line, path)
        parsed[key] = _clean_yaml_scalar(value)

    if names:
        parsed["names"] = names
    return parsed


def parse_yolo_label_file(path: Path) -> list[YoloBox]:
    boxes, errors = collect_yolo_label_errors(path)
    if errors:
        raise ValueError(errors[0])
    return boxes


def collect_yolo_label_errors(path: Path) -> tuple[list[YoloBox], list[str]]:
    boxes: list[YoloBox] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            boxes.append(_parse_yolo_label_line(line, path, line_number))
        except ValueError as exc:
            errors.append(str(exc))
    return boxes, errors


def _parse_yolo_label_line(line: str, path: Path, line_number: int) -> YoloBox:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{path}:{line_number} must contain 5 values; found {len(parts)}.")
    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} class id must be an integer.") from exc
    try:
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} contains non-numeric coordinate values.") from exc

    box = YoloBox(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )
    validate_yolo_box(box, path, line_number)
    return box


def parse_yolo_detection_or_segmentation_line(
    line: str,
    path: Path | None = None,
    line_number: int | None = None,
) -> ParsedYoloAnnotation:
    location_path = path or Path("<label>")
    location_line = line_number or 1
    parts = line.split()
    if len(parts) == 5:
        return ParsedYoloAnnotation(
            box=_parse_yolo_label_line(line, location_path, location_line),
            annotation_format="detection",
        )

    box = _parse_yolo_segmentation_line(line, location_path, location_line)
    return ParsedYoloAnnotation(box=box, annotation_format="segmentation")


def collect_yolo_detection_or_segmentation_errors(path: Path) -> tuple[list[ParsedYoloAnnotation], list[str]]:
    annotations: list[ParsedYoloAnnotation] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            annotations.append(parse_yolo_detection_or_segmentation_line(line, path, line_number))
        except ValueError as exc:
            errors.append(str(exc))
    return annotations, errors


def format_yolo_box(box: YoloBox) -> str:
    return (
        f"{box.class_id} "
        f"{box.x_center:.12g} "
        f"{box.y_center:.12g} "
        f"{box.width:.12g} "
        f"{box.height:.12g}"
    )


def segmentation_points_to_box(
    class_id: int,
    points: list[tuple[float, float]],
    path: Path | None = None,
    line_number: int | None = None,
) -> YoloBox:
    location = _format_location(path, line_number)
    if class_id != 0:
        raise ValueError(f"{location}class id must be 0 for 'ant'.")
    if len(points) < 3:
        raise ValueError(f"{location}segmentation polygon must contain at least 3 points.")

    for point_index, (x, y) in enumerate(points, start=1):
        if x < -YOLO_BOUNDS_TOLERANCE or x > 1.0 + YOLO_BOUNDS_TOLERANCE:
            raise ValueError(f"{location}segmentation point {point_index} x must be normalized between 0 and 1.")
        if y < -YOLO_BOUNDS_TOLERANCE or y > 1.0 + YOLO_BOUNDS_TOLERANCE:
            raise ValueError(f"{location}segmentation point {point_index} y must be normalized between 0 and 1.")

    polygon_area = abs(_shoelace_area(points))
    if polygon_area <= YOLO_POLYGON_AREA_TOLERANCE:
        raise ValueError(f"{location}segmentation polygon is degenerate.")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    box = YoloBox(
        class_id=class_id,
        x_center=(x_min + x_max) / 2,
        y_center=(y_min + y_max) / 2,
        width=x_max - x_min,
        height=y_max - y_min,
    )
    validate_yolo_box(box, path, line_number)
    return box


def _parse_yolo_segmentation_line(line: str, path: Path, line_number: int) -> YoloBox:
    parts = line.split()
    try:
        class_id = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} class id must be an integer.") from exc
    if class_id != 0:
        raise ValueError(f"{path}:{line_number} class id must be 0 for 'ant'.")

    coordinate_values = parts[1:]
    if len(coordinate_values) % 2 != 0:
        raise ValueError(f"{path}:{line_number} segmentation coordinate count must be even.")
    if len(coordinate_values) < 6:
        raise ValueError(f"{path}:{line_number} segmentation polygon must contain at least 3 points.")

    try:
        coordinates = [float(value) for value in coordinate_values]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} contains non-numeric coordinate values.") from exc

    points = list(zip(coordinates[0::2], coordinates[1::2]))
    return segmentation_points_to_box(class_id, points, path, line_number)


def validate_yolo_box(box: YoloBox, path: Path | None = None, line_number: int | None = None) -> None:
    location = _format_location(path, line_number)

    if box.class_id != 0:
        raise ValueError(f"{location}class id must be 0 for 'ant'.")
    for name, value in [
        ("x_center", box.x_center),
        ("y_center", box.y_center),
        ("width", box.width),
        ("height", box.height),
    ]:
        if value < -YOLO_BOUNDS_TOLERANCE or value > 1.0 + YOLO_BOUNDS_TOLERANCE:
            raise ValueError(f"{location}{name} must be normalized between 0 and 1.")
    if box.width <= 0.0 or box.height <= 0.0:
        raise ValueError(f"{location}width and height must be greater than 0.")

    x_min = box.x_center - box.width / 2
    x_max = box.x_center + box.width / 2
    y_min = box.y_center - box.height / 2
    y_max = box.y_center + box.height / 2
    if (
        x_min < -YOLO_BOUNDS_TOLERANCE
        or x_max > 1.0 + YOLO_BOUNDS_TOLERANCE
        or y_min < -YOLO_BOUNDS_TOLERANCE
        or y_max > 1.0 + YOLO_BOUNDS_TOLERANCE
    ):
        raise ValueError(f"{location}bounding box extends outside normalized image bounds.")


def find_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def matching_label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _format_location(path: Path | None, line_number: int | None) -> str:
    if path is None or line_number is None:
        return ""
    return f"{path}:{line_number} "


def _shoelace_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x_current, y_current) in enumerate(points):
        x_next, y_next = points[(index + 1) % len(points)]
        area += x_current * y_next - x_next * y_current
    return area / 2


def _split_key_value(line: str, path: Path) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Invalid YAML line in {path}: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _clean_yaml_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")
