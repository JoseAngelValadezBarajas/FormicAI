from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from formicai.dataset.yolo import (
    YoloBox,
    collect_yolo_label_errors,
    find_images,
    format_yolo_box,
    validate_yolo_box,
)


PUBLIC_ANTS_SEQUENCES = {
    "Seq0001": Path("Ant_dataset/IndoorDataset/Seq0001Object10Image94"),
    "Seq0006": Path("Ant_dataset/OutdoorDataset/Seq0006Object21Image64"),
}


@dataclass(frozen=True)
class AntsGroundTruthRow:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    flag: int

    def original_bbox(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class AntsSequenceInfo:
    name: str
    image_directory: str
    frame_rate: int
    sequence_length: int
    image_width: int
    image_height: int
    image_extension: str


@dataclass(frozen=True)
class AntsSequenceConversionResult:
    sequence: str
    raw_directory: Path
    prepared_directory: Path
    image_count: int
    label_file_count: int
    annotation_count: int
    unique_track_ids: int
    frame_min: int | None
    frame_max: int | None
    flag_values: tuple[int, ...]
    image_width: int
    image_height: int
    frame_rate: int
    images_without_gt: tuple[str, ...]
    gt_frames_without_image: tuple[int, ...]
    malformed_annotations: int
    invalid_rows: int
    invalid_class_ids: int
    out_of_bounds_boxes: int
    out_of_range_normalized_coordinates: int
    zero_or_negative_boxes: int
    duplicate_frame_track_ids: tuple[str, ...]
    seqinfo_name_mismatch: str | None
    tracking_metadata_path: Path
    dataset_yaml_path: Path
    ground_truth_montage_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "rawDirectory": str(self.raw_directory),
            "preparedDirectory": str(self.prepared_directory),
            "images": self.image_count,
            "labels": self.label_file_count,
            "annotations": self.annotation_count,
            "uniqueTrackIds": self.unique_track_ids,
            "frameRange": {
                "min": self.frame_min,
                "max": self.frame_max,
            },
            "flags": list(self.flag_values),
            "resolution": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "fps": self.frame_rate,
            "imagesWithoutGT": list(self.images_without_gt),
            "gtFramesWithoutImage": list(self.gt_frames_without_image),
            "malformedAnnotations": self.malformed_annotations,
            "invalidRows": self.invalid_rows,
            "invalidClassIds": self.invalid_class_ids,
            "outOfBoundsBoxes": self.out_of_bounds_boxes,
            "outOfRangeNormalizedCoordinates": self.out_of_range_normalized_coordinates,
            "zeroOrNegativeBoxes": self.zero_or_negative_boxes,
            "duplicateFrameTrackIds": list(self.duplicate_frame_track_ids),
            "seqinfoNameMismatch": self.seqinfo_name_mismatch,
            "trackingMetadata": str(self.tracking_metadata_path),
            "datasetYaml": str(self.dataset_yaml_path),
            "groundTruthMontage": str(self.ground_truth_montage_path),
        }


@dataclass(frozen=True)
class AntsMendeleyPreparationResult:
    dataset_root: Path
    conversion_report_path: Path
    sequences: tuple[AntsSequenceConversionResult, ...]

    def to_text(self) -> str:
        lines = [
            f"ANTS Mendeley root: {self.dataset_root}",
            f"conversion report: {self.conversion_report_path}",
            "",
            "Converted sequences:",
        ]
        for sequence in self.sequences:
            lines.append(
                f"- {sequence.sequence}: images={sequence.image_count}, labels={sequence.label_file_count}, "
                f"annotations={sequence.annotation_count}, uniqueTrackIds={sequence.unique_track_ids}, "
                f"duplicates={len(sequence.duplicate_frame_track_ids)}"
            )
        return "\n".join(lines)


def parse_ants_ground_truth_line(
    line: str,
    path: Path | None = None,
    line_number: int | None = None,
) -> AntsGroundTruthRow:
    parts = [part.strip() for part in line.strip().split(",")]
    location = _format_location(path, line_number)
    if len(parts) != 7:
        raise ValueError(f"{location}must contain exactly 7 comma-separated values; found {len(parts)}.")

    frame = _parse_integer(parts[0], "frame", location)
    track_id = _parse_integer(parts[1], "track_id", location)
    x = _parse_float(parts[2], "x", location)
    y = _parse_float(parts[3], "y", location)
    width = _parse_float(parts[4], "width", location)
    height = _parse_float(parts[5], "height", location)
    flag = _parse_integer(parts[6], "flag", location)

    if frame < 1:
        raise ValueError(f"{location}frame must be greater than or equal to 1.")
    if width <= 0.0:
        raise ValueError(f"{location}width must be greater than 0.")
    if height <= 0.0:
        raise ValueError(f"{location}height must be greater than 0.")

    return AntsGroundTruthRow(
        frame=frame,
        track_id=track_id,
        x=x,
        y=y,
        width=width,
        height=height,
        flag=flag,
    )


def parse_ants_ground_truth_file(path: Path) -> list[AntsGroundTruthRow]:
    rows: list[AntsGroundTruthRow] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        rows.append(parse_ants_ground_truth_line(line, path, line_number))
    return rows


def ants_row_to_yolo_box(
    row: AntsGroundTruthRow,
    image_width: int,
    image_height: int,
) -> YoloBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be greater than 0.")
    if row.x < 0 or row.y < 0 or row.x + row.width > image_width or row.y + row.height > image_height:
        raise ValueError(
            "ANTS ground-truth box extends outside image bounds: "
            f"frame={row.frame}, track_id={row.track_id}, bbox={row.original_bbox()}, "
            f"image={image_width}x{image_height}."
        )

    box = YoloBox(
        class_id=0,
        x_center=(row.x + row.width / 2) / image_width,
        y_center=(row.y + row.height / 2) / image_height,
        width=row.width / image_width,
        height=row.height / image_height,
    )
    validate_yolo_box(box)
    return box


def frame_to_image_name(frame: int, image_extension: str = ".jpg") -> str:
    if frame < 1:
        raise ValueError("frame must be greater than or equal to 1.")
    if not image_extension.startswith("."):
        raise ValueError("image_extension must start with '.'.")
    return f"{frame:06d}{image_extension}"


def collect_duplicate_track_id_errors(rows: list[AntsGroundTruthRow]) -> list[str]:
    counts = Counter((row.frame, row.track_id) for row in rows)
    return [
        f"frame {frame} track_id {track_id} appears {count} times"
        for (frame, track_id), count in sorted(counts.items())
        if count > 1
    ]


def parse_ants_seqinfo(path: Path) -> AntsSequenceInfo:
    parser = ConfigParser()
    parser.read(path, encoding="utf-8")
    if not parser.has_section("Sequence"):
        raise ValueError(f"Missing [Sequence] section in {path}")
    section = parser["Sequence"]
    return AntsSequenceInfo(
        name=section.get("name", fallback=""),
        image_directory=section.get("imDir", fallback="img"),
        frame_rate=section.getint("frameRate"),
        sequence_length=section.getint("seqLength"),
        image_width=section.getint("imWidth"),
        image_height=section.getint("imHeight"),
        image_extension=section.get("imExt", fallback=".jpg"),
    )


class AntsMendeleyPreparer:
    def __init__(
        self,
        dataset_root: Path = Path("datasets/public/ants_mendeley"),
        analysis_dir: Path = Path("artifacts/analysis/public_ants"),
    ) -> None:
        self._dataset_root = dataset_root
        self._raw_root = dataset_root / "raw"
        self._prepared_root = dataset_root / "prepared"
        self._metadata_root = dataset_root / "metadata"
        self._analysis_dir = analysis_dir

    def prepare(self, sequence_names: tuple[str, ...] = ("Seq0001", "Seq0006")) -> AntsMendeleyPreparationResult:
        results = tuple(self._prepare_sequence(sequence_name) for sequence_name in sequence_names)
        report_path = self._metadata_root / "conversion_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "name": "ANTS--ant detection and tracking",
                        "doi": "10.17632/9ws98g4npw.4",
                        "license": "CC0 1.0",
                    },
                    "phase": "ground_truth_conversion_only",
                    "noTraining": True,
                    "noModelEvaluation": True,
                    "sequences": {
                        result.sequence: result.to_dict()
                        for result in results
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return AntsMendeleyPreparationResult(
            dataset_root=self._dataset_root,
            conversion_report_path=report_path,
            sequences=results,
        )

    def _prepare_sequence(self, sequence_name: str) -> AntsSequenceConversionResult:
        if sequence_name not in PUBLIC_ANTS_SEQUENCES:
            raise ValueError(f"Unsupported ANTS sequence: {sequence_name}")

        raw_dir = self._raw_root / PUBLIC_ANTS_SEQUENCES[sequence_name]
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw sequence directory does not exist: {raw_dir}")

        seqinfo = parse_ants_seqinfo(raw_dir / "seqinfo.ini")
        gt_path = raw_dir / "gt" / "gt.txt"
        rows = parse_ants_ground_truth_file(gt_path)
        duplicate_errors = collect_duplicate_track_id_errors(rows)
        if duplicate_errors:
            details = "\n".join(f"- {error}" for error in duplicate_errors)
            raise ValueError(f"Duplicate (frame, track_id) rows found in {gt_path}\n{details}")

        images_dir = raw_dir / seqinfo.image_directory
        image_paths = _collect_sequence_images(images_dir, seqinfo)
        image_frames = {frame_index for frame_index, _ in image_paths}
        rows_by_frame: dict[int, list[AntsGroundTruthRow]] = defaultdict(list)
        for row in rows:
            rows_by_frame[row.frame].append(row)

        gt_frames = set(rows_by_frame)
        images_without_gt = tuple(
            frame_to_image_name(frame, seqinfo.image_extension)
            for frame in sorted(image_frames - gt_frames)
        )
        gt_frames_without_image = tuple(sorted(gt_frames - image_frames))
        if gt_frames_without_image:
            raise ValueError(f"GT frames without matching images in {sequence_name}: {gt_frames_without_image}")

        prepared_dir = self._prepared_root / sequence_name
        prepared_images_dir = prepared_dir / "images"
        prepared_labels_dir = prepared_dir / "labels"
        _clear_files(prepared_images_dir, {seqinfo.image_extension.lower()})
        _clear_files(prepared_labels_dir, {".txt"})

        tracking_path = self._metadata_root / f"tracking_{sequence_name.lower()}.jsonl"
        tracking_path.parent.mkdir(parents=True, exist_ok=True)
        out_of_bounds_boxes = 0
        with tracking_path.open("w", encoding="utf-8") as tracking_file:
            for frame, image_path in image_paths:
                target_image = prepared_images_dir / image_path.name
                shutil.copy2(image_path, target_image)

                label_lines = []
                for row in rows_by_frame.get(frame, []):
                    try:
                        yolo_box = ants_row_to_yolo_box(row, seqinfo.image_width, seqinfo.image_height)
                    except ValueError:
                        out_of_bounds_boxes += 1
                        raise
                    label_lines.append(format_yolo_box(yolo_box))
                    tracking_file.write(
                        json.dumps(
                            {
                                "sequence": sequence_name,
                                "frame": row.frame,
                                "image": image_path.name,
                                "trackId": row.track_id,
                                "bbox": row.original_bbox(),
                                "flag": row.flag,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                label_path = prepared_labels_dir / f"{image_path.stem}.txt"
                label_path.write_text(("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8")

        yolo_validation = _validate_yolo_labels(prepared_images_dir, prepared_labels_dir)
        yaml_path = self._write_sequence_yaml(sequence_name)
        montage_path = self._write_ground_truth_montage(
            sequence_name=sequence_name,
            images_dir=prepared_images_dir,
            rows_by_frame=rows_by_frame,
            seqinfo=seqinfo,
        )

        sequence_info_name = seqinfo.name
        seqinfo_name_mismatch = None
        if sequence_info_name and sequence_info_name != raw_dir.name:
            seqinfo_name_mismatch = f"folder={raw_dir.name}; seqinfo.ini name={sequence_info_name}"

        return AntsSequenceConversionResult(
            sequence=sequence_name,
            raw_directory=raw_dir,
            prepared_directory=prepared_dir,
            image_count=len(image_paths),
            label_file_count=len(sorted(prepared_labels_dir.glob("*.txt"))),
            annotation_count=len(rows),
            unique_track_ids=len({row.track_id for row in rows}),
            frame_min=min((row.frame for row in rows), default=None),
            frame_max=max((row.frame for row in rows), default=None),
            flag_values=tuple(sorted({row.flag for row in rows})),
            image_width=seqinfo.image_width,
            image_height=seqinfo.image_height,
            frame_rate=seqinfo.frame_rate,
            images_without_gt=images_without_gt,
            gt_frames_without_image=gt_frames_without_image,
            malformed_annotations=0,
            invalid_rows=0,
            invalid_class_ids=yolo_validation["invalid_class_ids"],
            out_of_bounds_boxes=out_of_bounds_boxes,
            out_of_range_normalized_coordinates=yolo_validation["out_of_range"],
            zero_or_negative_boxes=yolo_validation["zero_or_negative"],
            duplicate_frame_track_ids=tuple(duplicate_errors),
            seqinfo_name_mismatch=seqinfo_name_mismatch,
            tracking_metadata_path=tracking_path,
            dataset_yaml_path=yaml_path,
            ground_truth_montage_path=montage_path,
        )

    def _write_sequence_yaml(self, sequence_name: str) -> Path:
        yaml_path = self._metadata_root / f"{sequence_name.lower()}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_path = (self._prepared_root / sequence_name).resolve().as_posix()
        yaml_path.write_text(
            f"path: {prepared_path}\n"
            "train: images\n"
            "val: images\n"
            "test: images\n"
            "names:\n"
            "  0: ant\n",
            encoding="utf-8",
        )
        return yaml_path

    def _write_ground_truth_montage(
        self,
        sequence_name: str,
        images_dir: Path,
        rows_by_frame: dict[int, list[AntsGroundTruthRow]],
        seqinfo: AntsSequenceInfo,
    ) -> Path:
        selected_frames = _select_temporal_frames(1, seqinfo.sequence_length, count=10)
        tiles = []
        tile_width = 480
        tile_height = max(1, round(tile_width * seqinfo.image_height / seqinfo.image_width))
        for frame in selected_frames:
            image_path = images_dir / frame_to_image_name(frame, seqinfo.image_extension)
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image for montage: {image_path}")
            tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            scale_x = tile_width / seqinfo.image_width
            scale_y = tile_height / seqinfo.image_height
            for row in rows_by_frame.get(frame, []):
                x1 = int(round(row.x * scale_x))
                y1 = int(round(row.y * scale_y))
                x2 = int(round((row.x + row.width) * scale_x))
                y2 = int(round((row.y + row.height) * scale_y))
                cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    tile,
                    f"ant/{row.track_id}",
                    (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                tile,
                f"{sequence_name} frame {frame:06d}",
                (10, tile_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        montage = _make_montage(tiles, columns=5)
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        montage_path = self._analysis_dir / f"{sequence_name}_ground_truth_montage.jpg"
        if not cv2.imwrite(str(montage_path), montage):
            raise RuntimeError(f"Could not write montage: {montage_path}")
        return montage_path


def _collect_sequence_images(images_dir: Path, seqinfo: AntsSequenceInfo) -> list[tuple[int, Path]]:
    images = []
    missing = []
    for frame in range(1, seqinfo.sequence_length + 1):
        image_path = images_dir / frame_to_image_name(frame, seqinfo.image_extension)
        if not image_path.exists():
            missing.append(image_path.name)
        else:
            images.append((frame, image_path))
    if missing:
        raise ValueError(f"Missing expected frame image(s): {', '.join(missing[:20])}")

    extra_images = [
        image.name
        for image in find_images(images_dir)
        if image.name not in {path.name for _, path in images}
    ]
    if extra_images:
        raise ValueError(f"Unexpected image(s) in sequence directory: {', '.join(extra_images[:20])}")
    return images


def _validate_yolo_labels(images_dir: Path, labels_dir: Path) -> dict[str, int]:
    images = find_images(images_dir)
    label_paths = sorted(labels_dir.glob("*.txt"))
    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in label_paths}
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        raise ValueError(f"Image/label mismatch. missing={missing_labels[:10]}, extra={extra_labels[:10]}")

    invalid_class_ids = 0
    out_of_range = 0
    zero_or_negative = 0
    for label_path in label_paths:
        boxes, errors = collect_yolo_label_errors(label_path)
        for error in errors:
            if "class id" in error:
                invalid_class_ids += 1
            if "normalized between 0 and 1" in error or "outside normalized image bounds" in error:
                out_of_range += 1
            if "width and height must be greater than 0" in error:
                zero_or_negative += 1
        if errors:
            raise ValueError(f"Invalid YOLO label generated: {errors[0]}")
        for box in boxes:
            if box.class_id != 0:
                invalid_class_ids += 1
            if box.width <= 0.0 or box.height <= 0.0:
                zero_or_negative += 1
    return {
        "invalid_class_ids": invalid_class_ids,
        "out_of_range": out_of_range,
        "zero_or_negative": zero_or_negative,
    }


def _select_temporal_frames(frame_min: int, frame_max: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be greater than 0.")
    if frame_min > frame_max:
        raise ValueError("frame_min must be less than or equal to frame_max.")
    available = frame_max - frame_min + 1
    selected_count = min(count, available)
    if selected_count == 1:
        return (frame_min,)
    return tuple(
        frame_min + int(round(index * (available - 1) / (selected_count - 1)))
        for index in range(selected_count)
    )


def _make_montage(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    if not tiles:
        raise ValueError("tiles must not be empty.")
    rows = math.ceil(len(tiles) / columns)
    tile_height, tile_width = tiles[0].shape[:2]
    montage = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        y = row * tile_height
        x = col * tile_width
        montage[y : y + tile_height, x : x + tile_width] = tile
    return montage


def _clear_files(directory: Path, suffixes: set[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()


def _parse_float(value: str, field_name: str, location: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{location}{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{location}{field_name} must be finite.")
    return parsed


def _parse_integer(value: str, field_name: str, location: str) -> int:
    parsed = _parse_float(value, field_name, location)
    if not parsed.is_integer():
        raise ValueError(f"{location}{field_name} must be an integer.")
    return int(parsed)


def _format_location(path: Path | None, line_number: int | None) -> str:
    if path is None or line_number is None:
        return ""
    return f"{path}:{line_number} "
