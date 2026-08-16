from __future__ import annotations

import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from formicai.dataset.yolo import YoloBox, find_images, matching_label_path, parse_yolo_label_file


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_root: Path = Path("datasets/public/ants_mendeley")
    model_path: Path = Path("artifacts/models/ants_v1_baseline/weights/best.pt")
    output_path: Path = Path("artifacts/analysis/public_ants/external_benchmark_metrics.json")
    predictions_dir: Path = Path("artifacts/analysis/public_ants/predictions")
    video_dir: Path = Path("output/public_ants")
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640
    device: str | int | None = "0"
    end2end: bool | None = False
    sequences: tuple[str, ...] = ("Seq0001", "Seq0006")


@dataclass(frozen=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains_center_of(self, other: "PixelBox") -> bool:
        cx, cy = other.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def to_xyxy_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class PredictionBox:
    box: PixelBox
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class FrameEvaluation:
    image: str
    gt_count: int
    prediction_count: int
    center_matched: int
    center_missed: int
    center_unmatched_predictions: int
    iou50_matched: int
    iou50_missed: int
    iou50_unmatched_predictions: int


@dataclass(frozen=True)
class PublicAntsBenchmarkResult:
    output_path: Path
    sequences: dict[str, dict[str, object]]

    def to_text(self) -> str:
        lines = [f"metrics: {self.output_path}", "", "Public ANTS benchmark:"]
        for name, data in self.sequences.items():
            standard = data["standardMetrics"]
            center = data["centerBasedMetrics"]
            lines.append(
                f"- {name}: mAP50={standard['mAP50']:.4f}, "
                f"mAP50-95={standard['mAP50_95']:.4f}, "
                f"center_recall={center['centerBasedRecall']:.4f}, "
                f"predictions={standard['predictions']}"
            )
        return "\n".join(lines)


class PublicAntsBenchmarkEvaluator:
    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    def evaluate(self) -> PublicAntsBenchmarkResult:
        if not self._config.model_path.exists():
            raise FileNotFoundError(f"Model does not exist: {self._config.model_path}")

        try:
            import torch
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics and PyTorch are required for benchmark evaluation.") from exc

        model = YOLO(str(self._config.model_path))
        sequences: dict[str, dict[str, object]] = {}
        for sequence in self._config.sequences:
            sequences[sequence] = self._evaluate_sequence(model, sequence)

        payload = {
            "model": str(self._config.model_path),
            "configuration": {
                "end2end": self._config.end2end,
                "confidence": self._config.confidence,
                "iou": self._config.iou,
                "imageSize": self._config.image_size,
                "device": self._config.device,
                "roi": None,
            },
            "dataset": {
                "name": "ANTS--ant detection and tracking",
                "doi": "10.17632/9ws98g4npw.4",
                "license": "CC0 1.0",
                "root": str(self._config.dataset_root),
                "benchmarkUse": "external evaluation only; not training data",
            },
            "environment": _environment_info(torch, ultralytics),
            "sequences": sequences,
            "comparison": _internal_ant2_comparison(sequences),
        }
        self._config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return PublicAntsBenchmarkResult(output_path=self._config.output_path, sequences=sequences)

    def _evaluate_sequence(self, model: Any, sequence: str) -> dict[str, object]:
        sequence_dir = self._config.dataset_root / "prepared" / sequence
        images_dir = sequence_dir / "images"
        labels_dir = sequence_dir / "labels"
        dataset_yaml = self._config.dataset_root / "metadata" / f"{sequence.lower()}.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")

        gt_by_image = _load_ground_truth(images_dir, labels_dir)
        image_paths = sorted(gt_by_image)
        duplicate_yolo_labels = _collect_duplicate_yolo_label_rows(labels_dir)
        duplicate_yolo_extra_rows = sum(int(item["extraRows"]) for item in duplicate_yolo_labels)
        standard_metrics = self._run_ultralytics_validation(model, dataset_yaml, sequence)
        predictions_by_image = self._run_predictions(model, images_dir)

        frame_evaluations = []
        total_gt = 0
        total_predictions = 0
        total_center_matches = 0
        total_iou50_matches = 0
        for image_path in image_paths:
            gt_boxes = gt_by_image[image_path]
            predictions = predictions_by_image.get(image_path.name, [])
            center_matches = match_center_based(gt_boxes, predictions)
            iou50_matches = match_iou_threshold(gt_boxes, predictions, threshold=0.50)
            total_gt += len(gt_boxes)
            total_predictions += len(predictions)
            total_center_matches += len(center_matches)
            total_iou50_matches += len(iou50_matches)
            frame_evaluations.append(
                FrameEvaluation(
                    image=image_path.name,
                    gt_count=len(gt_boxes),
                    prediction_count=len(predictions),
                    center_matched=len(center_matches),
                    center_missed=len(gt_boxes) - len(center_matches),
                    center_unmatched_predictions=len(predictions) - len(center_matches),
                    iou50_matched=len(iou50_matches),
                    iou50_missed=len(gt_boxes) - len(iou50_matches),
                    iou50_unmatched_predictions=len(predictions) - len(iou50_matches),
                )
            )

        density = density_analysis(frame_evaluations)
        standard_metrics.update(
            {
                "images": len(image_paths),
                "groundTruthAnts": total_gt,
                "ultralyticsGroundTruthAnts": total_gt - duplicate_yolo_extra_rows,
                "ultralyticsDuplicateLabelsRemoved": duplicate_yolo_extra_rows,
                "duplicateYoloLabels": duplicate_yolo_labels,
                "predictions": total_predictions,
                "iou50TruePositives": total_iou50_matches,
                "iou50FalsePositives": total_predictions - total_iou50_matches,
                "iou50FalseNegatives": total_gt - total_iou50_matches,
            }
        )
        center_metrics = _center_metrics(total_gt, total_predictions, total_center_matches)
        selected_examples = _select_visual_examples(frame_evaluations)
        prediction_artifacts = self._write_prediction_examples(
            sequence=sequence,
            image_paths=image_paths,
            gt_by_image=gt_by_image,
            predictions_by_image=predictions_by_image,
            selected_examples=selected_examples,
        )
        video_path = self._write_prediction_video(sequence, image_paths, predictions_by_image)

        return {
            "standardMetrics": standard_metrics,
            "centerBasedMetrics": center_metrics,
            "densityAnalysis": density,
            "frameEvaluation": [frame.__dict__ for frame in frame_evaluations],
            "visualExamples": prediction_artifacts,
            "video": str(video_path),
        }

    def _run_ultralytics_validation(self, model: Any, dataset_yaml: Path, sequence: str) -> dict[str, float]:
        metrics = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=self._config.image_size,
            conf=self._config.confidence,
            iou=self._config.iou,
            device=self._config.device,
            end2end=self._config.end2end,
            verbose=False,
            plots=False,
            save_json=False,
            workers=0,
            project=str(self._config.output_path.parent / "ultralytics_val_runs"),
            name=sequence,
            exist_ok=True,
        )
        return {
            "precision": _metric_value(metrics, "mp", "metrics/precision(B)"),
            "recall": _metric_value(metrics, "mr", "metrics/recall(B)"),
            "mAP50": _metric_value(metrics, "map50", "metrics/mAP50(B)"),
            "mAP50_95": _metric_value(metrics, "map", "metrics/mAP50-95(B)"),
        }

    def _run_predictions(self, model: Any, images_dir: Path) -> dict[str, list[PredictionBox]]:
        predictions: dict[str, list[PredictionBox]] = {}
        results = model.predict(
            source=str(images_dir),
            imgsz=self._config.image_size,
            conf=self._config.confidence,
            iou=self._config.iou,
            device=self._config.device,
            end2end=self._config.end2end,
            stream=True,
            verbose=False,
        )
        for result in results:
            image_name = Path(result.path).name
            image_predictions = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id != 0:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                image_predictions.append(
                    PredictionBox(
                        box=PixelBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(box.conf.item()),
                        class_id=class_id,
                    )
                )
            predictions[image_name] = image_predictions
        return predictions

    def _write_prediction_examples(
        self,
        sequence: str,
        image_paths: list[Path],
        gt_by_image: dict[Path, list[PixelBox]],
        predictions_by_image: dict[str, list[PredictionBox]],
        selected_examples: list[FrameEvaluation],
    ) -> dict[str, object]:
        output_dir = self._config.predictions_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        by_name = {path.name: path for path in image_paths}
        example_paths = []
        tiles = []
        for frame in selected_examples:
            image_path = by_name[frame.image]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            gt = gt_by_image[image_path]
            predictions = predictions_by_image.get(image_path.name, [])
            side_by_side = _draw_gt_vs_predictions(image, sequence, frame.image, gt, predictions)
            output_path = output_dir / f"{sequence}_{Path(frame.image).stem}_gt_vs_formicai.jpg"
            if not cv2.imwrite(str(output_path), side_by_side):
                raise RuntimeError(f"Could not write prediction example: {output_path}")
            example_paths.append(str(output_path))
            tiles.append(_resize_to_width(side_by_side, width=640))

        montage = _make_montage(tiles, columns=2)
        montage_path = output_dir / f"{sequence}_prediction_examples_montage.jpg"
        if not cv2.imwrite(str(montage_path), montage):
            raise RuntimeError(f"Could not write prediction montage: {montage_path}")

        return {
            "selectionMethod": (
                "Representative frames selected from good center recall, false negatives, false positives, "
                "crowded/high-density, low-density, style-mismatch, and temporal coverage candidates."
            ),
            "frames": [frame.__dict__ for frame in selected_examples],
            "files": example_paths,
            "montage": str(montage_path),
        }

    def _write_prediction_video(
        self,
        sequence: str,
        image_paths: list[Path],
        predictions_by_image: dict[str, list[PredictionBox]],
    ) -> Path:
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"Could not read first frame for video: {image_paths[0]}")
        height, width = first.shape[:2]
        fps = 25 if sequence == "Seq0001" else 30
        self._config.video_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._config.video_dir / f"{sequence}_formicai.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {output_path}")

        try:
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Could not read video frame: {image_path}")
                annotated = image.copy()
                for prediction in predictions_by_image.get(image_path.name, []):
                    _draw_prediction(annotated, prediction)
                cv2.putText(
                    annotated,
                    f"{sequence} FormicAI predictions {image_path.stem}",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
        finally:
            writer.release()
        return output_path


def match_center_based(gt_boxes: list[PixelBox], predictions: list[PredictionBox]) -> list[tuple[int, int]]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        pred_center = prediction.box.center
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_box.contains_center_of(prediction.box):
                gt_center = gt_box.center
                distance = math.dist(pred_center, gt_center)
                candidates.append((-prediction.confidence, distance, pred_index, gt_index))

    return _greedy_one_to_one_matches(candidates)


def match_iou_threshold(
    gt_boxes: list[PixelBox],
    predictions: list[PredictionBox],
    threshold: float,
) -> list[tuple[int, int]]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou_value = _iou(gt_box, prediction.box)
            if iou_value >= threshold:
                candidates.append((-iou_value, -prediction.confidence, pred_index, gt_index))
    return _greedy_one_to_one_matches(candidates)


def density_analysis(frames: list[FrameEvaluation]) -> dict[str, dict[str, float | int]]:
    groups = _density_groups(frames)
    result = {}
    for group_name, group_frames in groups.items():
        gt = sum(frame.gt_count for frame in group_frames)
        predictions = sum(frame.prediction_count for frame in group_frames)
        center_matched = sum(frame.center_matched for frame in group_frames)
        iou50_matched = sum(frame.iou50_matched for frame in group_frames)
        result[group_name] = {
            "frames": len(group_frames),
            "minGtAntsPerFrame": min((frame.gt_count for frame in group_frames), default=0),
            "maxGtAntsPerFrame": max((frame.gt_count for frame in group_frames), default=0),
            "gtAnts": gt,
            "predictions": predictions,
            "standardIou50MatchedAnts": iou50_matched,
            "standardIou50MissedAnts": gt - iou50_matched,
            "standardIou50FalsePositives": predictions - iou50_matched,
            "standardIou50Precision": _safe_ratio(iou50_matched, predictions),
            "standardIou50Recall": _safe_ratio(iou50_matched, gt),
            "centerMatchedAnts": center_matched,
            "centerMissedAnts": gt - center_matched,
            "centerUnmatchedPredictions": predictions - center_matched,
            "centerPrecision": _safe_ratio(center_matched, predictions),
            "centerRecall": _safe_ratio(center_matched, gt),
        }
    return result


def _load_ground_truth(images_dir: Path, labels_dir: Path) -> dict[Path, list[PixelBox]]:
    image_paths = find_images(images_dir)
    ground_truth = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image_height, image_width = image.shape[:2]
        label_path = matching_label_path(image_path, labels_dir)
        boxes = parse_yolo_label_file(
            label_path,
            image_width=image_width,
            image_height=image_height,
        )
        ground_truth[image_path] = [_yolo_box_to_pixel_box(box, image_width, image_height) for box in boxes]
    return ground_truth


def _collect_duplicate_yolo_label_rows(labels_dir: Path) -> list[dict[str, object]]:
    duplicates: list[dict[str, object]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = Counter(rows)
        for row, count in sorted(counts.items()):
            if count <= 1:
                continue
            duplicates.append(
                {
                    "image": f"{label_path.stem}.jpg",
                    "label": str(label_path),
                    "yoloRow": row,
                    "count": count,
                    "extraRows": count - 1,
                    "note": "Ultralytics removes exact duplicate label rows during standard validation cache creation.",
                }
            )
    return duplicates


def _yolo_box_to_pixel_box(box: YoloBox, image_width: int, image_height: int) -> PixelBox:
    width = box.width * image_width
    height = box.height * image_height
    x_center = box.x_center * image_width
    y_center = box.y_center * image_height
    return PixelBox(
        x1=x_center - width / 2,
        y1=y_center - height / 2,
        x2=x_center + width / 2,
        y2=y_center + height / 2,
    )


def _greedy_one_to_one_matches(candidates: list[tuple[float, ...]]) -> list[tuple[int, int]]:
    matched_predictions = set()
    matched_ground_truth = set()
    matches = []
    for candidate in sorted(candidates):
        pred_index = int(candidate[-2])
        gt_index = int(candidate[-1])
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index))
    return matches


def _iou(a: PixelBox, b: PixelBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.width) * max(0.0, a.height)
    area_b = max(0.0, b.width) * max(0.0, b.height)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_metrics(total_gt: int, total_predictions: int, total_matches: int) -> dict[str, float | int]:
    return {
        "name": "center-based localization metrics",
        "matchingRule": "Prediction center lies inside GT bounding box; one prediction per GT and one GT per prediction.",
        "gtAnts": total_gt,
        "centerMatchedAnts": total_matches,
        "centerMissedAnts": total_gt - total_matches,
        "predictions": total_predictions,
        "unmatchedPredictions": total_predictions - total_matches,
        "centerBasedPrecision": _safe_ratio(total_matches, total_predictions),
        "centerBasedRecall": _safe_ratio(total_matches, total_gt),
    }


def _density_groups(frames: list[FrameEvaluation]) -> dict[str, list[FrameEvaluation]]:
    ranked = sorted(frames, key=lambda frame: (frame.gt_count, frame.image))
    total = len(ranked)
    low_end = total // 3
    medium_end = (2 * total) // 3
    return {
        "lowDensity": ranked[:low_end],
        "mediumDensity": ranked[low_end:medium_end],
        "highDensity": ranked[medium_end:],
    }


def _select_visual_examples(frames: list[FrameEvaluation], count: int = 10) -> list[FrameEvaluation]:
    selected: list[FrameEvaluation] = []
    selected_names: set[str] = set()

    def add(frame: FrameEvaluation | None) -> None:
        if frame is None or frame.image in selected_names or len(selected) >= count:
            return
        selected.append(frame)
        selected_names.add(frame.image)

    if not frames:
        return []
    groups = _density_groups(frames)
    add(max(frames, key=lambda f: (_safe_ratio(f.center_matched, f.gt_count), -f.center_unmatched_predictions)))
    add(max(frames, key=lambda f: (f.center_missed, f.gt_count)))
    add(max(frames, key=lambda f: (f.center_unmatched_predictions, f.prediction_count)))
    add(max(frames, key=lambda f: f.gt_count))
    add(min(frames, key=lambda f: f.gt_count))
    add(max(frames, key=lambda f: (f.center_matched - f.iou50_matched, f.center_matched)))
    add(max(frames, key=lambda f: (f.gt_count, f.center_missed)))
    for group in ["lowDensity", "mediumDensity", "highDensity"]:
        group_frames = groups[group]
        if group_frames:
            add(group_frames[len(group_frames) // 2])
    for frame in _temporal_sample(frames, count=count):
        add(frame)
    return selected


def _temporal_sample(frames: list[FrameEvaluation], count: int) -> list[FrameEvaluation]:
    ordered = sorted(frames, key=lambda frame: frame.image)
    if len(ordered) <= count:
        return ordered
    return [
        ordered[int(round(index * (len(ordered) - 1) / (count - 1)))]
        for index in range(count)
    ]


def _draw_gt_vs_predictions(
    image: np.ndarray,
    sequence: str,
    image_name: str,
    gt_boxes: list[PixelBox],
    predictions: list[PredictionBox],
) -> np.ndarray:
    left = image.copy()
    right = image.copy()
    for index, box in enumerate(gt_boxes, start=1):
        _draw_box(left, box, (0, 220, 255), f"GT {index}")
    for prediction in predictions:
        _draw_prediction(right, prediction)
    header_height = 52
    combined = np.zeros((image.shape[0] + header_height, image.shape[1] * 2, 3), dtype=np.uint8)
    combined[header_height:, : image.shape[1]] = left
    combined[header_height:, image.shape[1] :] = right
    cv2.putText(
        combined,
        f"{sequence} {image_name} | Ground Truth",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        "FormicAI Prediction",
        (image.shape[1] + 18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )
    return combined


def _draw_prediction(image: np.ndarray, prediction: PredictionBox) -> None:
    _draw_box(image, prediction.box, (255, 180, 0), f"ant {prediction.confidence:.2f}")


def _draw_box(image: np.ndarray, box: PixelBox, color: tuple[int, int, int], label: str) -> None:
    x1 = int(round(box.x1))
    y1 = int(round(box.y1))
    x2 = int(round(box.x2))
    y2 = int(round(box.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _make_montage(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    if not tiles:
        raise ValueError("tiles must not be empty.")
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    montage = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        y = row * tile_height
        x = col * tile_width
        montage[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return montage


def _metric_value(metrics: Any, box_attribute: str, result_key: str) -> float:
    box_metrics = getattr(metrics, "box", None)
    value = getattr(box_metrics, box_attribute, None) if box_metrics is not None else None
    if value is None:
        value = getattr(metrics, "results_dict", {}).get(result_key, 0.0)
    return float(value)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _environment_info(torch: Any, ultralytics: Any) -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "pytorch": getattr(torch, "__version__", "unknown"),
        "cudaAvailable": cuda_available,
        "cuda": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _internal_ant2_comparison(sequences: dict[str, dict[str, object]]) -> dict[str, object]:
    ant2_path = Path("artifacts/analysis/inference_comparison/validation_metrics_nms070.json")
    ant2 = None
    if ant2_path.exists():
        ant2_data = json.loads(ant2_path.read_text(encoding="utf-8"))
        ant2 = {
            "source": str(ant2_path),
            "precision": ant2_data.get("metrics/precision(B)"),
            "recall": ant2_data.get("metrics/recall(B)"),
            "mAP50": ant2_data.get("metrics/mAP50(B)"),
            "mAP50_95": ant2_data.get("metrics/mAP50-95(B)"),
            "note": "Internal ant2 validation used temporally separated frames from the training source video.",
        }
    external = {
        sequence: {
            "precision": data["standardMetrics"]["precision"],
            "recall": data["standardMetrics"]["recall"],
            "mAP50": data["standardMetrics"]["mAP50"],
            "mAP50_95": data["standardMetrics"]["mAP50_95"],
            "centerBasedRecall": data["centerBasedMetrics"]["centerBasedRecall"],
        }
        for sequence, data in sequences.items()
    }
    return {
        "internalAnt2Validation": ant2,
        "publicExternal": external,
        "caution": (
            "mAP values are not fully apples-to-apples because domains, species appearance, camera scale, "
            "backgrounds, and annotation box policy differ. ANTS uses fixed-size boxes while FormicAI was "
            "trained with tighter ant body boxes."
        ),
    }
