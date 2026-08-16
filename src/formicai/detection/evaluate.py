from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from formicai.dataset.yolo import find_images, matching_label_path, parse_yolo_label_file


@dataclass(frozen=True)
class ExternalEvaluationConfig:
    external_test_dir: Path
    model_path: Path
    output_path: Path = Path("artifacts/analysis/external_test_metrics.json")
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640
    device: str | int | None = None
    end2end: bool | None = None


@dataclass(frozen=True)
class ExternalVideoDataset:
    name: str
    root_dir: Path
    dataset_yaml: Path
    images: int
    annotations: int
    images_without_annotations: int


@dataclass(frozen=True)
class ExternalVideoMetrics:
    video: str
    dataset_yaml: Path
    images: int
    annotations: int
    images_without_annotations: int
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class ExternalEvaluationResult:
    external_test_dir: Path
    model_path: Path
    output_path: Path
    confidence: float
    iou: float
    image_size: int
    end2end: bool | None
    videos: tuple[ExternalVideoMetrics, ...]

    def to_text(self) -> str:
        lines = [
            f"External test: {self.external_test_dir}",
            f"Model: {self.model_path}",
            f"Metrics JSON: {self.output_path}",
            f"Confidence threshold: {self.confidence:.3f}",
            f"NMS IoU threshold: {self.iou:.3f}",
            f"End-to-end override: {self.end2end}",
            "",
            "Per-video metrics:",
        ]
        for video in self.videos:
            lines.append(
                f"- {video.video}: images={video.images}, annotations={video.annotations}, "
                f"precision={video.precision:.4f}, recall={video.recall:.4f}, "
                f"mAP50={video.map50:.4f}, mAP50-95={video.map50_95:.4f}"
            )
        return "\n".join(lines)


class ExternalEvaluator:
    def __init__(self, config: ExternalEvaluationConfig) -> None:
        self._config = config

    def evaluate(self) -> ExternalEvaluationResult:
        if not self._config.external_test_dir.exists():
            raise FileNotFoundError(f"External test directory does not exist: {self._config.external_test_dir}")
        if not self._config.model_path.exists():
            raise FileNotFoundError(f"Model does not exist: {self._config.model_path}")
        if not 0.0 <= self._config.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        video_datasets = self._discover_video_datasets()

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for external evaluation. Install with .[ml].") from exc

        model = YOLO(str(self._config.model_path))
        metrics = tuple(self._evaluate_video(model, dataset) for dataset in video_datasets)
        result = ExternalEvaluationResult(
            external_test_dir=self._config.external_test_dir,
            model_path=self._config.model_path,
            output_path=self._config.output_path,
            confidence=self._config.confidence,
            iou=self._config.iou,
            image_size=self._config.image_size,
            end2end=self._config.end2end,
            videos=metrics,
        )
        self._write_result(result)
        return result

    def _discover_video_datasets(self) -> tuple[ExternalVideoDataset, ...]:
        candidates = [
            path
            for path in sorted(self._config.external_test_dir.iterdir())
            if path.is_dir() and (path / "images").exists()
        ]
        if (self._config.external_test_dir / "images").exists():
            candidates = [self._config.external_test_dir]
        if not candidates:
            raise ValueError(
                "External test directory must contain per-video folders with images/ and labels/."
            )
        return tuple(self._validate_video_dataset(candidate) for candidate in candidates)

    def _validate_video_dataset(self, root_dir: Path) -> ExternalVideoDataset:
        images_dir = root_dir / "images"
        labels_dir = root_dir / "labels"
        if not labels_dir.exists():
            raise ValueError(f"Missing labels directory for external test video: {labels_dir}")

        images = find_images(images_dir)
        if not images:
            raise ValueError(f"External test video has no images: {images_dir}")

        annotations = 0
        images_without_annotations = 0
        seen_label_paths: set[Path] = set()
        errors: list[str] = []
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

        for label_path in sorted(labels_dir.glob("*.txt")):
            if label_path not in seen_label_paths:
                errors.append(f"Label without matching image: {label_path}")

        if errors:
            details = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"External test labels are not valid for {root_dir.name}.\n{details}")

        dataset_yaml = self._evaluation_dataset_yaml(root_dir)
        dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        dataset_yaml.write_text(
            f"path: {root_dir.resolve().as_posix()}\ntrain: images\nval: images\nnames:\n  0: ant\n",
            encoding="utf-8",
        )
        return ExternalVideoDataset(
            name=root_dir.name,
            root_dir=root_dir,
            dataset_yaml=dataset_yaml,
            images=len(images),
            annotations=annotations,
            images_without_annotations=images_without_annotations,
        )

    def _evaluation_dataset_yaml(self, root_dir: Path) -> Path:
        return self._config.output_path.parent / "external_test_dataset_yamls" / f"{root_dir.name}.yaml"

    def _evaluate_video(self, model: Any, dataset: ExternalVideoDataset) -> ExternalVideoMetrics:
        validation_kwargs: dict[str, Any] = {
            "data": str(dataset.dataset_yaml),
            "split": "val",
            "imgsz": self._config.image_size,
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "verbose": False,
            "plots": False,
            "save_json": False,
            "workers": 0,
            "project": str(self._config.output_path.parent / "external_test_val_runs"),
            "name": dataset.name,
            "exist_ok": True,
        }
        if self._config.device is not None:
            validation_kwargs["device"] = self._config.device
        if self._config.end2end is not None:
            validation_kwargs["end2end"] = self._config.end2end

        ultralytics_metrics = model.val(**validation_kwargs)
        return ExternalVideoMetrics(
            video=dataset.name,
            dataset_yaml=dataset.dataset_yaml,
            images=dataset.images,
            annotations=dataset.annotations,
            images_without_annotations=dataset.images_without_annotations,
            precision=_metric_value(ultralytics_metrics, "mp", "metrics/precision(B)"),
            recall=_metric_value(ultralytics_metrics, "mr", "metrics/recall(B)"),
            map50=_metric_value(ultralytics_metrics, "map50", "metrics/mAP50(B)"),
            map50_95=_metric_value(ultralytics_metrics, "map", "metrics/mAP50-95(B)"),
        )

    def _write_result(self, result: ExternalEvaluationResult) -> None:
        videos = [
            {
                "video": video.video,
                "datasetYaml": str(video.dataset_yaml),
                "images": video.images,
                "annotations": video.annotations,
                "imagesWithoutAnnotations": video.images_without_annotations,
                "precision": video.precision,
                "recall": video.recall,
                "mAP50": video.map50,
                "mAP50_95": video.map50_95,
            }
            for video in result.videos
        ]
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.output_path.write_text(
            json.dumps(
                {
                    "externalTestDir": str(result.external_test_dir),
                    "model": str(result.model_path),
                    "params": {
                        "confidence": result.confidence,
                        "iou": result.iou,
                        "imageSize": result.image_size,
                        "end2end": result.end2end,
                    },
                    "videos": videos,
                    "macroAverage": _macro_average(videos),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _metric_value(metrics: Any, box_attribute: str, result_key: str) -> float:
    box_metrics = getattr(metrics, "box", None)
    value = getattr(box_metrics, box_attribute, None) if box_metrics is not None else None
    if value is None:
        value = getattr(metrics, "results_dict", {}).get(result_key, 0.0)
    return float(value)


def _macro_average(videos: list[dict[str, object]]) -> dict[str, float | int]:
    if not videos:
        return {
            "videos": 0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
        }
    return {
        "videos": len(videos),
        "precision": mean(float(video["precision"]) for video in videos),
        "recall": mean(float(video["recall"]) for video in videos),
        "mAP50": mean(float(video["mAP50"]) for video in videos),
        "mAP50_95": mean(float(video["mAP50_95"]) for video in videos),
    }
