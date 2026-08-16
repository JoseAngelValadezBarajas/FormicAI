from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from formicai.detection.spec import CURRENT_CHAMPION, identify_detector_by_sha256, validate_model_classes
from formicai.utils.hashing import sha256_file
from formicai.utils.video import ROI, full_frame_roi


@dataclass(frozen=True)
class VideoDetectionConfig:
    input_path: Path
    model_path: Path
    output_video_path: Path = Path("output/ant_detections.mp4")
    output_jsonl_path: Path = Path("output/ant_detections.jsonl")
    output_metadata_path: Path | None = None
    roi: ROI | None = None
    confidence: float = CURRENT_CHAMPION.confidence
    iou: float = CURRENT_CHAMPION.iou
    end2end: bool = CURRENT_CHAMPION.end2end
    image_size: int = CURRENT_CHAMPION.image_size
    device: str | int | None = None
    expected_model_sha256: str | None = None


@dataclass(frozen=True)
class VideoDetectionResult:
    output_video_path: Path
    output_jsonl_path: Path
    output_metadata_path: Path
    processed_frames: int
    total_detections: int
    confidence: float
    iou: float
    end2end: bool
    image_size: int
    source_sha256: str
    model_sha256: str


class VideoDetector:
    def __init__(self, config: VideoDetectionConfig) -> None:
        self._config = config

    def detect(self) -> VideoDetectionResult:
        if not self._config.input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {self._config.input_path}")
        if not self._config.model_path.exists():
            raise FileNotFoundError(f"Model does not exist: {self._config.model_path}")
        if not 0.0 <= self._config.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        model_sha256 = sha256_file(self._config.model_path)
        if (
            self._config.expected_model_sha256 is not None
            and model_sha256.lower() != self._config.expected_model_sha256.lower()
        ):
            raise ValueError(
                "Champion model integrity check failed. "
                f"Expected SHA256 {self._config.expected_model_sha256}. "
                f"Actual SHA256 {model_sha256}."
            )

        source_sha256 = sha256_file(self._config.input_path)

        try:
            import ultralytics
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for video detection. Install with .[ml].") from exc

        model = ultralytics.YOLO(str(self._config.model_path))
        class_names = validate_model_classes(getattr(model, "names", None), CURRENT_CHAMPION.classes)
        frame_index = 0
        total_detections = 0
        metadata_path = self._config.output_metadata_path or default_run_metadata_path(self._config.output_jsonl_path)

        capture: Any | None = None
        writer: Any | None = None
        fps = 30.0
        try:
            capture = cv2.VideoCapture(str(self._config.input_path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {self._config.input_path}")

            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
            roi = (self._config.roi or full_frame_roi(frame_width, frame_height)).validate_within(
                frame_width,
                frame_height,
            )

            self._config.output_video_path.parent.mkdir(parents=True, exist_ok=True)
            self._config.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self._config.output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video writer: {self._config.output_video_path}")

            with self._config.output_jsonl_path.open("w", encoding="utf-8") as jsonl:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    crop = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
                    prediction_kwargs: dict[str, Any] = {
                        "source": crop,
                        "imgsz": self._config.image_size,
                        "conf": self._config.confidence,
                        "iou": self._config.iou,
                        "verbose": False,
                        "end2end": self._config.end2end,
                    }
                    if self._config.device is not None:
                        prediction_kwargs["device"] = self._config.device

                    result = model.predict(**prediction_kwargs)[0]
                    detections = []
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        if class_id not in class_names:
                            raise RuntimeError(f"Unexpected detection class id: {class_id}")
                        confidence = float(box.conf.item())
                        x_min, y_min, x_max, y_max = (float(value) for value in box.xyxy[0].tolist())
                        global_x = int(round(x_min + roi.x))
                        global_y = int(round(y_min + roi.y))
                        width = int(round(x_max - x_min))
                        height = int(round(y_max - y_min))
                        class_name = class_names[class_id]

                        detections.append(
                            {
                                "classId": class_id,
                                "className": class_name,
                                "confidence": confidence,
                                "bbox": {
                                    "x": global_x,
                                    "y": global_y,
                                    "width": width,
                                    "height": height,
                                },
                            }
                        )
                        _draw_detection(frame, class_name, confidence, global_x, global_y, width, height)

                    total_detections += len(detections)
                    jsonl.write(
                        json.dumps(
                            {
                                "frameIndex": frame_index,
                                "timestampSeconds": frame_index / fps,
                                "detections": detections,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    jsonl.flush()
                    writer.write(frame)
                    frame_index += 1

            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    _build_run_metadata(
                        config=self._config,
                        metadata_path=metadata_path,
                        source_sha256=source_sha256,
                        model_sha256=model_sha256,
                        classes=class_names,
                        processed_frames=frame_index,
                        total_detections=total_detections,
                        ultralytics_version=str(getattr(ultralytics, "__version__", "unknown")),
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()

        return VideoDetectionResult(
            output_video_path=self._config.output_video_path,
            output_jsonl_path=self._config.output_jsonl_path,
            output_metadata_path=metadata_path,
            processed_frames=frame_index,
            total_detections=total_detections,
            confidence=self._config.confidence,
            iou=self._config.iou,
            end2end=self._config.end2end,
            image_size=self._config.image_size,
            source_sha256=source_sha256,
            model_sha256=model_sha256,
        )


def default_run_metadata_path(output_jsonl_path: Path) -> Path:
    return output_jsonl_path.with_suffix(".run.json")


def _build_run_metadata(
    *,
    config: VideoDetectionConfig,
    metadata_path: Path,
    source_sha256: str,
    model_sha256: str,
    classes: dict[int, str],
    processed_frames: int,
    total_detections: int,
    ultralytics_version: str,
) -> dict[str, Any]:
    inference = _inference_metadata(config)
    detector_spec = identify_detector_by_sha256(model_sha256)
    if detector_spec is None:
        model_experiment = "CUSTOM / UNREGISTERED"
        model_status = "CUSTOM / UNREGISTERED"
        is_current_champion = False
    else:
        model_experiment = detector_spec.experiment
        model_status = detector_spec.status
        is_current_champion = detector_spec.sha256.lower() == CURRENT_CHAMPION.sha256.lower()

    return {
        "schemaVersion": 1,
        "createdAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "path": str(config.input_path),
            "sha256": source_sha256,
        },
        "model": {
            "path": str(config.model_path),
            "sha256": model_sha256,
            "experiment": model_experiment,
            "status": model_status,
            "isCurrentChampion": is_current_champion,
        },
        "inference": inference,
        "classes": {str(class_id): class_name for class_id, class_name in sorted(classes.items())},
        "environment": {
            "python": platform.python_version(),
            "ultralytics": ultralytics_version,
            "opencv": cv2.__version__,
        },
        "result": {
            "processedFrames": processed_frames,
            "totalDetections": total_detections,
            "outputVideo": str(config.output_video_path),
            "outputJsonl": str(config.output_jsonl_path),
            "outputMetadata": str(metadata_path),
        },
    }


def _inference_metadata(config: VideoDetectionConfig) -> dict[str, Any]:
    values = {
        "confidence": config.confidence,
        "iou": config.iou,
        "imageSize": config.image_size,
        "end2end": config.end2end,
    }
    canonical = {
        "confidence": CURRENT_CHAMPION.confidence,
        "iou": CURRENT_CHAMPION.iou,
        "imageSize": CURRENT_CHAMPION.image_size,
        "end2end": CURRENT_CHAMPION.end2end,
    }
    overrides = {
        key: {"canonical": canonical[key], "actual": value}
        for key, value in values.items()
        if value != canonical[key]
    }
    return {
        **values,
        "canonicalInference": not overrides,
        "overrides": overrides,
    }


def _draw_detection(
    frame: Any,
    class_name: str,
    confidence: float,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    color = (0, 220, 255)
    x2 = x + width
    y2 = y + height
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    label = f"{class_name} {confidence:.2f}"
    label_y = max(18, y - 6)
    cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
