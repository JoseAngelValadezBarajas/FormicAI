from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from formicai.utils.video import ROI, full_frame_roi


@dataclass(frozen=True)
class VideoDetectionConfig:
    input_path: Path
    model_path: Path
    output_video_path: Path = Path("output/ant_detections.mp4")
    output_jsonl_path: Path = Path("output/ant_detections.jsonl")
    roi: ROI | None = None
    confidence: float = 0.25
    iou: float = 0.70
    end2end: bool | None = None
    image_size: int = 640
    device: str | int | None = None


@dataclass(frozen=True)
class VideoDetectionResult:
    output_video_path: Path
    output_jsonl_path: Path
    processed_frames: int
    total_detections: int
    confidence: float
    iou: float
    end2end: bool | None


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

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for video detection. Install with .[ml].") from exc

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
            capture.release()
            raise RuntimeError(f"Could not create video writer: {self._config.output_video_path}")

        model = YOLO(str(self._config.model_path))
        frame_index = 0
        total_detections = 0

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
                }
                if self._config.device is not None:
                    prediction_kwargs["device"] = self._config.device
                if self._config.end2end is not None:
                    prediction_kwargs["end2end"] = self._config.end2end

                result = model.predict(**prediction_kwargs)[0]
                detections = []
                names = result.names if isinstance(result.names, dict) else {}
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    x_min, y_min, x_max, y_max = (float(value) for value in box.xyxy[0].tolist())
                    global_x = int(round(x_min + roi.x))
                    global_y = int(round(y_min + roi.y))
                    width = int(round(x_max - x_min))
                    height = int(round(y_max - y_min))
                    class_name = str(names.get(class_id, "ant"))

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

        capture.release()
        writer.release()
        return VideoDetectionResult(
            output_video_path=self._config.output_video_path,
            output_jsonl_path=self._config.output_jsonl_path,
            processed_frames=frame_index,
            total_detections=total_detections,
            confidence=self._config.confidence,
            iou=self._config.iou,
            end2end=self._config.end2end,
        )


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
