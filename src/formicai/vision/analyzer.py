from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from formicai.models.metrics import AnalysisResult, MetricsAggregator, VideoMetadata
from formicai.utils.video import ROI, full_frame_roi
from formicai.vision.heatmap import HeatmapAccumulator
from formicai.vision.motion_detector import MotionDetection, MotionDetector
from formicai.vision.video_writer import AnnotatedVideoWriter


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: Path
    output_dir: Path = Path("output")
    roi: ROI | None = None
    min_motion_area: float | None = None
    min_motion_area_ratio: float = 0.0001
    background_history: int = 500
    background_var_threshold: float = 64.0
    progress_interval: int = 100
    warmup_frames: int = 30
    write_motion_mask: bool = False

    def validate(self) -> None:
        if self.background_history <= 0:
            raise ValueError("background_history must be greater than 0.")
        if self.background_var_threshold <= 0:
            raise ValueError("background_var_threshold must be greater than 0.")
        if self.progress_interval < 0:
            raise ValueError("progress_interval must be greater than or equal to 0.")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be greater than or equal to 0.")
        resolve_min_motion_area(
            roi=ROI(x=0, y=0, width=1, height=1),
            min_motion_area=self.min_motion_area,
            min_motion_area_ratio=self.min_motion_area_ratio,
        )


class VideoAnalyzer:
    def __init__(self, config: AnalysisConfig) -> None:
        self._config = config

    def analyze(self) -> AnalysisResult:
        self._config.validate()
        input_path = self._config.input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")

        LOGGER.info("Opening video %s", input_path)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {input_path}")

        try:
            success, first_frame = capture.read()
            if not success or first_frame is None:
                raise RuntimeError(f"Video has no readable frames: {input_path}")

            metadata = self._read_metadata(capture, first_frame)
            LOGGER.info(
                "Video: %sx%s @ %.3f FPS",
                metadata.width,
                metadata.height,
                metadata.fps,
            )
            if metadata.frames > 0:
                LOGGER.info("Processing %s frames", metadata.frames)
            else:
                LOGGER.info("Processing frames until end of stream")

            roi = (self._config.roi or full_frame_roi(metadata.width, metadata.height)).validate_within(
                metadata.width,
                metadata.height,
            )

            output_dir = self._config.output_dir
            annotated_video_path = output_dir / "analyzed.mp4"
            heatmap_path = output_dir / "activity_heatmap.png"
            metrics_path = output_dir / "metrics.json"
            motion_mask_path = (
                output_dir / "motion_mask.mp4"
                if self._config.write_motion_mask
                else None
            )

            min_motion_area = self._resolve_min_motion_area(roi)
            LOGGER.info("Minimum motion contour area: %.1f px", min_motion_area)

            detector = MotionDetector(
                min_area=min_motion_area,
                history=self._config.background_history,
                var_threshold=self._config.background_var_threshold,
            )
            heatmap = HeatmapAccumulator(metadata.width, metadata.height)
            warmup_frames = max(0, self._config.warmup_frames)
            if warmup_frames > 0:
                LOGGER.info(
                    "Using %s warmup frames for background model stabilization",
                    warmup_frames,
                )

            metrics = MetricsAggregator(
                input_path,
                metadata,
                roi,
                warmup_frames=warmup_frames,
                parameters={
                    "minMotionAreaPixels": round(min_motion_area, 3),
                    "minMotionAreaRatio": self._config.min_motion_area_ratio,
                    "backgroundHistory": self._config.background_history,
                    "backgroundVarThreshold": self._config.background_var_threshold,
                    "writeMotionMask": self._config.write_motion_mask,
                },
            )
            writer_fps = metadata.fps if metadata.fps > 0 else 30.0

            with ExitStack() as stack:
                writer = stack.enter_context(
                    AnnotatedVideoWriter(
                        annotated_video_path,
                        fps=writer_fps,
                        frame_size=(metadata.width, metadata.height),
                    )
                )
                motion_mask_writer = (
                    stack.enter_context(
                        AnnotatedVideoWriter(
                            motion_mask_path,
                            fps=writer_fps,
                            frame_size=(metadata.width, metadata.height),
                        )
                    )
                    if motion_mask_path is not None
                    else None
                )
                processed_frames = self._process_frames(
                    capture=capture,
                    first_frame=first_frame,
                    fps=writer_fps,
                    roi=roi,
                    detector=detector,
                    heatmap=heatmap,
                    metrics=metrics,
                    writer=writer,
                    motion_mask_writer=motion_mask_writer,
                    total_frames=metadata.frames,
                )

            LOGGER.info("Processed %s frames", processed_frames)
            heatmap.save(heatmap_path)
            metrics_payload = metrics.summary()
            self._write_metrics(metrics_path, metrics_payload)

            return AnalysisResult(
                annotated_video_path=annotated_video_path,
                heatmap_path=heatmap_path,
                metrics_path=metrics_path,
                motion_mask_video_path=motion_mask_path,
                metrics=metrics_payload,
            )
        finally:
            capture.release()

    def _process_frames(
        self,
        capture: cv2.VideoCapture,
        first_frame: np.ndarray,
        fps: float,
        roi: ROI,
        detector: MotionDetector,
        heatmap: HeatmapAccumulator,
        metrics: MetricsAggregator,
        writer: AnnotatedVideoWriter,
        motion_mask_writer: AnnotatedVideoWriter | None,
        total_frames: int,
    ) -> int:
        frame_index = 0
        processed_frames = 0
        current_frame: np.ndarray | None = first_frame

        while current_frame is not None:
            timestamp_seconds = frame_index / fps if fps > 0 else 0.0
            roi_frame = current_frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            detection_result = detector.detect(roi_frame)
            is_warmup_frame = frame_index < max(0, self._config.warmup_frames)
            if is_warmup_frame:
                motion_mask = np.zeros_like(detection_result.motion_mask)
                detections: list[MotionDetection] = []
                activity_score = 0.0
            else:
                motion_mask = detection_result.motion_mask
                detections = detection_result.detections
                activity_score = detection_result.activity_score

            annotated = current_frame.copy()
            self._draw_annotations(
                frame=annotated,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                roi=roi,
                detections=detections,
                activity_score=activity_score,
            )
            heatmap.add_motion(motion_mask, roi.x, roi.y)
            if is_warmup_frame:
                metrics.add_warmup_frame()
            else:
                metrics.add_scored_frame(
                    timestamp_seconds=timestamp_seconds,
                    activity_score=activity_score,
                    active_regions=len(detections),
                )
            writer.write(annotated)
            if motion_mask_writer is not None:
                motion_mask_writer.write(
                    self._render_full_frame_motion_mask(
                        motion_mask=motion_mask,
                        roi=roi,
                        frame_width=annotated.shape[1],
                        frame_height=annotated.shape[0],
                    )
                )

            processed_frames += 1
            if (
                self._config.progress_interval > 0
                and processed_frames % self._config.progress_interval == 0
            ):
                if total_frames > 0:
                    LOGGER.info("Processed %s/%s frames", processed_frames, total_frames)
                else:
                    LOGGER.info("Processed %s frames", processed_frames)

            success, next_frame = capture.read()
            current_frame = next_frame if success and next_frame is not None else None
            frame_index += 1

        return processed_frames

    @staticmethod
    def _read_metadata(
        capture: cv2.VideoCapture,
        first_frame: np.ndarray,
    ) -> VideoMetadata:
        frame_height, frame_width = first_frame.shape[:2]
        captured_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        captured_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        width = captured_width if captured_width > 0 else frame_width
        height = captured_height if captured_height > 0 else frame_height
        duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frames=frame_count,
            duration_seconds=round(duration_seconds, 6),
        )

    def _resolve_min_motion_area(self, roi: ROI) -> float:
        return resolve_min_motion_area(
            roi=roi,
            min_motion_area=self._config.min_motion_area,
            min_motion_area_ratio=self._config.min_motion_area_ratio,
        )

    @staticmethod
    def _draw_annotations(
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
        roi: ROI,
        detections: list[MotionDetection],
        activity_score: float,
    ) -> None:
        if roi.x != 0 or roi.y != 0 or roi.width != frame.shape[1] or roi.height != frame.shape[0]:
            cv2.rectangle(
                frame,
                (roi.x, roi.y),
                (roi.x + roi.width, roi.y + roi.height),
                color=(255, 180, 0),
                thickness=2,
            )

        for detection in detections:
            x, y, width, height = detection.to_bbox()
            top_left = (roi.x + x, roi.y + y)
            bottom_right = (roi.x + x + width, roi.y + y + height)
            cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=2)

        lines = [
            f"Frame: {frame_index}",
            f"Timestamp: {timestamp_seconds:.2f}s",
            f"Active regions: {len(detections)}",
            f"Activity score: {activity_score:.4f}",
        ]
        VideoAnalyzer._draw_text_panel(frame, lines)

    @staticmethod
    def _draw_text_panel(frame: np.ndarray, lines: list[str]) -> None:
        x = 16
        y = 28
        line_height = 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        for index, line in enumerate(lines):
            position = (x, y + index * line_height)
            cv2.putText(
                frame,
                line,
                position,
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                position,
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _render_full_frame_motion_mask(
        motion_mask: np.ndarray,
        roi: ROI,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        full_frame_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        full_frame_mask[
            roi.y : roi.y + roi.height,
            roi.x : roi.x + roi.width,
        ] = motion_mask
        return cv2.cvtColor(full_frame_mask, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _write_metrics(output_path: Path, payload: dict[str, object]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")


def resolve_min_motion_area(
    roi: ROI,
    min_motion_area: float | None,
    min_motion_area_ratio: float,
) -> float:
    if min_motion_area is not None:
        if min_motion_area <= 0:
            raise ValueError("min_motion_area must be greater than 0.")
        return min_motion_area

    if min_motion_area_ratio <= 0:
        raise ValueError("min_motion_area_ratio must be greater than 0.")

    return max(1.0, roi.area * min_motion_area_ratio)
