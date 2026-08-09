from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from formicai.utils.video import ROI, full_frame_roi


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameExtractionConfig:
    input_path: Path
    output_dir: Path
    roi: ROI | None = None
    every_seconds: float = 0.25
    every_frames: int | None = None
    target_frame_count: int | None = None
    image_extension: str = ".jpg"

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {self.input_path}")
        if not self.input_path.is_file():
            raise ValueError(f"Input path is not a file: {self.input_path}")
        if self.every_seconds <= 0:
            raise ValueError("every_seconds must be greater than 0.")
        if self.every_frames is not None and self.every_frames <= 0:
            raise ValueError("every_frames must be greater than 0.")
        if self.target_frame_count is not None and self.target_frame_count <= 0:
            raise ValueError("target_frame_count must be greater than 0.")
        if self.target_frame_count is not None and self.every_frames is not None:
            raise ValueError("target_frame_count cannot be combined with every_frames.")
        if not self.image_extension.startswith("."):
            raise ValueError("image_extension must start with '.'.")


@dataclass(frozen=True)
class FrameExtractionResult:
    output_dir: Path
    metadata_path: Path
    extracted_frames: int
    source_frames: int
    fps: float
    roi: ROI
    selected_frame_indexes: tuple[int, ...]


class FrameExtractor:
    def __init__(self, config: FrameExtractionConfig) -> None:
        self._config = config

    def extract(self) -> FrameExtractionResult:
        self._config.validate()
        capture = cv2.VideoCapture(str(self._config.input_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video: {self._config.input_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width <= 0 or frame_height <= 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Video has no readable frames: {self._config.input_path}")
                frame_height, frame_width = frame.shape[:2]
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            roi = (self._config.roi or full_frame_roi(frame_width, frame_height)).validate_within(
                frame_width,
                frame_height,
            )
            selected_frame_indexes = self._resolve_target_frame_indexes(source_frames)
            selected_frame_index_set = set(selected_frame_indexes)
            interval_frames = None if selected_frame_indexes else self._resolve_interval_frames(fps)

            self._config.output_dir.mkdir(parents=True, exist_ok=True)
            records = []
            extracted = 0
            frame_index = 0
            source_stem = self._config.input_path.stem

            LOGGER.info("Extracting frames from %s", self._config.input_path)
            if selected_frame_indexes:
                LOGGER.info("Sampling %s temporally distributed frame(s)", len(selected_frame_indexes))
            else:
                LOGGER.info("Sampling every %s frame(s)", interval_frames)

            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                should_extract = (
                    frame_index in selected_frame_index_set
                    if selected_frame_indexes
                    else frame_index % interval_frames == 0
                )
                if should_extract:
                    timestamp_seconds = frame_index / fps if fps > 0 else 0.0
                    crop = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
                    image_name = (
                        f"{source_stem}_frame_{frame_index:06d}_"
                        f"t{timestamp_seconds:08.3f}{self._config.image_extension}"
                    )
                    image_path = self._config.output_dir / image_name
                    if not cv2.imwrite(str(image_path), crop):
                        raise RuntimeError(f"Could not write extracted frame: {image_path}")
                    records.append(
                        {
                            "image": image_name,
                            "sourceVideo": str(self._config.input_path),
                            "frameIndex": frame_index,
                            "timestampSeconds": round(timestamp_seconds, 6),
                            "sourceWidth": frame_width,
                            "sourceHeight": frame_height,
                            "roi": roi.to_dict(),
                        }
                    )
                    extracted += 1

                frame_index += 1

            metadata_path = self._config.output_dir / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "sourceVideo": str(self._config.input_path),
                        "sourceFrames": source_frames,
                        "fps": fps,
                        "samplingStrategy": "fixed_count" if selected_frame_indexes else "interval",
                        "intervalFrames": interval_frames,
                        "everySeconds": self._config.every_seconds,
                        "requestedFrameCount": self._config.target_frame_count,
                        "selectedFrameIndexes": list(selected_frame_indexes),
                        "roi": roi.to_dict(),
                        "frames": records,
                    },
                    file,
                    indent=2,
                )
                file.write("\n")

            LOGGER.info("Extracted %s frames to %s", extracted, self._config.output_dir)
            LOGGER.info("Wrote %s", metadata_path)

            return FrameExtractionResult(
                output_dir=self._config.output_dir,
                metadata_path=metadata_path,
                extracted_frames=extracted,
                source_frames=source_frames,
                fps=fps,
                roi=roi,
                selected_frame_indexes=tuple(
                    record["frameIndex"] for record in records
                ),
            )
        finally:
            capture.release()

    def _resolve_interval_frames(self, fps: float) -> int:
        if self._config.every_frames is not None:
            return self._config.every_frames
        if fps <= 0:
            raise ValueError("Video FPS is unavailable; use --every-frames instead.")
        return max(1, round(fps * self._config.every_seconds))

    def _resolve_target_frame_indexes(self, source_frames: int) -> tuple[int, ...]:
        if self._config.target_frame_count is None:
            return ()
        if source_frames <= 0:
            raise ValueError("Video frame count is unavailable; use --every-frames instead.")

        frame_count = min(self._config.target_frame_count, source_frames)
        if frame_count == 1:
            return (0,)

        return tuple(
            int(round(index * (source_frames - 1) / (frame_count - 1)))
            for index in range(frame_count)
        )
