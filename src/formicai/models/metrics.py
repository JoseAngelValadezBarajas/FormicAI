from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from formicai.utils.video import ROI


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frames: int
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
            "durationSeconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class AnalysisResult:
    annotated_video_path: Path
    heatmap_path: Path
    metrics_path: Path
    motion_mask_video_path: Path | None
    metrics: dict[str, object]


class MetricsAggregator:
    def __init__(
        self,
        source: Path,
        video_metadata: VideoMetadata,
        roi: ROI,
        warmup_frames: int = 0,
        parameters: dict[str, int | float] | None = None,
    ) -> None:
        self._source = source
        self._video_metadata = video_metadata
        self._roi = roi
        self._warmup_frames = warmup_frames
        self._parameters = parameters or {}
        self._processed_frames = 0
        self._warmup_frames_seen = 0
        self._scored_frames = 0
        self._total_activity_score = 0.0
        self._max_activity_score = 0.0
        self._peak_activity_timestamp_seconds = 0.0
        self._total_active_regions = 0
        self._max_active_regions = 0

    def add_warmup_frame(self) -> None:
        self._processed_frames += 1
        self._warmup_frames_seen += 1

    def add_scored_frame(
        self,
        timestamp_seconds: float,
        activity_score: float,
        active_regions: int,
    ) -> None:
        self._processed_frames += 1
        self._scored_frames += 1
        self._total_activity_score += activity_score
        self._total_active_regions += active_regions

        if activity_score > self._max_activity_score:
            self._max_activity_score = activity_score
            self._peak_activity_timestamp_seconds = timestamp_seconds

        if active_regions > self._max_active_regions:
            self._max_active_regions = active_regions

    def summary(self) -> dict[str, object]:
        average_activity = (
            self._total_activity_score / self._scored_frames
            if self._scored_frames
            else 0.0
        )
        average_regions = (
            self._total_active_regions / self._scored_frames
            if self._scored_frames
            else 0.0
        )

        return {
            "source": str(self._source),
            "video": self._video_metadata.to_dict(),
            "roi": self._roi.to_dict(),
            "parameters": self._parameters,
            "analysis": {
                "processedFrames": self._processed_frames,
                "warmupFrames": self._warmup_frames_seen,
                "configuredWarmupFrames": self._warmup_frames,
                "scoredFrames": self._scored_frames,
                "averageActivityScore": round(average_activity, 6),
                "maxActivityScore": round(self._max_activity_score, 6),
                "peakActivityTimestampSeconds": round(
                    self._peak_activity_timestamp_seconds,
                    6,
                ),
                "averageActiveRegions": round(average_regions, 6),
                "maxActiveRegions": self._max_active_regions,
            },
            "activityScoreDefinition": (
                "Number of pixels kept as motion after noise filtering and contour "
                "filtering divided by the total ROI pixel count. The value is clamped "
                "to the 0.0-1.0 range. Warmup frames are processed to stabilize the "
                "background model, but they are excluded from activity statistics."
            ),
        }
