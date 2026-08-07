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
class FrameMetrics:
    frame_index: int
    timestamp_seconds: float
    activity_score: float
    active_regions: int


@dataclass(frozen=True)
class AnalysisResult:
    annotated_video_path: Path
    heatmap_path: Path
    metrics_path: Path
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
        self._frames: list[FrameMetrics] = []

    def add_frame(
        self,
        frame_index: int,
        timestamp_seconds: float,
        activity_score: float,
        active_regions: int,
    ) -> None:
        self._frames.append(
            FrameMetrics(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                activity_score=activity_score,
                active_regions=active_regions,
            )
        )

    def summary(self) -> dict[str, object]:
        processed_frames = len(self._frames)
        activity_scores = [frame.activity_score for frame in self._frames]
        active_region_counts = [frame.active_regions for frame in self._frames]

        average_activity = (
            sum(activity_scores) / processed_frames if processed_frames else 0.0
        )
        max_activity = max(activity_scores, default=0.0)
        peak_frame = max(
            self._frames,
            key=lambda frame: frame.activity_score,
            default=None,
        )
        average_regions = (
            sum(active_region_counts) / processed_frames if processed_frames else 0.0
        )

        return {
            "source": str(self._source),
            "video": self._video_metadata.to_dict(),
            "roi": self._roi.to_dict(),
            "parameters": self._parameters,
            "analysis": {
                "processedFrames": processed_frames,
                "warmupFrames": min(self._warmup_frames, processed_frames),
                "averageActivityScore": round(average_activity, 6),
                "maxActivityScore": round(max_activity, 6),
                "peakActivityTimestampSeconds": round(
                    peak_frame.timestamp_seconds if peak_frame else 0.0,
                    6,
                ),
                "averageActiveRegions": round(average_regions, 6),
                "maxActiveRegions": max(active_region_counts, default=0),
            },
            "activityScoreDefinition": (
                "Number of pixels kept as motion after noise filtering and contour "
                "filtering divided by the total ROI pixel count. The value is clamped "
                "to the 0.0-1.0 range. Warmup frames are processed but recorded with "
                "score 0.0 while the background model stabilizes."
            ),
        }
