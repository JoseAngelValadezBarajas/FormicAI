from __future__ import annotations

from pathlib import Path

from formicai.models.metrics import MetricsAggregator, VideoMetadata
from formicai.utils.video import ROI


def test_metrics_aggregator_summarizes_activity() -> None:
    metadata = VideoMetadata(
        width=100,
        height=50,
        fps=10.0,
        frames=3,
        duration_seconds=0.3,
    )
    aggregator = MetricsAggregator(
        source=Path("samples/ants.mp4"),
        video_metadata=metadata,
        roi=ROI(0, 0, 100, 50),
    )

    aggregator.add_frame(frame_index=0, timestamp_seconds=0.0, activity_score=0.1, active_regions=2)
    aggregator.add_frame(frame_index=1, timestamp_seconds=0.1, activity_score=0.5, active_regions=4)
    aggregator.add_frame(frame_index=2, timestamp_seconds=0.2, activity_score=0.0, active_regions=0)

    summary = aggregator.summary()

    assert summary["analysis"] == {
        "processedFrames": 3,
        "warmupFrames": 0,
        "averageActivityScore": 0.2,
        "maxActivityScore": 0.5,
        "peakActivityTimestampSeconds": 0.1,
        "averageActiveRegions": 2.0,
        "maxActiveRegions": 4,
    }


def test_metrics_aggregator_handles_empty_input() -> None:
    metadata = VideoMetadata(width=10, height=10, fps=0.0, frames=0, duration_seconds=0.0)
    aggregator = MetricsAggregator(
        source=Path("empty.mp4"),
        video_metadata=metadata,
        roi=ROI(0, 0, 10, 10),
    )

    summary = aggregator.summary()

    assert summary["analysis"]["processedFrames"] == 0
    assert summary["analysis"]["warmupFrames"] == 0
    assert summary["analysis"]["averageActivityScore"] == 0.0
    assert summary["analysis"]["maxActiveRegions"] == 0
