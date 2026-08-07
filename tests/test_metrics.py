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

    aggregator.add_scored_frame(timestamp_seconds=0.0, activity_score=0.1, active_regions=2)
    aggregator.add_scored_frame(timestamp_seconds=0.1, activity_score=0.5, active_regions=4)
    aggregator.add_scored_frame(timestamp_seconds=0.2, activity_score=0.0, active_regions=0)

    summary = aggregator.summary()

    assert summary["analysis"] == {
        "processedFrames": 3,
        "warmupFrames": 0,
        "configuredWarmupFrames": 0,
        "scoredFrames": 3,
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
    assert summary["analysis"]["scoredFrames"] == 0
    assert summary["analysis"]["averageActivityScore"] == 0.0
    assert summary["analysis"]["maxActiveRegions"] == 0


def test_warmup_frames_do_not_affect_activity_statistics() -> None:
    metadata = VideoMetadata(width=10, height=10, fps=10.0, frames=4, duration_seconds=0.4)
    aggregator = MetricsAggregator(
        source=Path("samples/ants.mp4"),
        video_metadata=metadata,
        roi=ROI(0, 0, 10, 10),
        warmup_frames=2,
    )

    aggregator.add_warmup_frame()
    aggregator.add_warmup_frame()
    aggregator.add_scored_frame(timestamp_seconds=0.2, activity_score=0.2, active_regions=2)
    aggregator.add_scored_frame(timestamp_seconds=0.3, activity_score=0.4, active_regions=4)

    analysis = aggregator.summary()["analysis"]

    assert analysis["processedFrames"] == 4
    assert analysis["warmupFrames"] == 2
    assert analysis["configuredWarmupFrames"] == 2
    assert analysis["scoredFrames"] == 2
    assert analysis["averageActivityScore"] == 0.3
    assert analysis["maxActivityScore"] == 0.4
    assert analysis["peakActivityTimestampSeconds"] == 0.3
    assert analysis["averageActiveRegions"] == 3.0
    assert analysis["maxActiveRegions"] == 4


def test_only_warmup_frames_produce_zero_activity_statistics() -> None:
    metadata = VideoMetadata(width=10, height=10, fps=10.0, frames=2, duration_seconds=0.2)
    aggregator = MetricsAggregator(
        source=Path("samples/ants.mp4"),
        video_metadata=metadata,
        roi=ROI(0, 0, 10, 10),
        warmup_frames=2,
    )

    aggregator.add_warmup_frame()
    aggregator.add_warmup_frame()

    analysis = aggregator.summary()["analysis"]

    assert analysis["processedFrames"] == 2
    assert analysis["warmupFrames"] == 2
    assert analysis["scoredFrames"] == 0
    assert analysis["averageActivityScore"] == 0.0
    assert analysis["maxActivityScore"] == 0.0
    assert analysis["peakActivityTimestampSeconds"] == 0.0
    assert analysis["averageActiveRegions"] == 0.0
    assert analysis["maxActiveRegions"] == 0
