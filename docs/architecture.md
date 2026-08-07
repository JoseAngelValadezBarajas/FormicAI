# FormicAI Vision v0.1 Architecture

FormicAI Vision v0.1 focuses only on motion analysis in video. It does not identify ants.

```text
CLI
 |
 v
VideoAnalyzer
 |-- MotionDetector
 |-- MetricsAggregator
 |-- HeatmapAccumulator
 `-- AnnotatedVideoWriter
```

## Current Flow

1. The CLI parses the input video path, optional ROI, and output directory.
2. `VideoAnalyzer` validates the video, reads metadata, and streams frames one by one.
3. `MotionDetector` uses OpenCV background subtraction plus light noise filtering to produce moving regions.
4. `AnnotatedVideoWriter` writes bounding boxes and frame-level activity data to `output/analyzed.mp4`.
5. `HeatmapAccumulator` stores only an accumulated motion matrix and writes `output/activity_heatmap.png`.
6. `MetricsAggregator` summarizes activity into `output/metrics.json`.

## Future Direction

Future versions may add object detection, tracking, a .NET API, Angular dashboards, Google Cloud services, Gemini-based multimodal analysis, and IoT telemetry. Those are intentionally not implemented in v0.1.
