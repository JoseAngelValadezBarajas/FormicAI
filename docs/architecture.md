# FormicAI Vision Architecture

FormicAI Vision v0.1 focuses on motion analysis in video. FormicAI Vision v0.2 closes the first real ant-detection baseline through dataset preparation, YOLO inference, validation, and external benchmark tooling.

```text
                 VIDEO
                   |
        +----------+----------+
        |                     |
        v                     v
 Motion Analysis        Ant Detection
    MOG2                    YOLO
        |                     |
        v                     v
 activity metrics       bounding boxes
 heatmap                confidence
 motion mask            detection JSON
```

## Motion Analysis

```text
CLI
 |
 v
VideoAnalyzer
 |-- MotionDetector
 |-- MetricsAggregator
 |-- HeatmapAccumulator
 |-- AnnotatedVideoWriter
 `-- Optional motion mask writer
```

## Current Flow

1. The CLI parses the input video path, optional ROI, and output directory.
2. `VideoAnalyzer` validates the video, reads metadata, and streams frames one by one.
3. `MotionDetector` uses OpenCV background subtraction plus light noise filtering to produce moving regions.
4. `AnnotatedVideoWriter` writes bounding boxes and frame-level activity data to `output/analyzed.mp4`.
5. If enabled, the same writer infrastructure writes the final binary motion mask to `output/motion_mask.mp4`.
6. `HeatmapAccumulator` stores only an accumulated motion matrix and writes `output/activity_heatmap.png`.
7. `MetricsAggregator` keeps O(1) incremental counters and summarizes scored frames into `output/metrics.json`.

## Dataset Preparation

```text
CLI dataset commands
 |
 v
FrameExtractor
 |-- ROI crop
 |-- deterministic frame names
 `-- metadata.json

DatasetValidator
 |-- dataset.yaml
 |-- images/train, images/val
 `-- labels/train, labels/val
```

## Future Direction

The closed v0.2 path includes Roboflow import, temporal dataset preparation, a frozen YOLO baseline, validation metrics, visual validation, video inference, and public ANTS benchmark evaluation. Tracking, .NET APIs, Angular dashboards, Google Cloud services, Gemini-based multimodal analysis, RAG, and IoT telemetry remain out of scope for this iteration.
