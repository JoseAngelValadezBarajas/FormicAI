from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from formicai.vision.analyzer import AnalysisConfig, VideoAnalyzer


def create_synthetic_motion_video(path: Path) -> None:
    width = 320
    height = 240
    fps = 12.0
    frames = 60
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic test video.")

    for frame_index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if 10 <= frame_index < 45:
            x = 20 + (frame_index - 10) * 4
            cv2.rectangle(frame, (x, 100), (x + 30, 120), (255, 255, 255), thickness=-1)
        writer.write(frame)

    writer.release()


def test_video_analyzer_pipeline_outputs_are_readable(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic_motion.mp4"
    output_dir = tmp_path / "output"
    create_synthetic_motion_video(video_path)

    result = VideoAnalyzer(
        AnalysisConfig(
            input_path=video_path,
            output_dir=output_dir,
            min_motion_area=20,
            background_var_threshold=16,
            warmup_frames=8,
            progress_interval=0,
            write_motion_mask=True,
        )
    ).analyze()

    assert result.annotated_video_path.exists()
    assert result.heatmap_path.exists()
    assert result.metrics_path.exists()
    assert result.motion_mask_video_path is not None
    assert result.motion_mask_video_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    analysis = metrics["analysis"]
    assert analysis["processedFrames"] > 0
    assert analysis["warmupFrames"] == 8
    assert analysis["scoredFrames"] > 0
    assert analysis["maxActivityScore"] > 0
    assert analysis["maxActiveRegions"] > 0

    analyzed_capture = cv2.VideoCapture(str(result.annotated_video_path))
    assert analyzed_capture.isOpened()
    assert int(analyzed_capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    ok, frame = analyzed_capture.read()
    analyzed_capture.release()
    assert ok
    assert frame is not None

    motion_mask_capture = cv2.VideoCapture(str(result.motion_mask_video_path))
    assert motion_mask_capture.isOpened()
    ok, mask_frame = motion_mask_capture.read()
    motion_mask_capture.release()
    assert ok
    assert mask_frame is not None

    heatmap = cv2.imread(str(result.heatmap_path))
    assert heatmap is not None
