from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from formicai.dataset.extract_frames import FrameExtractionConfig, FrameExtractor
from formicai.utils.video import ROI


def create_test_video(path: Path, frames: int = 30, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (160, 120),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create test video.")
    for frame_index in range(frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + frame_index, 60), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_frame_extractor_writes_deterministic_images_and_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    output_dir = tmp_path / "raw"
    create_test_video(video_path)

    result = FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            every_seconds=0.5,
            roi=ROI(x=10, y=20, width=80, height=60),
        )
    ).extract()

    images = sorted(output_dir.glob("*.jpg"))
    assert len(images) == 6
    assert images[0].name == "source_frame_000000_t0000.000.jpg"
    assert result.extracted_frames == 6
    assert result.metadata_path.exists()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["frames"]) == 6
    assert metadata["frames"][0]["frameIndex"] == 0
    assert metadata["frames"][0]["roi"] == {
        "x": 10,
        "y": 20,
        "width": 80,
        "height": 60,
    }

    sample = cv2.imread(str(images[0]))
    assert sample is not None
    assert sample.shape[:2] == (60, 80)


def test_frame_extractor_every_frames_overrides_seconds(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    output_dir = tmp_path / "raw"
    create_test_video(video_path, frames=12, fps=12.0)

    result = FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            every_seconds=0.25,
            every_frames=4,
        )
    ).extract()

    assert result.extracted_frames == 3


def test_frame_extractor_count_selects_temporally_distributed_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    output_dir = tmp_path / "external_test"
    create_test_video(video_path, frames=10, fps=10.0)

    result = FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            target_frame_count=4,
        )
    ).extract()

    images = sorted(output_dir.glob("*.jpg"))
    assert result.extracted_frames == 4
    assert result.selected_frame_indexes == (0, 3, 6, 9)
    assert [image.name for image in images] == [
        "source_frame_000000_t0000.000.jpg",
        "source_frame_000003_t0000.300.jpg",
        "source_frame_000006_t0000.600.jpg",
        "source_frame_000009_t0000.900.jpg",
    ]

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["samplingStrategy"] == "fixed_count"
    assert metadata["requestedFrameCount"] == 4
    assert metadata["selectedFrameIndexes"] == [0, 3, 6, 9]


def test_frame_extractor_refuses_existing_extraction_outputs_without_overwrite(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    output_dir = tmp_path / "raw"
    create_test_video(video_path, frames=10, fps=10.0)
    first = FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            target_frame_count=10,
        )
    ).extract()
    image_bytes = {path.name: path.read_bytes() for path in sorted(output_dir.glob("*.jpg"))}
    metadata_before = first.metadata_path.read_bytes()

    try:
        FrameExtractor(
            FrameExtractionConfig(
                input_path=video_path,
                output_dir=output_dir,
                target_frame_count=4,
            )
        ).extract()
    except ValueError as exc:
        assert "Use --overwrite" in str(exc)
    else:
        raise AssertionError("Expected extraction into populated output_dir to fail.")

    assert {path.name: path.read_bytes() for path in sorted(output_dir.glob("*.jpg"))} == image_bytes
    assert first.metadata_path.read_bytes() == metadata_before


def test_frame_extractor_overwrite_removes_stale_generated_images(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    output_dir = tmp_path / "raw"
    create_test_video(video_path, frames=10, fps=10.0)
    FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            target_frame_count=10,
        )
    ).extract()

    result = FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            target_frame_count=4,
            overwrite=True,
        )
    ).extract()

    images = sorted(output_dir.glob("*.jpg"))
    assert result.extracted_frames == 4
    assert len(images) == 4
    assert [image.name for image in images] == [
        "source_frame_000000_t0000.000.jpg",
        "source_frame_000003_t0000.300.jpg",
        "source_frame_000006_t0000.600.jpg",
        "source_frame_000009_t0000.900.jpg",
    ]


def test_frame_extractor_overwrite_preserves_unrelated_files(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    output_dir = tmp_path / "raw"
    create_test_video(video_path, frames=10, fps=10.0)
    FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            target_frame_count=10,
        )
    ).extract()
    notes = output_dir / "notes.txt"
    notes.write_text("keep me\n", encoding="utf-8")

    FrameExtractor(
        FrameExtractionConfig(
            input_path=video_path,
            output_dir=output_dir,
            target_frame_count=4,
            overwrite=True,
        )
    ).extract()

    assert notes.read_text(encoding="utf-8") == "keep me\n"
