from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from formicai.cli import build_detect_video_parser, resolve_detect_video_model
from formicai.detection.spec import CURRENT_CHAMPION, validate_model_classes
from formicai.detection.video import VideoDetectionConfig, VideoDetector
from formicai.utils.hashing import sha256_file


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _BoxCoordinates:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __getitem__(self, index: int) -> "_BoxCoordinates":
        assert index == 0
        return self

    def tolist(self) -> list[float]:
        return self._values


class _FakeBox:
    def __init__(self) -> None:
        self.cls = _Scalar(0)
        self.conf = _Scalar(0.91)
        self.xyxy = _BoxCoordinates([1.0, 2.0, 5.0, 7.0])


class _FakeCapture:
    def __init__(self, path: str) -> None:
        self.path = path
        self.released = False
        self._frames = [np.zeros((12, 20, 3), dtype=np.uint8)]

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 20
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 12
        if prop == cv2.CAP_PROP_FPS:
            return 10
        return 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class _FakeWriter:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.released = False
        self.frames_written = 0

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames_written += 1

    def release(self) -> None:
        self.released = True


def _install_fake_video_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)


def _install_fake_yolo(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, object]]) -> None:
    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            calls.append({"event": "init", "model_path": model_path})
            self.names = {0: "ant"}

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            calls.append({"event": "predict", **kwargs})
            return [SimpleNamespace(names={0: "ant"}, boxes=[_FakeBox()])]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO, __version__="test-ultra"))


def _assert_output_collision_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_video_path: Path | None = None,
    output_jsonl_path: Path | None = None,
    output_metadata_path: Path | None = None,
) -> None:
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "custom.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")
    writer_calls: list[object] = []

    class FailingYOLO:
        def __init__(self, model_path: str) -> None:
            raise AssertionError("YOLO must not be instantiated for colliding output paths")

    def failing_writer(*args: object, **kwargs: object) -> _FakeWriter:
        writer_calls.append(args)
        raise AssertionError("VideoWriter must not be created for colliding output paths")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FailingYOLO, __version__="test-ultra"))
    monkeypatch.setattr(cv2, "VideoWriter", failing_writer)

    with pytest.raises(ValueError, match="Output paths must be distinct"):
        VideoDetector(
            VideoDetectionConfig(
                input_path=video_path,
                model_path=model_path,
                output_video_path=output_video_path or (tmp_path / "detections.mp4"),
                output_jsonl_path=output_jsonl_path or (tmp_path / "detections.jsonl"),
                output_metadata_path=output_metadata_path,
            )
        ).detect()

    for path in {output_video_path, output_jsonl_path, output_metadata_path}:
        if path is not None:
            assert not path.resolve().exists()
    assert writer_calls == []


def test_current_champion_spec_is_canonical() -> None:
    assert CURRENT_CHAMPION.experiment == "ants_v3_mixedscale_e01"
    assert CURRENT_CHAMPION.sha256 == "424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf"
    assert CURRENT_CHAMPION.confidence == 0.25
    assert CURRENT_CHAMPION.iou == 0.70
    assert CURRENT_CHAMPION.image_size == 640
    assert CURRENT_CHAMPION.end2end is False
    assert CURRENT_CHAMPION.classes == {0: "ant"}


def test_sha256_file_streams_known_bytes(tmp_path: Path) -> None:
    payload = b"formicai audit-02\n"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == "6894eecce8e23c02aadb01f892d15473501df8075b97e8089242867b0402c919"


def test_detect_video_model_options_are_mutually_exclusive() -> None:
    parser = build_detect_video_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["samples/ant2.mp4", "--champion", "--model", "custom.pt"])


def test_champion_mode_resolves_expected_path() -> None:
    parser = build_detect_video_parser()
    args = parser.parse_args(["samples/ant2.mp4", "--champion"])

    assert resolve_detect_video_model(args) == CURRENT_CHAMPION.model_path


def test_champion_sha_mismatch_fails_before_yolo_is_instantiated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "selected.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"wrong model bytes")

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            raise AssertionError("YOLO must not be instantiated after champion SHA mismatch")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO, __version__="test-ultra"))

    with pytest.raises(ValueError, match="Champion model integrity check failed"):
        VideoDetector(
            VideoDetectionConfig(
                input_path=video_path,
                model_path=model_path,
                expected_model_sha256=CURRENT_CHAMPION.sha256,
            )
        ).detect()


def test_jsonl_metadata_output_path_collision_is_rejected_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_path = tmp_path / "same.json"

    _assert_output_collision_rejected(
        tmp_path,
        monkeypatch,
        output_jsonl_path=same_path,
        output_metadata_path=same_path,
    )


def test_video_metadata_output_path_collision_is_rejected_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_path = tmp_path / "same.out"

    _assert_output_collision_rejected(
        tmp_path,
        monkeypatch,
        output_video_path=same_path,
        output_metadata_path=same_path,
    )


def test_video_jsonl_output_path_collision_is_rejected_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_path = tmp_path / "same.out"

    _assert_output_collision_rejected(
        tmp_path,
        monkeypatch,
        output_video_path=same_path,
        output_jsonl_path=same_path,
    )


def test_normalized_equivalent_output_path_collision_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _assert_output_collision_rejected(
        tmp_path,
        monkeypatch,
        output_jsonl_path=output_dir / "file.json",
        output_metadata_path=output_dir / "." / "file.json",
    )


def test_detect_video_parser_defaults_are_canonical() -> None:
    parser = build_detect_video_parser()
    args = parser.parse_args(["samples/ant2.mp4", "--champion"])

    assert args.conf == CURRENT_CHAMPION.confidence
    assert args.iou == CURRENT_CHAMPION.iou
    assert args.imgsz == CURRENT_CHAMPION.image_size
    assert args.end2end is CURRENT_CHAMPION.end2end


def test_model_class_map_exact_ant_passes() -> None:
    assert validate_model_classes({0: "Ant"}, {0: "ant"}) == {0: "ant"}


def test_model_class_map_wrong_semantic_fails() -> None:
    with pytest.raises(ValueError, match="Unexpected detector class name"):
        validate_model_classes({0: "person"}, {0: "ant"})


def test_model_class_map_multiple_classes_fail() -> None:
    with pytest.raises(ValueError, match="Unexpected detector class ids"):
        validate_model_classes({0: "ant", 1: "queen"}, {0: "ant"})


def test_custom_model_sha_is_recorded_in_run_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_io(monkeypatch)
    calls: list[dict[str, object]] = []
    _install_fake_yolo(monkeypatch, calls)
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "custom.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"custom model")
    metadata_path = tmp_path / "detections.run.json"

    VideoDetector(
        VideoDetectionConfig(
            input_path=video_path,
            model_path=model_path,
            output_video_path=tmp_path / "detections.mp4",
            output_jsonl_path=tmp_path / "detections.jsonl",
            output_metadata_path=metadata_path,
        )
    ).detect()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"]["sha256"] == sha256_file(video_path)
    assert metadata["model"]["sha256"] == sha256_file(model_path)
    assert metadata["model"]["experiment"] == "CUSTOM / UNREGISTERED"
    assert metadata["model"]["isCurrentChampion"] is False


def test_known_champion_sha_from_different_path_is_identified_by_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_io(monkeypatch)
    calls: list[dict[str, object]] = []
    _install_fake_yolo(monkeypatch, calls)
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "renamed.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"arbitrary bytes")
    metadata_path = tmp_path / "detections.run.json"

    def fake_sha256_file(path: Path) -> str:
        if path == model_path:
            return CURRENT_CHAMPION.sha256
        return "0" * 64

    monkeypatch.setattr("formicai.detection.video.sha256_file", fake_sha256_file)

    VideoDetector(
        VideoDetectionConfig(
            input_path=video_path,
            model_path=model_path,
            output_video_path=tmp_path / "detections.mp4",
            output_jsonl_path=tmp_path / "detections.jsonl",
            output_metadata_path=metadata_path,
        )
    ).detect()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model"]["experiment"] == CURRENT_CHAMPION.experiment
    assert metadata["model"]["status"] == CURRENT_CHAMPION.status
    assert metadata["model"]["isCurrentChampion"] is True


def test_run_metadata_records_detection_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_io(monkeypatch)
    calls: list[dict[str, object]] = []
    _install_fake_yolo(monkeypatch, calls)
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "custom.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"custom model")
    jsonl_path = tmp_path / "detections.jsonl"

    result = VideoDetector(
        VideoDetectionConfig(
            input_path=video_path,
            model_path=model_path,
            output_video_path=tmp_path / "detections.mp4",
            output_jsonl_path=jsonl_path,
        )
    ).detect()

    metadata_path = tmp_path / "detections.run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert result.output_metadata_path == metadata_path
    assert metadata["schemaVersion"] == 1
    assert metadata["source"]["sha256"] == sha256_file(video_path)
    assert metadata["model"]["sha256"] == sha256_file(model_path)
    assert metadata["inference"]["confidence"] == CURRENT_CHAMPION.confidence
    assert metadata["inference"]["iou"] == CURRENT_CHAMPION.iou
    assert metadata["inference"]["imageSize"] == CURRENT_CHAMPION.image_size
    assert metadata["inference"]["end2end"] is CURRENT_CHAMPION.end2end
    assert metadata["inference"]["canonicalInference"] is True
    assert metadata["inference"]["overrides"] == {}
    assert metadata["classes"] == {"0": "ant"}
    assert metadata["environment"]["python"]
    assert metadata["environment"]["ultralytics"] == "test-ultra"
    assert metadata["environment"]["opencv"] == cv2.__version__
    assert metadata["result"]["processedFrames"] == 1
    assert metadata["result"]["totalDetections"] == 1
    assert metadata["result"]["outputJsonl"] == str(jsonl_path)


def test_distinct_default_output_paths_continue_working(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_io(monkeypatch)
    calls: list[dict[str, object]] = []
    _install_fake_yolo(monkeypatch, calls)
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "custom.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"custom model")

    result = VideoDetector(
        VideoDetectionConfig(
            input_path=video_path,
            model_path=model_path,
            output_video_path=tmp_path / "detections.mp4",
            output_jsonl_path=tmp_path / "detections.jsonl",
        )
    ).detect()

    assert result.output_jsonl_path == tmp_path / "detections.jsonl"
    assert result.output_metadata_path == tmp_path / "detections.run.json"
    assert result.output_jsonl_path.exists()
    assert result.output_metadata_path.exists()


def test_inference_override_marks_metadata_noncanonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_io(monkeypatch)
    calls: list[dict[str, object]] = []
    _install_fake_yolo(monkeypatch, calls)
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "custom.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"custom model")
    metadata_path = tmp_path / "detections.run.json"

    VideoDetector(
        VideoDetectionConfig(
            input_path=video_path,
            model_path=model_path,
            output_video_path=tmp_path / "detections.mp4",
            output_jsonl_path=tmp_path / "detections.jsonl",
            output_metadata_path=metadata_path,
            confidence=0.35,
        )
    ).detect()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["inference"]["canonicalInference"] is False
    assert metadata["inference"]["overrides"] == {
        "confidence": {"canonical": CURRENT_CHAMPION.confidence, "actual": 0.35}
    }


def test_video_detector_releases_capture_and_writer_after_prediction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[_FakeCapture] = []
    writers: list[_FakeWriter] = []

    class FailingCapture(_FakeCapture):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            captures.append(self)

    class FailingWriter(_FakeWriter):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            writers.append(self)

    class FailingYOLO:
        def __init__(self, model_path: str) -> None:
            self.names = {0: "ant"}

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            raise RuntimeError("prediction failed")

    monkeypatch.setattr(cv2, "VideoCapture", FailingCapture)
    monkeypatch.setattr(cv2, "VideoWriter", FailingWriter)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FailingYOLO, __version__="test-ultra"))
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "custom.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"custom model")

    with pytest.raises(RuntimeError, match="prediction failed"):
        VideoDetector(
            VideoDetectionConfig(
                input_path=video_path,
                model_path=model_path,
                output_video_path=tmp_path / "detections.mp4",
                output_jsonl_path=tmp_path / "detections.jsonl",
            )
        ).detect()

    assert captures
    assert writers
    assert captures[0].released is True
    assert writers[0].released is True
    assert not (tmp_path / "detections.run.json").exists()
