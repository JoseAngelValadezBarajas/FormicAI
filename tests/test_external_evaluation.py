from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from formicai.detection.evaluate import ExternalEvaluationConfig, ExternalEvaluator


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def write_external_test_record(root: Path, video: str, stem: str) -> None:
    write_image(root / video / "images" / f"{stem}.jpg")
    label_path = root / video / "labels" / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")


def test_external_evaluator_writes_metrics_by_video(tmp_path: Path, monkeypatch) -> None:
    external_test_dir = tmp_path / "external_test"
    write_external_test_record(external_test_dir, "ant3", "ant3_frame_000000_t0000.000")
    write_external_test_record(external_test_dir, "ant4", "ant4_frame_000000_t0000.000")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    output_path = tmp_path / "metrics.json"
    calls: list[dict[str, object]] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(model_path)

        def val(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.6, map=0.5),
                results_dict={},
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    assert not (external_test_dir / "ant3" / "dataset.yaml").exists()
    assert not (external_test_dir / "ant4" / "dataset.yaml").exists()

    result = ExternalEvaluator(
        ExternalEvaluationConfig(
            external_test_dir=external_test_dir,
            model_path=model_path,
            output_path=output_path,
            device="0",
            end2end=False,
        )
    ).evaluate()

    assert len(result.videos) == 2
    assert [video.video for video in result.videos] == ["ant3", "ant4"]
    assert len(calls) == 2
    assert calls[0]["split"] == "val"
    assert calls[0]["conf"] == 0.25
    assert calls[0]["iou"] == 0.70
    assert calls[0]["end2end"] is False
    assert not (external_test_dir / "ant3" / "dataset.yaml").exists()
    assert not (external_test_dir / "ant4" / "dataset.yaml").exists()
    generated_yaml = output_path.parent / "external_test_dataset_yamls" / "ant3.yaml"
    assert generated_yaml.exists()
    assert calls[0]["data"] == str(generated_yaml)
    generated_yaml_text = generated_yaml.read_text(encoding="utf-8")
    assert f"path: {(external_test_dir / 'ant3').resolve().as_posix()}" in generated_yaml_text
    assert "train: images" in generated_yaml_text
    assert "val: images" in generated_yaml_text

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["macroAverage"]["videos"] == 2
    assert data["macroAverage"]["precision"] == 0.8
    assert data["videos"][0]["video"] == "ant3"


def test_external_evaluator_rejects_direct_output_inside_external_test_dir(tmp_path: Path, monkeypatch) -> None:
    external_test_dir = tmp_path / "external_test"
    write_external_test_record(external_test_dir, "ant3", "ant3_frame_000000_t0000.000")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    output_path = external_test_dir / "metrics.json"
    original_label = external_test_dir / "ant3" / "labels" / "ant3_frame_000000_t0000.000.txt"
    original_label_bytes = original_label.read_bytes()

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            raise AssertionError("YOLO must not be constructed for unsafe output paths")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(ValueError, match="outside the external test directory"):
        ExternalEvaluator(
            ExternalEvaluationConfig(
                external_test_dir=external_test_dir,
                model_path=model_path,
                output_path=output_path,
            )
        ).evaluate()

    assert not output_path.exists()
    assert not (external_test_dir / "external_test_dataset_yamls").exists()
    assert not (external_test_dir / "external_test_val_runs").exists()
    assert not (external_test_dir / "ant3" / "dataset.yaml").exists()
    assert original_label.read_bytes() == original_label_bytes


def test_external_evaluator_rejects_nested_output_inside_external_test_dir(tmp_path: Path, monkeypatch) -> None:
    external_test_dir = tmp_path / "external_test"
    write_external_test_record(external_test_dir, "ant3", "ant3_frame_000000_t0000.000")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    output_path = external_test_dir / "some" / "nested" / "metrics.json"

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            raise AssertionError("YOLO must not be constructed for unsafe output paths")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(ValueError, match="outside the external test directory"):
        ExternalEvaluator(
            ExternalEvaluationConfig(
                external_test_dir=external_test_dir,
                model_path=model_path,
                output_path=output_path,
            )
        ).evaluate()

    assert not (external_test_dir / "some").exists()
    assert not (external_test_dir / "external_test_dataset_yamls").exists()
    assert not (external_test_dir / "external_test_val_runs").exists()


def test_external_evaluator_allows_safe_sibling_output_path(tmp_path: Path, monkeypatch) -> None:
    external_test_dir = tmp_path / "external_test"
    write_external_test_record(external_test_dir, "ant3", "ant3_frame_000000_t0000.000")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    output_path = tmp_path / "artifacts" / "metrics.json"
    calls: list[dict[str, object]] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(model_path)

        def val(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.6, map=0.5),
                results_dict={},
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    result = ExternalEvaluator(
        ExternalEvaluationConfig(
            external_test_dir=external_test_dir,
            model_path=model_path,
            output_path=output_path,
        )
    ).evaluate()

    assert len(result.videos) == 1
    assert output_path.exists()
    generated_yaml = output_path.parent / "external_test_dataset_yamls" / "ant3.yaml"
    assert generated_yaml.exists()
    assert calls[0]["data"] == str(generated_yaml)
    assert not (external_test_dir / "ant3" / "dataset.yaml").exists()


def test_external_evaluator_does_not_touch_existing_dataset_yaml(tmp_path: Path, monkeypatch) -> None:
    external_test_dir = tmp_path / "external_test"
    write_external_test_record(external_test_dir, "ant3", "ant3_frame_000000_t0000.000")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    output_path = tmp_path / "metrics.json"
    sentinel = external_test_dir / "ant3" / "dataset.yaml"
    sentinel.write_text("sentinel: keep me exactly\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(model_path)

        def val(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.6, map=0.5),
                results_dict={},
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    ExternalEvaluator(
        ExternalEvaluationConfig(
            external_test_dir=external_test_dir,
            model_path=model_path,
            output_path=output_path,
        )
    ).evaluate()

    assert sentinel.read_text(encoding="utf-8") == "sentinel: keep me exactly\n"
    generated_yaml = output_path.parent / "external_test_dataset_yamls" / "ant3.yaml"
    assert generated_yaml.exists()
    assert calls[0]["data"] == str(generated_yaml)
    assert calls[0]["data"] != str(sentinel)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["videos"][0]["datasetYaml"] == str(generated_yaml)
    assert data["videos"][0]["precision"] == 0.8
