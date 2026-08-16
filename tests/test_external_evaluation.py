from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from formicai.detection.evaluate import ExternalEvaluationConfig, ExternalEvaluator


def write_image(path: Path, *, width: int = 32, height: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def write_external_test_record(
    root: Path,
    video: str,
    stem: str,
    *,
    label: str = "0 0.500000 0.500000 0.250000 0.250000\n",
    width: int = 32,
    height: int = 32,
) -> None:
    write_image(root / video / "images" / f"{stem}.jpg", width=width, height=height)
    label_path = root / video / "labels" / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label, encoding="utf-8")


def yolo_label_from_edges(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    return (
        f"0 {(x_min + x_max) / 2:.12g} {(y_min + y_max) / 2:.12g} "
        f"{x_max - x_min:.12g} {y_max - y_min:.12g}\n"
    )


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


def test_external_evaluator_accepts_half_pixel_external_label_rounding(tmp_path: Path, monkeypatch) -> None:
    external_test_dir = tmp_path / "external_test"
    label = yolo_label_from_edges(0.2, 0.2, 1.0 + 0.49 / 400, 0.8)
    write_external_test_record(
        external_test_dir,
        "ant4",
        "ant4_frame_000016_t0000.533",
        label=label,
        width=400,
        height=472,
    )
    label_path = external_test_dir / "ant4" / "labels" / "ant4_frame_000016_t0000.533.txt"
    image_path = external_test_dir / "ant4" / "images" / "ant4_frame_000016_t0000.533.jpg"
    original_label_bytes = label_path.read_bytes()
    original_image_bytes = image_path.read_bytes()
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

    result = ExternalEvaluator(
        ExternalEvaluationConfig(
            external_test_dir=external_test_dir,
            model_path=model_path,
            output_path=output_path,
        )
    ).evaluate()

    assert result.videos[0].annotations == 1
    assert len(calls) == 1
    assert label_path.read_bytes() == original_label_bytes
    assert image_path.read_bytes() == original_image_bytes


def test_external_evaluator_rejects_external_label_rounding_over_half_pixel_before_yolo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_test_dir = tmp_path / "external_test"
    label = yolo_label_from_edges(0.2, 0.2, 1.0 + 0.51 / 400, 0.8)
    write_external_test_record(
        external_test_dir,
        "ant4",
        "ant4_frame_000016_t0000.533",
        label=label,
        width=400,
        height=472,
    )
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            raise AssertionError("YOLO must not be constructed when external labels fail validation")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(ValueError, match="more than 0.5 pixels"):
        ExternalEvaluator(
            ExternalEvaluationConfig(
                external_test_dir=external_test_dir,
                model_path=model_path,
                output_path=tmp_path / "metrics.json",
            )
        ).evaluate()

    assert not (tmp_path / "metrics.json").exists()


def test_external_evaluator_rejects_unreadable_external_test_image_before_yolo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_test_dir = tmp_path / "external_test"
    image_path = external_test_dir / "ant4" / "images" / "ant4_frame_000016_t0000.533.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_text("not an image", encoding="utf-8")
    label_path = external_test_dir / "ant4" / "labels" / "ant4_frame_000016_t0000.533.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            raise AssertionError("YOLO must not be constructed when image decoding fails")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(ValueError, match="Could not read image dimensions"):
        ExternalEvaluator(
            ExternalEvaluationConfig(
                external_test_dir=external_test_dir,
                model_path=model_path,
                output_path=tmp_path / "metrics.json",
            )
        ).evaluate()
