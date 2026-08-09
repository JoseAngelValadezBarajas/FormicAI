from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

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
    assert (external_test_dir / "ant3" / "dataset.yaml").exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["macroAverage"]["videos"] == 2
    assert data["macroAverage"]["precision"] == 0.8
    assert data["videos"][0]["video"] == "ant3"
