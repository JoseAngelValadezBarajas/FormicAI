from __future__ import annotations

import importlib

from formicai.utils import environment


def test_environment_metadata_reports_core_runtime_versions() -> None:
    metadata = environment.collect_environment_metadata()

    assert metadata["python"]
    assert metadata["platform"]
    assert metadata["numpy"]
    assert metadata["opencv"]
    assert "ultralytics" in metadata
    assert "torch" in metadata
    assert "cudaAvailable" in metadata
    assert "torchCudaVersion" in metadata


def test_environment_metadata_treats_missing_optional_ml_packages_as_none(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(module_name: str):
        if module_name in {"torch", "ultralytics"}:
            raise RuntimeError(module_name)
        return real_import_module(module_name)

    monkeypatch.setattr(environment.importlib, "import_module", fake_import_module)

    metadata = environment.collect_environment_metadata()

    assert metadata["ultralytics"] is None
    assert metadata["torch"] is None
    assert metadata["cudaAvailable"] is None
    assert metadata["torchCudaVersion"] is None
