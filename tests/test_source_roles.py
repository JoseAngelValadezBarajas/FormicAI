from __future__ import annotations

from pathlib import Path

import pytest

from formicai.dataset.source_roles import (
    ForbiddenSourceError,
    assert_training_source_audit_allowed,
    inspect_training_source_file,
)


def make_registry() -> dict:
    return {
        "forbiddenForTrainingSourceAudit": {
            "identifiers": ["v5_final_test", "videonuevoants.mp4"],
            "paths": [
                "datasets/external_test/v5_final_test",
                "samples/v2/videonuevoants.mp4",
            ],
        },
        "roles": {
            "FINAL_TEST": {
                "sources": [
                    {
                        "sourceId": "v5_final_test",
                        "filename": "videonuevoants.mp4",
                        "path": "samples/v2/videonuevoants.mp4",
                        "pathAliases": ["datasets/external_test/v5_final_test"],
                        "trainEligible": False,
                        "forbiddenForDesign": True,
                    }
                ]
            },
            "TRAINING_SOURCE": {
                "sources": [
                    {
                        "sourceId": "training_source_001",
                        "filename": "ants_v5_training_source_001.mp4",
                        "path": "samples/v2/ants_v5_training_source_001.mp4",
                        "trainEligible": True,
                    }
                ]
            },
        },
    }


class FakeStat:
    st_size = 12345


def test_forbidden_final_test_video_is_rejected_before_filesystem_inspection() -> None:
    calls: list[str] = []

    def forbidden_callback(name: str):
        def callback(path: Path):
            calls.append(name)
            raise AssertionError(f"{name} should not inspect {path}")

        return callback

    with pytest.raises(ForbiddenSourceError):
        inspect_training_source_file(
            Path("samples/v2/videonuevoants.mp4"),
            make_registry(),
            stat_func=forbidden_callback("stat"),
            sha256_func=forbidden_callback("hash"),
            video_probe_func=forbidden_callback("probe"),
        )

    assert calls == []


def test_forbidden_final_test_dataset_child_path_is_rejected() -> None:
    with pytest.raises(ForbiddenSourceError):
        assert_training_source_audit_allowed(
            Path("datasets/external_test/v5_final_test/images/frame.jpg"),
            make_registry(),
        )


def test_training_source_remains_inspectable_after_guard_passes() -> None:
    calls: list[str] = []

    def fake_stat(path: Path) -> FakeStat:
        calls.append(f"stat:{path.as_posix()}")
        return FakeStat()

    def fake_sha256(path: Path) -> str:
        calls.append(f"hash:{path.as_posix()}")
        return "a" * 64

    def fake_probe(path: Path) -> dict[str, float]:
        calls.append(f"probe:{path.as_posix()}")
        return {"durationSeconds": 60.0}

    result = inspect_training_source_file(
        Path("samples/v2/ants_v5_training_source_001.mp4"),
        make_registry(),
        stat_func=fake_stat,
        sha256_func=fake_sha256,
        video_probe_func=fake_probe,
    )

    assert result.file_size_bytes == 12345
    assert result.sha256 == "a" * 64
    assert result.video_metadata == {"durationSeconds": 60.0}
    assert calls == [
        "stat:samples/v2/ants_v5_training_source_001.mp4",
        "probe:samples/v2/ants_v5_training_source_001.mp4",
        "hash:samples/v2/ants_v5_training_source_001.mp4",
    ]
