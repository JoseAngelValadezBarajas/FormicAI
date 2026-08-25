from __future__ import annotations

from pathlib import Path

import pytest

from formicai.dataset.source_roles import (
    ForbiddenSourceError,
    assert_training_source_audit_allowed,
    classify_v5_development_path,
    assert_v5_development_path_allowed,
    filter_v5_development_paths,
    inspect_training_source_file,
    load_source_role_registry,
)


V5_SOURCE_ROLE_REGISTRY = Path("experiments/ants_v5/source_role_registry.json")


def make_registry() -> dict:
    return {
        "forbiddenForTrainingSourceAudit": {
            "identifiers": ["v5_final_test", "videonuevoants.mp4"],
            "paths": [
                "datasets/external_test/v5_final_test",
                "samples/v2/videonuevoants.mp4",
            ],
            "sealedDetailedMetadataPaths": [
                "experiments/ants_v5/final_test_ground_truth_manifest.json",
                "experiments/ants_v5/final_test_source_frame_manifest.json",
                "datasets/external_test/v5_final_test/metadata.json",
            ],
        },
        "roles": {
            "FINAL_TEST": {
                "sources": [],
            },
            "OBSERVED_REFERENCE": {
                "sources": [
                    {
                        "sourceId": "former_v5_final_test_001",
                        "formerSourceId": "v5_final_test",
                        "filename": "videonuevoants.mp4",
                        "path": "samples/v2/videonuevoants.mp4",
                        "pathAliases": ["datasets/external_test/v5_final_test"],
                        "trainEligible": False,
                        "developmentValidationEligible": False,
                        "checkpointSelectionEligible": False,
                        "tuningEligible": False,
                        "autopsyInputEligible": False,
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


def test_final_test_video_path_role_check_is_allowed_as_non_semantic_metadata() -> None:
    policy = classify_v5_development_path(
        Path("samples/v2/videonuevoants.mp4"),
        make_registry(),
    )

    assert policy.protected is True
    assert policy.semantic_access_allowed is False
    assert policy.non_semantic_metadata_allowed is True
    assert "samples/v2/videonuevoants.mp4" in policy.matched_tokens


def test_forbidden_final_test_dataset_child_path_is_rejected() -> None:
    with pytest.raises(ForbiddenSourceError):
        assert_training_source_audit_allowed(
            Path("datasets/external_test/v5_final_test/images/frame.jpg"),
            make_registry(),
        )


def test_final_test_image_and_label_semantic_paths_are_rejected() -> None:
    for path in [
        Path("datasets/external_test/v5_final_test/images/frame.jpg"),
        Path("datasets/external_test/v5_final_test/labels/frame.txt"),
    ]:
        with pytest.raises(ForbiddenSourceError):
            assert_v5_development_path_allowed(path, make_registry())


def test_sealed_final_test_metadata_path_is_rejected_before_file_read() -> None:
    with pytest.raises(ForbiddenSourceError):
        assert_v5_development_path_allowed(
            Path("experiments/ants_v5/final_test_ground_truth_manifest.json"),
            make_registry(),
        )


def test_annotation_pool_records_remain_allowed_for_v5_development() -> None:
    allowed = filter_v5_development_paths(
        [
            Path("experiments/ants_v5/training_source_001_annotation_pool_manifest.json"),
            Path("experiments/ants_v5/final_test_source_frame_manifest.json"),
        ],
        make_registry(),
    )

    assert allowed == [
        Path("experiments/ants_v5/training_source_001_annotation_pool_manifest.json")
    ]


def test_active_registry_retires_exposed_v5_final_test_role() -> None:
    registry = load_source_role_registry(V5_SOURCE_ROLE_REGISTRY)

    assert registry["roles"].get("FINAL_TEST", {}).get("sources", []) == []
    assert registry["scientificBoundary"]["v5FinalTestRole"] == (
        "RETIRED_OBSERVED_REFERENCE"
    )
    assert registry["scientificBoundary"]["newStrictFinalTestRequired"] is True


def test_active_registry_records_former_final_test_as_observed_reference() -> None:
    registry = load_source_role_registry(V5_SOURCE_ROLE_REGISTRY)
    observed_sources = registry["roles"]["OBSERVED_REFERENCE"]["sources"]

    former_source = next(
        source
        for source in observed_sources
        if isinstance(source, dict)
        and source.get("sourceId") == "former_v5_final_test_001"
    )

    assert former_source["formerSourceId"] == "v5_final_test"
    assert former_source["reason"] == "DEVELOPMENT_GT_METADATA_EXPOSURE"
    assert former_source["trainEligible"] is False
    assert former_source["developmentValidationEligible"] is False
    assert former_source["checkpointSelectionEligible"] is False
    assert former_source["tuningEligible"] is False
    assert former_source["autopsyInputEligible"] is False
    assert former_source["autopsyInputEligibilityCondition"] == (
        "only after v5 selected.pt is frozen"
    )


def test_active_registry_keeps_former_final_test_paths_semantically_blocked() -> None:
    registry = load_source_role_registry(V5_SOURCE_ROLE_REGISTRY)

    policy = classify_v5_development_path(
        Path("datasets/external_test/v5_final_test/images/frame.jpg"),
        registry,
    )

    assert policy.protected is True
    assert policy.semantic_access_allowed is False


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
