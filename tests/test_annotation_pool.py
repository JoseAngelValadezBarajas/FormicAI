from __future__ import annotations

import pytest

from formicai.dataset.annotation_pool import (
    CandidatePoolIntegrityError,
    select_annotation_pool_candidates,
    summarize_adjacent_redundancy,
    verify_candidate_pool_document,
)


def candidate(index: int, *, sha: str | None = None, model_inference: bool = False) -> dict:
    return {
        "candidateIndex": index,
        "frameIndex": index * 10,
        "height": 1080,
        "imageFilename": f"candidate_{index:03d}.jpg",
        "imageSha256": sha or f"{index:064x}",
        "manualStrataStatus": "requires_manual_annotation_for_exact_density_scale_crowding",
        "modelInferenceUsed": model_inference,
        "temporalBin": index,
        "timestampSeconds": float(index),
        "width": 1920,
    }


def test_verify_candidate_pool_rejects_model_derived_metadata() -> None:
    document = {
        "candidateCounts": {
            "requested": 2,
            "decodedOk": 2,
            "uniqueImageSha256": 2,
        },
        "extractionPolicy": {
            "algorithm": "Divide full video frame interval into 120 equal bins and extract the center frame from each bin. Uses timing/frame count only.",
            "usesFinalTestContent": False,
            "usesModelInference": False,
            "usesPseudoLabeling": False,
        },
        "source": {"sha256": "source-sha"},
        "candidates": [
            candidate(0),
            candidate(1, model_inference=True),
        ],
    }

    with pytest.raises(CandidatePoolIntegrityError, match="model-derived"):
        verify_candidate_pool_document(
            document,
            expected_source_sha256="source-sha",
            expected_candidate_count=2,
        )


def test_select_annotation_pool_is_deterministic_and_region_balanced() -> None:
    candidates = [candidate(index) for index in range(12)]
    dhashes = {
        record["imageFilename"]: hash_value
        for record, hash_value in zip(
            candidates,
            [
                0b00000000,
                0b00000001,
                0b11110000,
                0b11110001,
                0b00001111,
                0b00001110,
                0b10101010,
                0b10101011,
                0b00110011,
                0b11001100,
                0b00111100,
                0b11000011,
            ],
            strict=True,
        )
    }

    selected = select_annotation_pool_candidates(
        candidates,
        dhashes,
        region_count=3,
        target_count=6,
        near_duplicate_threshold=1,
        min_temporal_spacing_seconds=1.0,
    )

    assert [record["imageFilename"] for record in selected.selected] == [
        "candidate_000.jpg",
        "candidate_002.jpg",
        "candidate_004.jpg",
        "candidate_006.jpg",
        "candidate_008.jpg",
        "candidate_009.jpg",
    ]
    assert selected.region_counts == {"region_00": 2, "region_01": 2, "region_02": 2}
    assert selected.algorithm["targetCount"] == 6


def test_select_annotation_pool_fills_by_temporal_spread_when_similarity_is_redundant() -> None:
    candidates = [candidate(index) for index in range(6)]
    dhashes = {record["imageFilename"]: 0 for record in candidates}

    selected = select_annotation_pool_candidates(
        candidates,
        dhashes,
        region_count=1,
        target_count=3,
        near_duplicate_threshold=4,
        min_temporal_spacing_seconds=10.0,
    )

    assert [record["candidateIndex"] for record in selected.selected] == [0, 2, 5]
    assert {
        record["candidateIndex"]: record["regionSelectionRank"]
        for record in selected.selected
    } == {
        0: 0,
        5: 1,
        2: 2,
    }
    assert {record["selectionStage"] for record in selected.selected} == {
        "strict_temporal_and_perceptual",
        "temporal_spread_fill",
    }


def test_summarize_adjacent_redundancy_reports_runs_and_distances() -> None:
    candidates = [candidate(index) for index in range(5)]
    dhashes = {
        "candidate_000.jpg": 0,
        "candidate_001.jpg": 1,
        "candidate_002.jpg": 3,
        "candidate_003.jpg": 255,
        "candidate_004.jpg": 254,
    }

    summary = summarize_adjacent_redundancy(
        candidates,
        dhashes,
        near_duplicate_threshold=2,
    )

    assert summary["adjacentDhashDistanceSummary"]["count"] == 4
    assert summary["nearDuplicateAdjacentPairCount"] == 3
    assert summary["effectivelyDistinctVisualStates"] == 2
    assert summary["clusters"] == [
        {
            "clusterId": "cluster_000",
            "startCandidateIndex": 0,
            "endCandidateIndex": 2,
            "startTimestampSeconds": 0.0,
            "endTimestampSeconds": 2.0,
            "candidateCount": 3,
        },
        {
            "clusterId": "cluster_001",
            "startCandidateIndex": 3,
            "endCandidateIndex": 4,
            "startTimestampSeconds": 3.0,
            "endTimestampSeconds": 4.0,
            "candidateCount": 2,
        },
    ]
