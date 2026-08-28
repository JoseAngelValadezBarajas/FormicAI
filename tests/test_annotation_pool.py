from __future__ import annotations

import pytest

from formicai.dataset.annotation_pool import (
    CandidatePoolIntegrityError,
    choose_largest_feasible_spacing,
    remediated_redundancy_decision,
    select_annotation_pool_candidates,
    select_maximum_cardinality_frames,
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


def test_select_maximum_cardinality_beats_greedy_order() -> None:
    result = select_maximum_cardinality_frames(
        [0, 4, 5, 9],
        target_count=2,
        spacing_frames=5,
    )

    assert result.selected_frame_indexes == [0, 9]
    assert result.source_shortfall is False


def test_select_maximum_cardinality_uses_deterministic_tie_breaks() -> None:
    first = select_maximum_cardinality_frames(
        [0, 10, 20, 30, 40],
        target_count=3,
        spacing_frames=10,
    )
    second = select_maximum_cardinality_frames(
        [40, 30, 20, 10, 0],
        target_count=3,
        spacing_frames=10,
    )

    assert first.selected_frame_indexes == [0, 20, 40]
    assert second.selected_frame_indexes == first.selected_frame_indexes


def test_choose_largest_feasible_spacing_retains_preferred_when_possible() -> None:
    result = choose_largest_feasible_spacing(
        [0, 60, 120],
        target_count=3,
        preferred_spacing_frames=60,
        minimum_allowed_spacing_frames=30,
    )

    assert result.spacing_frames == 60
    assert result.selected_frame_indexes == [0, 60, 120]
    assert result.source_shortfall is False


def test_choose_largest_feasible_spacing_adapts_to_largest_working_value() -> None:
    result = choose_largest_feasible_spacing(
        [0, 48, 96],
        target_count=3,
        preferred_spacing_frames=60,
        minimum_allowed_spacing_frames=30,
    )

    assert result.spacing_frames == 48
    assert result.selected_frame_indexes == [0, 48, 96]
    assert result.source_shortfall is False


def test_choose_largest_feasible_spacing_respects_one_second_floor_shortfall() -> None:
    result = choose_largest_feasible_spacing(
        [0, 10, 20],
        target_count=3,
        preferred_spacing_frames=60,
        minimum_allowed_spacing_frames=30,
    )

    assert result.spacing_frames == 30
    assert result.selected_frame_indexes == [0]
    assert result.source_shortfall is True


def test_remediated_redundancy_decision_requires_exact_or_both_hashes() -> None:
    assert remediated_redundancy_decision(
        exact_sha_duplicate=True,
        dhash_distance=64,
        phash_distance=64,
    ).reason == "EXACT_SHA_DUPLICATE"

    dhash_only = remediated_redundancy_decision(
        exact_sha_duplicate=False,
        dhash_distance=5,
        phash_distance=9,
    )
    assert dhash_only.hard_reject is False
    assert dhash_only.flag == "PERCEPTUAL_SIMILARITY_FLAG"

    phash_only = remediated_redundancy_decision(
        exact_sha_duplicate=False,
        dhash_distance=6,
        phash_distance=8,
    )
    assert phash_only.hard_reject is False
    assert phash_only.flag == "PERCEPTUAL_SIMILARITY_FLAG"

    both = remediated_redundancy_decision(
        exact_sha_duplicate=False,
        dhash_distance=5,
        phash_distance=8,
    )
    assert both.hard_reject is True
    assert both.reason == "COMBINED_PERCEPTUAL_DUPLICATE"
