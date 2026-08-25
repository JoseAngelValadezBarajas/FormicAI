from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


class CandidatePoolIntegrityError(ValueError):
    """Raised when an annotation-pool candidate document is not trustworthy."""


@dataclass(frozen=True)
class SelectionResult:
    selected: list[dict[str, Any]]
    region_counts: dict[str, int]
    algorithm: dict[str, Any]


def verify_candidate_pool_document(
    document: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    expected_candidate_count: int,
) -> None:
    source = document.get("source", {})
    if source.get("sha256") != expected_source_sha256:
        raise CandidatePoolIntegrityError("candidate pool source SHA256 mismatch")

    candidates = _sorted_candidates(document.get("candidates", []))
    if len(candidates) != expected_candidate_count:
        raise CandidatePoolIntegrityError(
            f"candidate count must be {expected_candidate_count}, got {len(candidates)}"
        )

    counts = document.get("candidateCounts", {})
    if counts.get("requested") != expected_candidate_count:
        raise CandidatePoolIntegrityError("requested candidate count mismatch")
    if counts.get("decodedOk") != expected_candidate_count:
        raise CandidatePoolIntegrityError("decoded candidate count mismatch")
    if counts.get("uniqueImageSha256") != expected_candidate_count:
        raise CandidatePoolIntegrityError("unique image SHA count mismatch")

    policy = document.get("extractionPolicy", {})
    if policy.get("usesFinalTestContent") is not False:
        raise CandidatePoolIntegrityError("candidate pool may use final-test content")
    if policy.get("usesModelInference") is not False:
        raise CandidatePoolIntegrityError("candidate pool may use model-derived metadata")
    if policy.get("usesPseudoLabeling") is not False:
        raise CandidatePoolIntegrityError("candidate pool may use pseudo-labeling")
    algorithm = str(policy.get("algorithm", ""))
    if "120 equal bins" not in algorithm or "center frame" not in algorithm:
        raise CandidatePoolIntegrityError("candidate extraction algorithm changed")

    filenames = [str(candidate.get("imageFilename")) for candidate in candidates]
    indexes = [int(candidate.get("candidateIndex")) for candidate in candidates]
    image_hashes = [str(candidate.get("imageSha256")) for candidate in candidates]
    if len(set(filenames)) != len(filenames):
        raise CandidatePoolIntegrityError("candidate filenames are not unique")
    if len(set(indexes)) != len(indexes):
        raise CandidatePoolIntegrityError("candidate indexes are not unique")
    if len(set(image_hashes)) != len(image_hashes):
        raise CandidatePoolIntegrityError("candidate image hashes are not unique")

    for candidate in candidates:
        if candidate.get("modelInferenceUsed") is not False:
            raise CandidatePoolIntegrityError(
                f"candidate {candidate.get('imageFilename')} contains model-derived metadata"
            )


def load_candidate_pool_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_candidate_dhashes(
    candidates: Sequence[Mapping[str, Any]],
    candidate_frame_dir: Path,
) -> dict[str, int]:
    hashes: dict[str, int] = {}
    for candidate in candidates:
        filename = _candidate_filename(candidate)
        hashes[filename] = dhash_file(candidate_frame_dir / filename)
    return hashes


def dhash_file(path: Path, *, hash_size: int = 8) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CandidatePoolIntegrityError(f"cannot read candidate image: {path}")
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def summarize_adjacent_redundancy(
    candidates: Sequence[Mapping[str, Any]],
    dhashes: Mapping[str, int],
    *,
    near_duplicate_threshold: int,
) -> dict[str, Any]:
    ordered = _sorted_candidates(candidates)
    distances: list[int] = []
    near_duplicate_pairs: list[dict[str, Any]] = []

    cluster_start = 0
    clusters: list[dict[str, Any]] = []
    for index in range(len(ordered) - 1):
        left = ordered[index]
        right = ordered[index + 1]
        distance = hamming_distance(
            dhashes[_candidate_filename(left)],
            dhashes[_candidate_filename(right)],
        )
        distances.append(distance)
        if distance <= near_duplicate_threshold:
            near_duplicate_pairs.append(
                {
                    "a": _candidate_filename(left),
                    "b": _candidate_filename(right),
                    "hamming": distance,
                }
            )
        else:
            clusters.append(_cluster_record(len(clusters), ordered[cluster_start : index + 1]))
            cluster_start = index + 1

    if ordered:
        clusters.append(_cluster_record(len(clusters), ordered[cluster_start:]))

    return {
        "adjacentDhashDistanceSummary": _numeric_summary(distances),
        "nearDuplicateAdjacentPairCount": len(near_duplicate_pairs),
        "nearDuplicateThreshold": near_duplicate_threshold,
        "effectivelyDistinctVisualStates": len(clusters),
        "clusters": clusters,
        "nearDuplicateAdjacentPairs": near_duplicate_pairs,
    }


def select_annotation_pool_candidates(
    candidates: Sequence[Mapping[str, Any]],
    dhashes: Mapping[str, int],
    *,
    region_count: int,
    target_count: int,
    near_duplicate_threshold: int,
    min_temporal_spacing_seconds: float,
) -> SelectionResult:
    if region_count <= 0:
        raise ValueError("region_count must be positive")
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if min_temporal_spacing_seconds < 0:
        raise ValueError("min_temporal_spacing_seconds must be non-negative")

    ordered = _sorted_candidates(candidates)
    if not ordered:
        return SelectionResult(
            selected=[],
            region_counts={},
            algorithm=_selection_algorithm(
                region_count,
                target_count,
                near_duplicate_threshold,
                min_temporal_spacing_seconds,
            ),
        )

    regions = _partition_temporal_regions(ordered, region_count)
    quotas = _region_quotas(target_count, len(regions))
    selected: list[dict[str, Any]] = []
    region_counts: dict[str, int] = {}

    for region_id, region_candidates in regions.items():
        quota = quotas[region_id]
        region_selected = _select_region_candidates(
            region_candidates,
            dhashes,
            quota=quota,
            near_duplicate_threshold=near_duplicate_threshold,
            min_temporal_spacing_seconds=min_temporal_spacing_seconds,
            region_id=region_id,
        )
        selected.extend(region_selected)
        region_counts[region_id] = len(region_selected)

    selected.sort(key=lambda record: (record["timestampSeconds"], record["candidateIndex"]))
    if len(selected) > target_count:
        selected = _downsample_by_temporal_coverage(selected, target_count)
        region_counts = _count_regions(selected)

    return SelectionResult(
        selected=selected,
        region_counts=region_counts,
        algorithm=_selection_algorithm(
            region_count,
            target_count,
            near_duplicate_threshold,
            min_temporal_spacing_seconds,
        ),
    )


def materialize_annotation_pool_images(
    selected: Sequence[Mapping[str, Any]],
    *,
    source_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_filenames = {_candidate_filename(record) for record in selected}
    for existing in output_dir.iterdir():
        if existing.is_file() and existing.name not in selected_filenames:
            existing.unlink()

    for record in selected:
        filename = _candidate_filename(record)
        shutil.copy2(source_dir / filename, output_dir / filename)


def combined_image_content_sha256(selected: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(selected, key=lambda item: _candidate_filename(item)):
        digest.update(_candidate_filename(record).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["imageSha256"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _select_region_candidates(
    candidates: Sequence[Mapping[str, Any]],
    dhashes: Mapping[str, int],
    *,
    quota: int,
    near_duplicate_threshold: int,
    min_temporal_spacing_seconds: float,
    region_id: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for candidate in candidates:
        if len(selected) >= quota:
            break
        if _passes_temporal_spacing(candidate, selected, min_temporal_spacing_seconds) and _passes_dhash_spacing(
            candidate,
            selected,
            dhashes,
            near_duplicate_threshold,
        ):
            selected.append(_selected_record(candidate, region_id, "strict_temporal_and_perceptual"))

    for candidate in candidates:
        if len(selected) >= quota:
            break
        if _candidate_filename(candidate) in {_candidate_filename(record) for record in selected}:
            continue
        if _passes_dhash_spacing(candidate, selected, dhashes, near_duplicate_threshold):
            selected.append(_selected_record(candidate, region_id, "relaxed_temporal_perceptual"))

    while len(selected) < quota and len(selected) < len(candidates):
        remaining = [
            candidate
            for candidate in candidates
            if _candidate_filename(candidate) not in {_candidate_filename(record) for record in selected}
        ]
        best = max(
            remaining,
            key=lambda candidate: (
                _minimum_temporal_distance(candidate, selected),
                -int(candidate["candidateIndex"]),
            ),
        )
        selected.append(_selected_record(best, region_id, "temporal_spread_fill"))

    for rank, record in enumerate(selected):
        record["regionSelectionRank"] = rank
    selected.sort(key=lambda record: (record["timestampSeconds"], record["candidateIndex"]))
    return selected


def _passes_temporal_spacing(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    min_temporal_spacing_seconds: float,
) -> bool:
    return all(
        abs(float(candidate["timestampSeconds"]) - float(record["timestampSeconds"]))
        >= min_temporal_spacing_seconds
        for record in selected
    )


def _passes_dhash_spacing(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    dhashes: Mapping[str, int],
    near_duplicate_threshold: int,
) -> bool:
    candidate_hash = dhashes[_candidate_filename(candidate)]
    return all(
        hamming_distance(candidate_hash, dhashes[_candidate_filename(record)])
        > near_duplicate_threshold
        for record in selected
    )


def _minimum_temporal_distance(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> float:
    if not selected:
        return math.inf
    timestamp = float(candidate["timestampSeconds"])
    return min(abs(timestamp - float(record["timestampSeconds"])) for record in selected)


def _partition_temporal_regions(
    candidates: Sequence[Mapping[str, Any]],
    region_count: int,
) -> dict[str, list[Mapping[str, Any]]]:
    timestamps = [float(candidate["timestampSeconds"]) for candidate in candidates]
    start = min(timestamps)
    end = max(timestamps)
    if end == start:
        return {"region_00": list(candidates)}

    regions: dict[str, list[Mapping[str, Any]]] = {
        f"region_{index:02d}": [] for index in range(region_count)
    }
    width = (end - start + 1e-9) / region_count
    for candidate in candidates:
        region_index = min(
            int((float(candidate["timestampSeconds"]) - start) / width),
            region_count - 1,
        )
        regions[f"region_{region_index:02d}"].append(candidate)

    return {key: value for key, value in regions.items() if value}


def _region_quotas(target_count: int, region_count: int) -> dict[str, int]:
    base = target_count // region_count
    remainder = target_count % region_count
    return {
        f"region_{index:02d}": base + (1 if index < remainder else 0)
        for index in range(region_count)
    }


def _downsample_by_temporal_coverage(
    candidates: Sequence[Mapping[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    ordered = list(candidates)
    while len(selected) < target_count:
        best = max(
            [candidate for candidate in ordered if candidate not in selected],
            key=lambda candidate: (
                _minimum_temporal_distance(candidate, selected),
                -int(candidate["candidateIndex"]),
            ),
        )
        selected.append(best)
    return [
        dict(record)
        for record in sorted(selected, key=lambda item: (item["timestampSeconds"], item["candidateIndex"]))
    ]


def _selected_record(
    candidate: Mapping[str, Any],
    region_id: str,
    selection_stage: str,
) -> dict[str, Any]:
    record = dict(candidate)
    record["regionId"] = region_id
    record["selectionStage"] = selection_stage
    return record


def _count_regions(selected: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in selected:
        region_id = str(record["regionId"])
        counts[region_id] = counts.get(region_id, 0) + 1
    return counts


def _selection_algorithm(
    region_count: int,
    target_count: int,
    near_duplicate_threshold: int,
    min_temporal_spacing_seconds: float,
) -> dict[str, Any]:
    return {
        "name": "deterministic_temporal_region_perceptual_pruning",
        "regionCount": region_count,
        "targetCount": target_count,
        "nearDuplicateRule": f"dHash Hamming distance <= {near_duplicate_threshold}",
        "nearDuplicateThreshold": near_duplicate_threshold,
        "minTemporalSpacingSeconds": min_temporal_spacing_seconds,
        "tieBreakers": [
            "earliest candidate in strict pass",
            "maximum minimum timestamp distance during fill",
            "lower candidateIndex",
        ],
        "usesModelInference": False,
        "usesPseudoLabeling": False,
        "usesFinalTestContent": False,
    }


def _sorted_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(candidates, key=lambda item: (float(item["timestampSeconds"]), int(item["candidateIndex"])))


def _candidate_filename(candidate: Mapping[str, Any]) -> str:
    if "imageFilename" in candidate:
        return str(candidate["imageFilename"])
    return str(candidate["filename"])


def _cluster_record(cluster_index: int, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = candidates[0]
    last = candidates[-1]
    return {
        "clusterId": f"cluster_{cluster_index:03d}",
        "startCandidateIndex": int(first["candidateIndex"]),
        "endCandidateIndex": int(last["candidateIndex"]),
        "startTimestampSeconds": float(first["timestampSeconds"]),
        "endTimestampSeconds": float(last["timestampSeconds"]),
        "candidateCount": len(candidates),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }
