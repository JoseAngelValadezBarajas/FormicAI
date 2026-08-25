from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from formicai.utils.hashing import sha256_file


class ForbiddenSourceError(ValueError):
    """Raised when a source-role guard blocks filesystem inspection."""


@dataclass(frozen=True)
class SourceFileInspection:
    path: str
    file_size_bytes: int
    sha256: str
    video_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class V5DevelopmentPathPolicy:
    path: str
    protected: bool
    semantic_access_allowed: bool
    non_semantic_metadata_allowed: bool
    matched_tokens: tuple[str, ...]


def load_source_role_registry(registry_path: Path) -> dict[str, Any]:
    return json.loads(registry_path.read_text(encoding="utf-8"))


def assert_training_source_audit_allowed(
    source_path: Path | str,
    source_role_registry: Mapping[str, Any] | Path,
) -> None:
    """Reject forbidden sources before semantic filesystem inspection."""

    registry = _coerce_registry(source_role_registry)
    matches = forbidden_training_source_audit_matches(source_path, registry)
    if matches:
        match_list = ", ".join(sorted(matches))
        raise ForbiddenSourceError(
            "Source is forbidden for semantic training-source audit/design "
            f"inspection: {source_path} matched {match_list}"
        )


def classify_v5_development_path(
    path: Path | str,
    source_role_registry: Mapping[str, Any] | Path,
) -> V5DevelopmentPathPolicy:
    """Classify a path without touching the filesystem.

    Non-semantic path metadata checks are allowed for protected paths so guards
    can be enforced. Semantic access remains blocked for matched protected paths.
    """

    registry = _coerce_registry(source_role_registry)
    matches = forbidden_training_source_audit_matches(path, registry)
    protected = bool(matches)
    return V5DevelopmentPathPolicy(
        path=str(path),
        protected=protected,
        semantic_access_allowed=not protected,
        non_semantic_metadata_allowed=True,
        matched_tokens=tuple(sorted(matches)),
    )


def assert_v5_development_path_allowed(
    path: Path | str,
    source_role_registry: Mapping[str, Any] | Path,
) -> None:
    """Reject sealed V5 final-test paths before semantic read/search operations."""

    assert_training_source_audit_allowed(path, source_role_registry)


def filter_v5_development_paths(
    paths: list[Path],
    source_role_registry: Mapping[str, Any] | Path,
) -> list[Path]:
    """Return only paths allowed for V5 development search/read workflows."""

    allowed: list[Path] = []
    for path in paths:
        try:
            assert_v5_development_path_allowed(path, source_role_registry)
        except ForbiddenSourceError:
            continue
        allowed.append(path)
    return allowed


def forbidden_training_source_audit_matches(
    source_path: Path | str,
    source_role_registry: Mapping[str, Any] | Path,
) -> set[str]:
    registry = _coerce_registry(source_role_registry)
    candidate = _normalize_identifier(source_path)
    matches: set[str] = set()

    for token in _iter_forbidden_training_audit_tokens(registry):
        normalized_token = _normalize_identifier(token)
        if _path_or_identifier_matches(candidate, normalized_token):
            matches.add(str(token))

    return matches


def inspect_training_source_file(
    source_path: Path | str,
    source_role_registry: Mapping[str, Any] | Path,
    *,
    stat_func: Callable[[Path], Any] | None = None,
    sha256_func: Callable[[Path], str] | None = None,
    video_probe_func: Callable[[Path], Mapping[str, Any]] | None = None,
) -> SourceFileInspection:
    """Inspect an allowed training source after role guards have passed.

    The role guard intentionally runs before stat/hash/probe callbacks, so tests
    can prove final-test sources are blocked before semantic file inspection.
    """

    path = Path(source_path)
    assert_training_source_audit_allowed(path, source_role_registry)

    stat = stat_func or Path.stat
    hasher = sha256_func or sha256_file

    stat_result = stat(path)
    video_metadata = video_probe_func(path) if video_probe_func is not None else None

    return SourceFileInspection(
        path=str(path),
        file_size_bytes=int(stat_result.st_size),
        sha256=hasher(path),
        video_metadata=video_metadata,
    )


def _coerce_registry(source_role_registry: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    if isinstance(source_role_registry, Mapping):
        return source_role_registry
    return load_source_role_registry(Path(source_role_registry))


def _iter_forbidden_training_audit_tokens(registry: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()

    explicit_guard = registry.get("forbiddenForTrainingSourceAudit", {})
    if isinstance(explicit_guard, Mapping):
        tokens.update(str(value) for value in explicit_guard.get("identifiers", []))
        tokens.update(str(value) for value in explicit_guard.get("paths", []))
        tokens.update(
            str(value) for value in explicit_guard.get("sealedDetailedMetadataPaths", [])
        )
        for source in explicit_guard.get("sources", []):
            tokens.update(_forbidden_tokens_from_source("FINAL_TEST", source))

    roles = registry.get("roles", {})
    if isinstance(roles, Mapping):
        for role_name, role_record in roles.items():
            if not isinstance(role_record, Mapping):
                continue
            sources = role_record.get("sources", [])
            for source in sources:
                tokens.update(_forbidden_tokens_from_source(role_name, source))

    return {token for token in tokens if token}


def _forbidden_tokens_from_source(role_name: str, source: Any) -> set[str]:
    tokens: set[str] = set()
    is_final_test_role = role_name == "FINAL_TEST"

    if isinstance(source, str):
        if is_final_test_role:
            tokens.add(source)
        return tokens

    if not isinstance(source, Mapping):
        return tokens

    forbidden_for_design = bool(
        source.get("forbiddenForDesign")
        or source.get("forbiddenForTrainingSourceAudit")
        or (source.get("finalTest") and source.get("trainEligible") is False)
        or is_final_test_role
    )
    if not forbidden_for_design:
        return tokens

    for key in (
        "sourceId",
        "source",
        "name",
        "filename",
        "path",
        "datasetPath",
        "root",
    ):
        value = source.get(key)
        if isinstance(value, str):
            tokens.add(value)

    for key in ("aliases", "pathAliases", "identifiers"):
        values = source.get(key, [])
        if isinstance(values, list):
            tokens.update(str(value) for value in values)

    return tokens


def _normalize_identifier(value: Path | str) -> str:
    normalized = str(value).replace("\\", "/").strip().lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _path_or_identifier_matches(candidate: str, token: str) -> bool:
    if not candidate or not token:
        return False

    candidate_parts = [part for part in candidate.split("/") if part]

    if "/" not in token:
        token_stem = token.rsplit(".", maxsplit=1)[0]
        return (
            token in candidate_parts
            or candidate_parts[-1:] == [token]
            or token_stem in candidate_parts
            or candidate_parts[-1:] == [token_stem]
        )

    return (
        candidate == token
        or candidate.startswith(f"{token}/")
        or candidate.endswith(f"/{token}")
        or f"/{token}/" in f"/{candidate}/"
    )
