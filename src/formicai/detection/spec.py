from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class DetectorSpec:
    experiment: str
    model_path: Path
    sha256: str
    status: str
    confidence: float
    iou: float
    image_size: int
    end2end: bool
    classes: dict[int, str]


CURRENT_CHAMPION = DetectorSpec(
    experiment="ants_v3_mixedscale_e01",
    model_path=Path("artifacts/models/ants_v3_mixedscale_e01/selected.pt"),
    sha256="424d508ef2b881740134c3c3a320f0dba4ea12d2f4a9f77a057451539347daaf",
    status="CURRENT_CHAMPION",
    confidence=0.25,
    iou=0.70,
    image_size=640,
    end2end=False,
    classes={0: "ant"},
)

KNOWN_DETECTOR_SPECS = (CURRENT_CHAMPION,)


def identify_detector_by_sha256(sha256: str) -> DetectorSpec | None:
    normalized = sha256.lower()
    for spec in KNOWN_DETECTOR_SPECS:
        if spec.sha256.lower() == normalized:
            return spec
    return None


def validate_model_classes(model_names: object, expected_classes: Mapping[int, str]) -> dict[int, str]:
    names = _normalize_class_map(model_names)
    expected_ids = set(expected_classes)
    if set(names) != expected_ids:
        raise ValueError(
            "Unexpected detector class ids. "
            f"Expected {sorted(expected_ids)}, found {sorted(names)}."
        )

    normalized: dict[int, str] = {}
    for class_id, expected_name in expected_classes.items():
        actual_name = names[class_id]
        if _semantic_name(actual_name) != _semantic_name(expected_name):
            raise ValueError(
                "Unexpected detector class name. "
                f"Class {class_id} expected {expected_name!r}, found {actual_name!r}."
            )
        normalized[class_id] = expected_name
    return normalized


def _normalize_class_map(model_names: object) -> dict[int, str]:
    if isinstance(model_names, Mapping):
        normalized: dict[int, str] = {}
        for raw_id, raw_name in model_names.items():
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid detector class id: {raw_id!r}") from exc
            normalized[class_id] = str(raw_name)
        return normalized

    if isinstance(model_names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(model_names)}

    raise ValueError("Detector class map must be a mapping or sequence.")


def _semantic_name(value: str) -> str:
    return value.strip().lower()
