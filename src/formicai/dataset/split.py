from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class TemporalSplitSummary:
    train_count: int
    val_count: int
    split_timestamp_seconds: float


T = TypeVar("T")


def temporal_split(
    records: list[T],
    train_fraction: float = 0.8,
    gap_count: int = 0,
    timestamp_key: str = "timestampSeconds",
) -> dict[str, list[T]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if gap_count < 0:
        raise ValueError("gap_count must be greater than or equal to 0.")
    if not records:
        return {"train": [], "val": []}

    sorted_records = sorted(records, key=lambda record: _timestamp(record, timestamp_key))
    split_index = max(1, min(len(sorted_records) - 1, round(len(sorted_records) * train_fraction)))

    if gap_count and len(sorted_records) > 2:
        gap_count = min(gap_count, len(sorted_records) - 2)
        train_end = max(1, split_index - gap_count // 2)
        val_start = min(len(sorted_records) - 1, train_end + gap_count)
        return {
            "train": sorted_records[:train_end],
            "gap": sorted_records[train_end:val_start],
            "val": sorted_records[val_start:],
        }

    return {
        "train": sorted_records[:split_index],
        "gap": [],
        "val": sorted_records[split_index:],
    }


def _timestamp(record: object, timestamp_key: str) -> float:
    if isinstance(record, dict):
        return float(record[timestamp_key])
    return float(getattr(record, timestamp_key))
