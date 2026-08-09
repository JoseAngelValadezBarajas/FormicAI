from __future__ import annotations

import pytest

from formicai.dataset.split import temporal_split


def test_temporal_split_preserves_time_order_and_uses_blocks() -> None:
    records = [
        {"image": "f3.jpg", "timestampSeconds": 3.0},
        {"image": "f1.jpg", "timestampSeconds": 1.0},
        {"image": "f2.jpg", "timestampSeconds": 2.0},
        {"image": "f4.jpg", "timestampSeconds": 4.0},
        {"image": "f5.jpg", "timestampSeconds": 5.0},
    ]

    split = temporal_split(records, train_fraction=0.6)

    assert [record["image"] for record in split["train"]] == ["f1.jpg", "f2.jpg", "f3.jpg"]
    assert [record["image"] for record in split["val"]] == ["f4.jpg", "f5.jpg"]


def test_temporal_split_can_omit_gap_at_boundary() -> None:
    records = [
        {"image": f"f{index}.jpg", "timestampSeconds": float(index)}
        for index in range(1, 7)
    ]

    split = temporal_split(records, train_fraction=0.67, gap_count=1)

    assert [record["image"] for record in split["train"]] == ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"]
    assert [record["image"] for record in split["gap"]] == ["f5.jpg"]
    assert [record["image"] for record in split["val"]] == ["f6.jpg"]


def test_temporal_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        temporal_split([], train_fraction=1.0)
