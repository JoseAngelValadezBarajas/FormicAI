from __future__ import annotations

from formicai.cli import normalize_argv


def test_normalize_argv_accepts_analyze_subcommand() -> None:
    assert normalize_argv(["analyze", "samples/ants.mp4"]) == ["samples/ants.mp4"]


def test_normalize_argv_accepts_direct_video_path() -> None:
    assert normalize_argv(["samples/ants.mp4"]) == ["samples/ants.mp4"]
