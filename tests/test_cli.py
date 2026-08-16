from __future__ import annotations

from pathlib import Path

from formicai.cli import build_dataset_parser, build_detect_video_parser, normalize_argv


def test_normalize_argv_accepts_analyze_subcommand() -> None:
    assert normalize_argv(["analyze", "samples/ants.mp4"]) == ["samples/ants.mp4"]


def test_normalize_argv_accepts_direct_video_path() -> None:
    assert normalize_argv(["samples/ants.mp4"]) == ["samples/ants.mp4"]


def test_dataset_extract_frames_parser() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "extract-frames",
            "samples/ant2.mp4",
            "--output",
            "datasets/raw/ant2",
            "--every-seconds",
            "0.25",
            "--roi",
            "60,180,600,700",
        ]
    )

    assert args.dataset_command == "extract-frames"
    assert args.input_video == Path("samples/ant2.mp4")
    assert args.output == Path("datasets/raw/ant2")
    assert args.every_seconds == 0.25
    assert args.roi.to_dict() == {"x": 60, "y": 180, "width": 600, "height": 700}
    assert args.count is None


def test_dataset_extract_frames_parser_accepts_count() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "extract-frames",
            "samples/ant4.mp4",
            "--output",
            "datasets/external_test/ant4/images",
            "--count",
            "20",
            "--roi",
            "40,122,400,472",
        ]
    )

    assert args.dataset_command == "extract-frames"
    assert args.input_video == Path("samples/ant4.mp4")
    assert args.output == Path("datasets/external_test/ant4/images")
    assert args.count == 20
    assert args.roi.to_dict() == {"x": 40, "y": 122, "width": 400, "height": 472}
    assert args.overwrite is False


def test_dataset_extract_frames_parser_accepts_overwrite() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "extract-frames",
            "samples/ant4.mp4",
            "--output",
            "datasets/external_test/ant4/images",
            "--count",
            "20",
            "--overwrite",
        ]
    )

    assert args.dataset_command == "extract-frames"
    assert args.overwrite is True


def test_dataset_prepare_roboflow_parser() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "prepare-roboflow",
            "exported_dataset",
            "--output",
            "datasets/prepared/ants_v1",
            "--train-fraction",
            "0.8",
            "--gap-count",
            "1",
        ]
    )

    assert args.dataset_command == "prepare-roboflow"
    assert args.export_dir == Path("exported_dataset")
    assert args.output == Path("datasets/prepared/ants_v1")
    assert args.train_fraction == 0.8
    assert args.gap_count == 1


def test_detect_video_parser() -> None:
    parser = build_detect_video_parser()

    args = parser.parse_args(
        [
            "samples/ant2.mp4",
            "--model",
            "artifacts/models/ants_v1_baseline/weights/best.pt",
            "--roi",
            "60,180,600,700",
            "--conf",
            "0.25",
            "--iou",
            "0.70",
            "--no-end2end",
            "--device",
            "0",
        ]
    )

    assert args.input_video == Path("samples/ant2.mp4")
    assert args.model == Path("artifacts/models/ants_v1_baseline/weights/best.pt")
    assert args.roi.to_dict() == {"x": 60, "y": 180, "width": 600, "height": 700}
    assert args.conf == 0.25
    assert args.iou == 0.70
    assert args.end2end is False
    assert args.device == "0"


def test_dataset_evaluate_external_parser() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "evaluate-external",
            "datasets/external_test",
            "--model",
            "artifacts/models/ants_v1_baseline/weights/best.pt",
            "--output",
            "artifacts/analysis/external_test_metrics.json",
            "--conf",
            "0.25",
            "--iou",
            "0.70",
            "--imgsz",
            "640",
            "--device",
            "0",
            "--no-end2end",
        ]
    )

    assert args.dataset_command == "evaluate-external"
    assert args.external_test_dir == Path("datasets/external_test")
    assert args.model == Path("artifacts/models/ants_v1_baseline/weights/best.pt")
    assert args.output == Path("artifacts/analysis/external_test_metrics.json")
    assert args.conf == 0.25
    assert args.iou == 0.70
    assert args.imgsz == 640
    assert args.device == "0"
    assert args.end2end is False


def test_dataset_prepare_ants_mendeley_parser() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "prepare-ants-mendeley",
            "--root",
            "datasets/public/ants_mendeley",
            "--analysis-dir",
            "artifacts/analysis/public_ants",
            "--sequences",
            "Seq0001",
            "Seq0006",
        ]
    )

    assert args.dataset_command == "prepare-ants-mendeley"
    assert args.root == Path("datasets/public/ants_mendeley")
    assert args.analysis_dir == Path("artifacts/analysis/public_ants")
    assert args.sequences == ["Seq0001", "Seq0006"]


def test_dataset_evaluate_ants_benchmark_parser() -> None:
    parser = build_dataset_parser()

    args = parser.parse_args(
        [
            "evaluate-ants-benchmark",
            "--root",
            "datasets/public/ants_mendeley",
            "--model",
            "artifacts/models/ants_v1_baseline/weights/best.pt",
            "--output",
            "artifacts/analysis/public_ants/external_benchmark_metrics.json",
            "--conf",
            "0.25",
            "--iou",
            "0.70",
            "--imgsz",
            "640",
            "--device",
            "0",
            "--no-end2end",
            "--sequences",
            "Seq0001",
            "Seq0006",
        ]
    )

    assert args.dataset_command == "evaluate-ants-benchmark"
    assert args.root == Path("datasets/public/ants_mendeley")
    assert args.model == Path("artifacts/models/ants_v1_baseline/weights/best.pt")
    assert args.output == Path("artifacts/analysis/public_ants/external_benchmark_metrics.json")
    assert args.conf == 0.25
    assert args.iou == 0.70
    assert args.imgsz == 640
    assert args.device == "0"
    assert args.end2end is False
    assert args.sequences == ["Seq0001", "Seq0006"]
