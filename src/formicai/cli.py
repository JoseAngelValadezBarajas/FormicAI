from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from formicai.detection.video import VideoDetectionConfig, VideoDetector
from formicai.detection.evaluate import ExternalEvaluationConfig, ExternalEvaluator
from formicai.detection.public_ants import BenchmarkConfig, PublicAntsBenchmarkEvaluator
from formicai.dataset.ants_mendeley import AntsMendeleyPreparer
from formicai.dataset.extract_frames import FrameExtractionConfig, FrameExtractor
from formicai.dataset.roboflow import RoboflowDatasetPreparer, inspect_roboflow_export
from formicai.dataset.statistics import DatasetStatisticsCalculator
from formicai.dataset.validation import DatasetValidator
from formicai.utils.video import parse_roi
from formicai.vision.analyzer import AnalysisConfig, VideoAnalyzer


def build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formicai",
        description="Analyze motion activity in a video and generate an annotated video, heatmap, and metrics.",
    )
    parser.add_argument("input_video", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        metavar="x,y,width,height",
        help="Optional region of interest to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where analyzed.mp4, activity_heatmap.png, and metrics.json are written.",
    )
    parser.add_argument(
        "--min-motion-area",
        type=float,
        default=None,
        help=(
            "Minimum contour area in pixels for a moving region to be kept. "
            "If omitted, it is computed from --min-motion-area-ratio and the ROI size."
        ),
    )
    parser.add_argument(
        "--min-motion-area-ratio",
        type=float,
        default=0.0001,
        help="Default contour area as a ratio of ROI pixels when --min-motion-area is omitted.",
    )
    parser.add_argument(
        "--background-history",
        type=int,
        default=500,
        help="Number of frames used by OpenCV MOG2 for background modeling.",
    )
    parser.add_argument(
        "--background-var-threshold",
        type=float,
        default=64.0,
        help="OpenCV MOG2 variance threshold. Higher values reduce sensitivity to small changes.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Log progress every N processed frames.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames used to warm up the background model before scoring motion.",
    )
    parser.add_argument(
        "--write-motion-mask",
        action="store_true",
        help="Write output/motion_mask.mp4 with the final binary mask used for scoring.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def build_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formicai dataset",
        description="Prepare and validate FormicAI datasets.",
    )
    subparsers = parser.add_subparsers(dest="dataset_command", required=True)

    extract = subparsers.add_parser(
        "extract-frames",
        help="Extract deterministic candidate frames from a video.",
    )
    extract.add_argument("input_video", type=Path, help="Path to source video.")
    extract.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where extracted frames and metadata.json are written.",
    )
    extract.add_argument(
        "--every-seconds",
        type=float,
        default=0.25,
        help="Extract one frame approximately every N seconds.",
    )
    extract.add_argument(
        "--every-frames",
        type=int,
        default=None,
        help="Extract one frame every N frames. Overrides --every-seconds.",
    )
    extract.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Extract exactly N temporally distributed frames across the source video. "
            "Cannot be combined with --every-frames."
        ),
    )
    extract.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        metavar="x,y,width,height",
        help="Optional crop region to write as extracted images.",
    )
    extract.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    inspect_export = subparsers.add_parser(
        "inspect-export",
        help="Inspect a Roboflow YOLO export without preparing it.",
    )
    inspect_export.add_argument("export_dir", type=Path, help="Path to Roboflow export directory.")
    inspect_export.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    prepare_roboflow = subparsers.add_parser(
        "prepare-roboflow",
        help="Validate and temporally split a Roboflow YOLO detection export.",
    )
    prepare_roboflow.add_argument("export_dir", type=Path, help="Path to Roboflow export directory.")
    prepare_roboflow.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/prepared/ants_v1"),
        help="Prepared dataset output directory.",
    )
    prepare_roboflow.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of temporally sorted frames assigned to train before applying any gap.",
    )
    prepare_roboflow.add_argument(
        "--gap-count",
        type=int,
        default=1,
        help="Number of boundary frames to omit between train and val.",
    )
    prepare_roboflow.add_argument(
        "--allow-test-as-training-source",
        action="store_true",
        help=(
            "Dangerous legacy override: allow Roboflow test split records to be used as source data "
            "for prepared train/val construction. By default Roboflow test is inspected but excluded "
            "to protect true holdouts."
        ),
    )
    prepare_roboflow.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate a prepared YOLO detection dataset.",
    )
    validate.add_argument("dataset_dir", type=Path, help="Path to prepared dataset directory.")
    validate.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    stats = subparsers.add_parser(
        "stats",
        help="Print simple statistics for a prepared YOLO detection dataset.",
    )
    stats.add_argument("dataset_dir", type=Path, help="Path to prepared dataset directory.")
    stats.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    evaluate_external = subparsers.add_parser(
        "evaluate-external",
        help="Evaluate a detector against annotated external_test video folders without training.",
    )
    evaluate_external.add_argument(
        "external_test_dir",
        type=Path,
        help="Directory containing per-video folders with images/ and labels/.",
    )
    evaluate_external.add_argument("--model", type=Path, required=True, help="Path to detector weights.")
    evaluate_external.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/external_test_metrics.json"),
        help="JSON metrics output path.",
    )
    evaluate_external.add_argument("--conf", type=float, default=0.25, help="Validation confidence threshold.")
    evaluate_external.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    evaluate_external.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    evaluate_external.add_argument(
        "--device",
        default=None,
        help="Ultralytics device value, such as 0, cuda:0, or cpu.",
    )
    evaluate_external.add_argument(
        "--end2end",
        dest="end2end",
        action="store_true",
        default=None,
        help="Force YOLO end-to-end validation mode.",
    )
    evaluate_external.add_argument(
        "--no-end2end",
        dest="end2end",
        action="store_false",
        help="Disable YOLO end-to-end mode and use traditional NMS.",
    )
    evaluate_external.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    prepare_ants_mendeley = subparsers.add_parser(
        "prepare-ants-mendeley",
        help="Convert selected public ANTS Mendeley benchmark sequences to YOLO labels without training.",
    )
    prepare_ants_mendeley.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/public/ants_mendeley"),
        help="ANTS Mendeley dataset root containing raw/, prepared/, and metadata/.",
    )
    prepare_ants_mendeley.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("artifacts/analysis/public_ants"),
        help="Directory where ground-truth sanity montages are written.",
    )
    prepare_ants_mendeley.add_argument(
        "--sequences",
        nargs="+",
        default=["Seq0001", "Seq0006"],
        help="Public ANTS sequence names to prepare.",
    )
    prepare_ants_mendeley.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    evaluate_ants_benchmark = subparsers.add_parser(
        "evaluate-ants-benchmark",
        help="Evaluate frozen best.pt on public ANTS benchmark sequences without training.",
    )
    evaluate_ants_benchmark.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/public/ants_mendeley"),
        help="ANTS Mendeley dataset root.",
    )
    evaluate_ants_benchmark.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/ants_v1_baseline/weights/best.pt"),
        help="Frozen detector weights.",
    )
    evaluate_ants_benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/public_ants/external_benchmark_metrics.json"),
        help="Consolidated benchmark JSON path.",
    )
    evaluate_ants_benchmark.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("artifacts/analysis/public_ants/predictions"),
        help="Directory for GT-vs-prediction examples.",
    )
    evaluate_ants_benchmark.add_argument(
        "--video-dir",
        type=Path,
        default=Path("output/public_ants"),
        help="Directory for reconstructed prediction videos.",
    )
    evaluate_ants_benchmark.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    evaluate_ants_benchmark.add_argument("--iou", type=float, default=0.70, help="Traditional NMS IoU threshold.")
    evaluate_ants_benchmark.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    evaluate_ants_benchmark.add_argument("--device", default="0", help="Ultralytics device value.")
    evaluate_ants_benchmark.add_argument(
        "--end2end",
        dest="end2end",
        action="store_true",
        default=None,
        help="Force YOLO end-to-end mode.",
    )
    evaluate_ants_benchmark.add_argument(
        "--no-end2end",
        dest="end2end",
        action="store_false",
        help="Disable YOLO end-to-end mode and use traditional NMS.",
    )
    evaluate_ants_benchmark.set_defaults(end2end=False)
    evaluate_ants_benchmark.add_argument(
        "--sequences",
        nargs="+",
        default=["Seq0001", "Seq0006"],
        help="Public ANTS sequence names to evaluate.",
    )
    evaluate_ants_benchmark.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser


def build_detect_video_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formicai detect-video",
        description="Run an ant object detector on a video and write annotated MP4 plus JSONL detections.",
    )
    parser.add_argument("input_video", type=Path, help="Path to source video.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained detector weights.")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        metavar="x,y,width,height",
        help="Optional region of interest. Output JSON boxes are written in global frame coordinates.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("output/ant_detections.mp4"),
        help="Annotated video output path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/ant_detections.jsonl"),
        help="Incremental JSONL detections output path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold when end-to-end mode is disabled.",
    )
    parser.add_argument(
        "--end2end",
        dest="end2end",
        action="store_true",
        default=None,
        help="Force YOLO end-to-end inference mode.",
    )
    parser.add_argument(
        "--no-end2end",
        dest="end2end",
        action="store_false",
        help="Disable YOLO end-to-end mode and use one-to-many predictions with traditional NMS.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device value, such as 0, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def normalize_argv(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None

    normalized = list(argv)
    if normalized and normalized[0] == "analyze":
        return normalized[1:]
    return normalized


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s  %(message)s",
    )


def run_analysis(args: argparse.Namespace) -> int:
    config = AnalysisConfig(
        input_path=args.input_video,
        output_dir=args.output_dir,
        roi=args.roi,
        min_motion_area=args.min_motion_area,
        min_motion_area_ratio=args.min_motion_area_ratio,
        background_history=args.background_history,
        background_var_threshold=args.background_var_threshold,
        progress_interval=args.progress_interval,
        warmup_frames=args.warmup_frames,
        write_motion_mask=args.write_motion_mask,
    )

    try:
        result = VideoAnalyzer(config).analyze()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except Exception as exc:
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Unexpected error while analyzing video")
        else:
            logger.error("Unexpected error while analyzing video: %s", exc)
        return 1

    logging.getLogger(__name__).info("Generated %s", result.annotated_video_path)
    logging.getLogger(__name__).info("Generated %s", result.heatmap_path)
    logging.getLogger(__name__).info("Generated %s", result.metrics_path)
    if result.motion_mask_video_path is not None:
        logging.getLogger(__name__).info("Generated %s", result.motion_mask_video_path)
    return 0


def run_dataset(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        if args.dataset_command == "extract-frames":
            result = FrameExtractor(
                FrameExtractionConfig(
                    input_path=args.input_video,
                    output_dir=args.output,
                    roi=args.roi,
                    every_seconds=args.every_seconds,
                    every_frames=args.every_frames,
                    target_frame_count=args.count,
                )
            ).extract()
            logger.info("Extracted %s frames", result.extracted_frames)
            logger.info("Metadata: %s", result.metadata_path)
            return 0

        if args.dataset_command == "inspect-export":
            result = inspect_roboflow_export(args.export_dir)
            print(result.to_text())
            return 0 if result.ok else 1

        if args.dataset_command == "prepare-roboflow":
            result = RoboflowDatasetPreparer(
                export_dir=args.export_dir,
                output_dir=args.output,
                train_fraction=args.train_fraction,
                gap_count=args.gap_count,
                allow_test_as_training_source=args.allow_test_as_training_source,
            ).prepare()
            print(result.to_text())
            return 0

        if args.dataset_command == "validate":
            result = DatasetValidator(args.dataset_dir).validate()
            print(result.to_text())
            return 0 if result.ok else 1

        if args.dataset_command == "stats":
            print(DatasetStatisticsCalculator(args.dataset_dir).calculate().to_text())
            return 0

        if args.dataset_command == "evaluate-external":
            result = ExternalEvaluator(
                ExternalEvaluationConfig(
                    external_test_dir=args.external_test_dir,
                    model_path=args.model,
                    output_path=args.output,
                    confidence=args.conf,
                    iou=args.iou,
                    image_size=args.imgsz,
                    device=args.device,
                    end2end=args.end2end,
                )
            ).evaluate()
            print(result.to_text())
            return 0

        if args.dataset_command == "prepare-ants-mendeley":
            result = AntsMendeleyPreparer(
                dataset_root=args.root,
                analysis_dir=args.analysis_dir,
            ).prepare(tuple(args.sequences))
            print(result.to_text())
            return 0

        if args.dataset_command == "evaluate-ants-benchmark":
            result = PublicAntsBenchmarkEvaluator(
                BenchmarkConfig(
                    dataset_root=args.root,
                    model_path=args.model,
                    output_path=args.output,
                    predictions_dir=args.predictions_dir,
                    video_dir=args.video_dir,
                    confidence=args.conf,
                    iou=args.iou,
                    image_size=args.imgsz,
                    device=args.device,
                    end2end=args.end2end,
                    sequences=tuple(args.sequences),
                )
            ).evaluate()
            print(result.to_text())
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Unexpected dataset command error")
        else:
            logger.error("Unexpected dataset command error: %s", exc)
        return 1

    logger.error("Unknown dataset command: %s", args.dataset_command)
    return 1


def run_detect_video(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        result = VideoDetector(
            VideoDetectionConfig(
                input_path=args.input_video,
                model_path=args.model,
                output_video_path=args.output_video,
                output_jsonl_path=args.output_jsonl,
                roi=args.roi,
                confidence=args.conf,
                iou=args.iou,
                end2end=args.end2end,
                image_size=args.imgsz,
                device=args.device,
            )
        ).detect()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Unexpected video detection error")
        else:
            logger.error("Unexpected video detection error: %s", exc)
        return 1

    logger.info("Processed %s frames", result.processed_frames)
    logger.info("Detections: %s", result.total_detections)
    logger.info("Confidence threshold: %.3f", result.confidence)
    logger.info("NMS IoU threshold: %.3f", result.iou)
    logger.info("End-to-end override: %s", result.end2end)
    logger.info("Generated %s", result.output_video_path)
    logger.info("Generated %s", result.output_jsonl_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv and raw_argv[0] == "dataset":
        parser = build_dataset_parser()
        args = parser.parse_args(raw_argv[1:])
        return run_dataset(args)
    if raw_argv and raw_argv[0] == "detect-video":
        parser = build_detect_video_parser()
        args = parser.parse_args(raw_argv[1:])
        return run_detect_video(args)

    normalized_argv = normalize_argv(raw_argv)
    parser = build_analyze_parser()
    args = parser.parse_args(normalized_argv)
    configure_logging(args.log_level)
    return run_analysis(args)
