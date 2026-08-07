from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from formicai.utils.video import parse_roi
from formicai.vision.analyzer import AnalysisConfig, VideoAnalyzer


def build_parser() -> argparse.ArgumentParser:
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


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    normalized_argv = normalize_argv(raw_argv)
    parser = build_parser()
    args = parser.parse_args(normalized_argv)
    configure_logging(args.log_level)

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
