from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def validate_within(self, frame_width: int, frame_height: int) -> "ROI":
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be greater than or equal to 0.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be greater than 0.")
        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError(
                "ROI must fit inside the video frame "
                f"({frame_width}x{frame_height})."
            )
        return self

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def full_frame_roi(frame_width: int, frame_height: int) -> ROI:
    return ROI(x=0, y=0, width=frame_width, height=frame_height)


def parse_roi(value: str) -> ROI:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "ROI must use the format x,y,width,height."
        )

    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROI values must be integers."
        ) from exc

    roi = ROI(x=x, y=y, width=width, height=height)
    if roi.x < 0 or roi.y < 0:
        raise argparse.ArgumentTypeError("ROI x and y must be non-negative.")
    if roi.width <= 0 or roi.height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive.")
    return roi
