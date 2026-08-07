from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class HeatmapAccumulator:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self._accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    def add_motion(self, motion_mask: np.ndarray, x_offset: int = 0, y_offset: int = 0) -> None:
        height, width = motion_mask.shape[:2]
        motion = (motion_mask > 0).astype(np.float32)
        self._accumulator[y_offset : y_offset + height, x_offset : x_offset + width] += motion

    def render(self) -> np.ndarray:
        max_value = float(self._accumulator.max())
        if max_value <= 0.0:
            normalized = np.zeros(self._accumulator.shape, dtype=np.uint8)
        else:
            normalized = cv2.normalize(
                self._accumulator,
                dst=None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            ).astype(np.uint8)

        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), self.render())
        if not success:
            raise RuntimeError(f"Could not write heatmap image to {output_path}.")
