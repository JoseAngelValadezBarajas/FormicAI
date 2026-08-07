from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


class AnnotatedVideoWriter:
    def __init__(
        self,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        self._output_path = output_path

        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}.")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
