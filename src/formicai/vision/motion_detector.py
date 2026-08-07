from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionDetection:
    x: int
    y: int
    width: int
    height: int
    area: float

    def to_bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class MotionDetectionResult:
    motion_mask: np.ndarray
    detections: list[MotionDetection]
    activity_score: float


def calculate_activity_score(motion_mask: np.ndarray) -> float:
    total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
    if total_pixels <= 0:
        return 0.0
    moving_pixels = int(np.count_nonzero(motion_mask))
    return min(1.0, max(0.0, moving_pixels / total_pixels))


def filter_motion_contours(
    binary_mask: np.ndarray,
    min_area: float,
) -> tuple[np.ndarray, list[MotionDetection]]:
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered_mask = np.zeros_like(binary_mask)
    detections: list[MotionDetection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            MotionDetection(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                area=area,
            )
        )
        cv2.drawContours(filtered_mask, [contour], contourIdx=-1, color=255, thickness=-1)

    detections.sort(key=lambda detection: detection.area, reverse=True)
    return filtered_mask, detections


class MotionDetector:
    def __init__(
        self,
        min_area: float = 25.0,
        history: int = 500,
        var_threshold: float = 16.0,
    ) -> None:
        self._min_area = min_area
        self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> MotionDetectionResult:
        blurred = cv2.GaussianBlur(frame, (5, 5), sigmaX=0)
        foreground_mask = self._background_subtractor.apply(blurred)
        _, binary_mask = cv2.threshold(
            foreground_mask,
            thresh=127,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )

        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self._open_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._close_kernel)

        filtered_mask, detections = filter_motion_contours(
            cleaned,
            min_area=self._min_area,
        )
        return MotionDetectionResult(
            motion_mask=filtered_mask,
            detections=detections,
            activity_score=calculate_activity_score(filtered_mask),
        )
