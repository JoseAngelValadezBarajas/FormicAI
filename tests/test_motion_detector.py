from __future__ import annotations

import cv2
import numpy as np

from formicai.vision.motion_detector import calculate_activity_score, filter_motion_contours


def test_calculate_activity_score_counts_nonzero_pixels() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:2, 0:5] = 255

    assert calculate_activity_score(mask) == 0.1


def test_filter_motion_contours_removes_small_regions() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(mask, (1, 1), (2, 2), color=255, thickness=-1)
    cv2.rectangle(mask, (10, 10), (24, 24), color=255, thickness=-1)

    filtered_mask, detections = filter_motion_contours(mask, min_area=20.0)

    assert len(detections) == 1
    assert detections[0].to_bbox() == (10, 10, 15, 15)
    assert np.count_nonzero(filtered_mask[1:3, 1:3]) == 0
    assert np.count_nonzero(filtered_mask[10:25, 10:25]) > 0
