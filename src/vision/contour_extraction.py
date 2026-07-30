from typing import Sequence

import cv2
import numpy as np

from src.domain import types


class ColourMaskContourExtractor:
    def __init__(self, color: types.Color, mask_tolerance: int):
        self.color = color
        self.mask_tolerance = mask_tolerance

    def extract(self, frame: types.Frame) -> Sequence[types.Contour]:
        color_array = np.array(self.color, dtype=np.uint8)
        lower = np.clip(color_array - self.mask_tolerance, 0, 255)
        upper = np.clip(color_array + self.mask_tolerance, 0, 255)
        mask = cv2.inRange(frame, lower, upper).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
