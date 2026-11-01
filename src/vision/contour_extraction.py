from abc import ABC, abstractmethod
from typing import Sequence

import cv2

from src.domain import types


class ContourExtractor(ABC):
    @abstractmethod
    def extract(self, mask: types.Mask) -> Sequence[types.Contour]:
        pass


class MaskContourExtractor(ContourExtractor):
    def __init__(self, min_area: float):
        self.min_area = min_area

    def extract(self, mask: types.Mask) -> Sequence[types.Contour]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            contour for contour in contours if cv2.contourArea(contour) >= self.min_area
        ]
