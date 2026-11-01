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


def compute_centroid(contour: types.Contour) -> types.Position:
    """m00 is area; m10 and m01 are x/y first moments used for centroid."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        centroid = (0, 0)
    else:
        centroid = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
    return types.Position(centroid)
