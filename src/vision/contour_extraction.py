from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np

from src.domain import types


class MaskExtractor(ABC):
    @abstractmethod
    def extract(self, frame: types.Frame) -> types.Mask:
        pass


class ColourMaskExtractor(MaskExtractor):
    def __init__(self, colour: types.Colour, mask_tolerance: int):
        self.colour = colour
        self.mask_tolerance = mask_tolerance

    def extract(self, frame: types.Frame) -> types.Mask:
        colour_array = np.array(self.colour, dtype=np.int16)
        lower = np.clip(colour_array - self.mask_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(colour_array + self.mask_tolerance, 0, 255).astype(np.uint8)
        return cv2.inRange(frame, lower, upper).astype(np.uint8)


class DilatingMaskExtractor(MaskExtractor):
    def __init__(self, mask_extractor: MaskExtractor, dilation_size: int):
        self.mask_extractor = mask_extractor
        self.dilation_size = dilation_size

    def extract(self, frame: types.Frame) -> types.Mask:
        mask = self.mask_extractor.extract(frame)
        kernel = np.ones((self.dilation_size, self.dilation_size), dtype=np.uint8)
        return cv2.dilate(mask, kernel)


class MorphologicalClosingMaskExtractor(MaskExtractor):
    def __init__(self, mask_extractor: MaskExtractor, kernel_size: int):
        self.mask_extractor = mask_extractor
        self.kernel_size = kernel_size

    def extract(self, frame: types.Frame) -> types.Mask:
        mask = self.mask_extractor.extract(frame)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size),
        )
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


class ContourExtractor(ABC):
    @abstractmethod
    def extract(self, frame: types.Frame) -> Sequence[types.Contour]:
        pass


class MaskContourExtractor(ContourExtractor):
    def __init__(
        self,
        mask_extractor: MaskExtractor,
        min_area: float,
    ):
        self.mask_extractor = mask_extractor
        self.min_area = min_area

    def extract(self, frame: types.Frame) -> Sequence[types.Contour]:
        mask = self.mask_extractor.extract(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            contour for contour in contours if cv2.contourArea(contour) >= self.min_area
        ]
