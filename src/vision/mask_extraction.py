from abc import ABC, abstractmethod

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


class MaskTransformer(ABC):
    @abstractmethod
    def transform(self, mask: types.Mask) -> types.Mask:
        pass


class DilatingMaskTransformer(MaskTransformer):
    def __init__(self, dilation_size: int):
        self.dilation_size = dilation_size

    def transform(self, mask: types.Mask) -> types.Mask:
        kernel = np.ones((self.dilation_size, self.dilation_size), dtype=np.uint8)
        return cv2.dilate(mask, kernel)


class MorphologicalClosingMaskTransformer(MaskTransformer):
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def transform(self, mask: types.Mask) -> types.Mask:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size),
        )
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
