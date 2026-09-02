from abc import ABC, abstractmethod
from typing import Callable, Sequence, cast

import cv2
import numpy as np
import skimage.morphology

from src.domain import types
from src.vision import config


# scikit-image does not expose a complete type signature for this function.
_typed_skeletonize = cast(
    Callable[[np.ndarray], np.ndarray],
    getattr(skimage.morphology, "skeletonize"),
)


class MaskExtractor(ABC):
    @abstractmethod
    def extract(self, frame: types.Frame) -> types.Mask:
        pass


class ColourMaskExtractor(MaskExtractor):
    def __init__(self, color: types.Color, mask_tolerance: int):
        self.color = color
        self.mask_tolerance = mask_tolerance

    def extract(self, frame: types.Frame) -> types.Mask:
        color_array = np.array(self.color, dtype=np.int16)
        lower = np.clip(color_array - self.mask_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(color_array + self.mask_tolerance, 0, 255).astype(np.uint8)
        return cv2.inRange(frame, lower, upper).astype(np.uint8)


class DilatingMaskExtractor(MaskExtractor):
    def __init__(self, mask_extractor: MaskExtractor, dilation_size: int):
        self.mask_extractor = mask_extractor
        self.dilation_size = dilation_size

    def extract(self, frame: types.Frame) -> types.Mask:
        mask = self.mask_extractor.extract(frame)
        kernel = np.ones((self.dilation_size, self.dilation_size), dtype=np.uint8)
        return cv2.dilate(mask, kernel)


class SkeletonMaskExtractor(MaskExtractor):
    def __init__(self, mask_extractor: MaskExtractor):
        self.mask_extractor = mask_extractor

    def extract(self, frame: types.Frame) -> types.Mask:
        mask = self.mask_extractor.extract(frame)
        return _typed_skeletonize(mask > 0).astype(np.uint8) * 255


class ContourExtractor:
    def __init__(
        self,
        mask_extractor: MaskExtractor,
        min_area: float = config.MIN_CONTOUR_AREA,
    ):
        self.mask_extractor = mask_extractor
        self.min_area = min_area

    def extract(self, frame: types.Frame) -> Sequence[types.Contour]:
        mask = self.mask_extractor.extract(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            contour for contour in contours if cv2.contourArea(contour) >= self.min_area
        ]
