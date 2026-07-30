from typing import List
import logging

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import contour_extraction


logger = logging.getLogger(__name__)


class LineDetector:
    def __init__(
        self, contour_extractor: contour_extraction.ColourMaskContourExtractor
    ):
        self.counter_extractor = contour_extractor

    def detect(self, frame: types.Frame) -> List[models.Line]:
        raise NotImplementedError("Line detection not implemented yet")


def build_line_detector() -> LineDetector:
    return LineDetector(
        contour_extractor=contour_extraction.ColourMaskContourExtractor(
            color=config.LINE_COLOR,
            mask_tolerance=config.MASK_TOLERANCE,
        )
    )
