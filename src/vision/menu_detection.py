import logging

import cv2

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import contour_extraction
from src.vision import mask_extraction


logger = logging.getLogger(__name__)


class MenuDetector:
    def __init__(
        self,
        mask_extractors: dict[types.LineColour, mask_extraction.MaskExtractor],
        contour_extractor: contour_extraction.ContourExtractor,
        diameter_ratios: dict[types.MenuState, float],
    ):
        self.mask_extractors = mask_extractors
        self.contour_extractor = contour_extractor
        self.diameter_ratios = diameter_ratios

    def detect(self, frame: types.Frame) -> dict[types.LineColour, models.Menu]:
        return {
            menu.line_id: menu
            for line_colour in self.mask_extractors
            if (menu := self._detect_menu(frame, line_colour)) is not None
        }

    def _detect_menu(
        self, frame: types.Frame, line_colour: types.LineColour
    ) -> models.Menu | None:
        mask = self.mask_extractors[line_colour].extract(frame)
        contours = self.contour_extractor.extract(mask)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        menu = models.Menu(
            line_id=line_colour,
            position=contour_extraction.compute_centroid(contour),
            state=self._classify_state(contour, frame.shape[0]),
        )
        logger.info("Detected %s menu", line_colour.value)
        return menu

    def _classify_state(
        self, contour: types.Contour, frame_height: int
    ) -> types.MenuState:
        _, radius = cv2.minEnclosingCircle(contour)
        diameter_ratio = 2 * radius / frame_height
        state = min(
            self.diameter_ratios,
            key=lambda state: abs(diameter_ratio - self.diameter_ratios[state]),
        )
        return state


def build_menu_detector() -> MenuDetector:
    return MenuDetector(
        mask_extractors={
            line_colour: mask_extraction.ColourMaskExtractor(
                colour=mask_colour,
                mask_tolerance=config.STATION_MASK_TOLERANCE,
            )
            for line_colour, mask_colour in config.LINE_COLOURS.items()
        },
        contour_extractor=contour_extraction.MaskContourExtractor(
            min_area=config.MIN_CONTOUR_AREA,
        ),
        diameter_ratios=config.MENU_DIAMETER_RATIOS,
    )
