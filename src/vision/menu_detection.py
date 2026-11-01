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
        mask_extractor: mask_extraction.MaskExtractor,
        contour_extractor: contour_extraction.ContourExtractor,
        diameter_ratios: dict[types.MenuState, float],
    ):
        self.mask_extractor = mask_extractor
        self.contour_extractor = contour_extractor
        self.diameter_ratios = diameter_ratios

    def detect(self, frame: types.Frame) -> models.Menu:
        mask = self.mask_extractor.extract(frame)
        contours = self.contour_extractor.extract(mask)
        contour = max(contours, key=cv2.contourArea)
        position = contour_extraction.compute_centroid(contour)
        menu = models.Menu(
            position=position,
            line_id="yellow",
            state=self._classify_state(contour, frame.shape[0]),
        )
        logger.info("Detected %s menu", menu.line_id)
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
        mask_extractor=mask_extraction.ColourMaskExtractor(
            colour=config.LINE_COLOUR,
            mask_tolerance=config.STATION_MASK_TOLERANCE,
        ),
        contour_extractor=contour_extraction.MaskContourExtractor(
            min_area=config.MIN_CONTOUR_AREA,
        ),
        diameter_ratios={
            types.MenuState.AVAILABLE: config.MENU_AVAILABLE_DIAMETER_RATIO,
            types.MenuState.ACTIVE: config.MENU_ACTIVE_DIAMETER_RATIO,
            types.MenuState.SELECTED: config.MENU_SELECTED_DIAMETER_RATIO,
        },
    )
