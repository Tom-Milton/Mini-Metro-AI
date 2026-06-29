from typing import List
import logging

import cv2
import numpy as np

from src.game_state import Station, GameState
from src.types import Color, Frame, Mask, Position, Contour, StationShape


logger = logging.getLogger(__name__)

WHITE: Color = np.array([248, 248, 248], dtype=np.uint8)
_VERTICES_TO_SHAPE: dict[int, StationShape] = {m.value.vertices: m for m in StationShape}


class StationDetector:
    def _create_color_mask(self, frame: Frame, color: Color, tolerance: int) -> Mask:
        lower = np.clip(color - tolerance, 0, 255)
        upper = np.clip(color + tolerance, 0, 255)
        return cv2.inRange(frame, lower, upper).astype(np.uint8)

    def detect(self, frame: Frame) -> List[Station]:
        white_mask = self._create_color_mask(frame, WHITE, tolerance=3)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug("Found %d contours", len(contours))

        stations: list[Station] = []
        for contour in contours:
            position = self._get_centroid(contour)
            shape = self._classify_shape(contour)
            logger.debug("Detected %s at %s", shape, position)
            stations.append(Station(position=position, shape=shape))

        logger.info("Detected %d stations", len(stations))
        return stations

    def _get_centroid(self, contour: Contour) -> Position:
        return tuple(contour.mean(axis=0).astype(int)[0])

    def _classify_shape(self, contour: Contour) -> StationShape:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        return _VERTICES_TO_SHAPE.get(len(polygon), StationShape.CIRCLE)


def extract_game_state(frame: Frame) -> GameState:
    detector = StationDetector()
    stations = detector.detect(frame)
    return GameState(stations=stations)
