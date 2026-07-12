from typing import List, Sequence
import logging

import cv2
import numpy as np

import src.vision.config as config
import src.domain.types as types
import src.domain.models as models


logger = logging.getLogger(__name__)

_VERTICES_TO_SHAPE: dict[int, types.StationShape] = {m.value.vertices: m for m in types.StationShape}


class StationDetector:
    def __init__(self, color: types.Color = config.STATION_COLOR, tolerance: int = config.STATION_TOLERANCE):
        self.color = color
        self.tolerance = tolerance

    def detect(self, frame: types.Frame) -> List[models.Station]:
        contours = self._extract_contours(frame)
        logger.debug("Found %d contours", len(contours))

        stations: list[models.Station] = []
        for contour in contours:
            position = self._compute_centroid(contour)
            shape = self._classify_shape(contour)
            logger.debug("Detected %s at %s", shape, position)
            stations.append(models.Station(position=position, shape=shape))

        logger.info("Detected %d stations", len(stations))
        return stations

    def _extract_contours(self, frame: types.Frame) -> Sequence[types.Contour]:
        color_array = np.array(self.color, dtype=np.uint8)
        lower = np.clip(color_array - self.tolerance, 0, 255)
        upper = np.clip(color_array + self.tolerance, 0, 255)
        mask = cv2.inRange(frame, lower, upper).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _compute_centroid(self, contour: types.Contour) -> types.Position:
        return tuple(contour.mean(axis=0).astype(int)[0])

    def _classify_shape(self, contour: types.Contour) -> types.StationShape:
        epsilon = config.POLYGON_EPSILON_RATIO * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        return _VERTICES_TO_SHAPE.get(len(polygon), types.StationShape.CIRCLE)
