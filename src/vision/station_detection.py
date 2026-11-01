from typing import List
import logging

import cv2

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import contour_extraction
from src.vision import mask_extraction


logger = logging.getLogger(__name__)

_VERTICES_TO_SHAPE: dict[int, types.StationShape] = {
    m.value.vertices: m for m in types.StationShape
}


class StationDetector:
    def __init__(
        self,
        mask_extractor: mask_extraction.MaskExtractor,
        contour_extractor: contour_extraction.ContourExtractor,
    ):
        self.mask_extractor = mask_extractor
        self.contour_extractor = contour_extractor

    def detect(self, frame: types.Frame) -> List[models.Station]:
        mask = self.mask_extractor.extract(frame)
        contours = self.contour_extractor.extract(mask)
        stations = [self._create_station(contour) for contour in contours]
        logger.info("Detected %d stations", len(stations))
        return stations

    def _create_station(self, contour: types.Contour) -> models.Station:
        position = self._compute_centroid(contour)
        shape = self._classify_shape(contour)
        return models.Station(position=position, shape=shape)

    def _compute_centroid(self, contour: types.Contour) -> types.Position:
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

    def _classify_shape(self, contour: types.Contour) -> types.StationShape:
        epsilon = config.POLYGON_EPSILON_RATIO * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        return _VERTICES_TO_SHAPE.get(len(polygon), types.StationShape.CIRCLE)


def build_station_detector() -> StationDetector:
    return StationDetector(
        mask_extractor=mask_extraction.ColourMaskExtractor(
            colour=config.STATION_COLOUR,
            mask_tolerance=config.STATION_MASK_TOLERANCE,
        ),
        contour_extractor=contour_extraction.MaskContourExtractor(
            min_area=config.MIN_CONTOUR_AREA,
        ),
    )
