from typing import List
import logging

import cv2
import numpy as np

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import contour_extraction


logger = logging.getLogger(__name__)


class LineDetector:
    def __init__(self, contour_extractor: contour_extraction.ContourExtractor):
        self.contour_extractor = contour_extractor

    def detect(
        self,
        frame: types.Frame,
        stations: List[models.Station],
    ) -> List[models.Line]:
        contours = self.contour_extractor.extract(frame)
        lines = [self._create_line(contour, stations) for contour in contours]
        return lines

    def _create_line(
        self,
        contour: types.Contour,
        stations: List[models.Station],
    ) -> models.Line:
        matched_stations = self._find_stations(contour, stations)
        station_ids = self._order_stations(contour, matched_stations)
        return models.Line(station_ids=station_ids)

    def _find_stations(
        self,
        contour: types.Contour,
        stations: List[models.Station],
    ) -> List[models.Station]:
        """Return the stations that are contained within the given contour."""
        return [
            station
            for station in stations
            if abs(cv2.pointPolygonTest(contour, station.position, True))
            <= config.MAX_STATION_LINE_DISTANCE
        ]

    def _order_stations(
        self,
        contour: types.Contour,
        stations: List[models.Station],
    ) -> List[types.StationId]:
        """Return the station IDs ordered by their position along the contour."""
        return [
            station.id
            for station in sorted(
                stations,
                key=lambda station: self._contour_position(contour, station),
            )
        ]

    def _contour_position(
        self,
        contour: types.Contour,
        station: models.Station,
    ) -> int:
        """Return the index of the contour point closest to the station."""
        return min(
            range(len(contour)),
            key=lambda index: cv2.norm(contour[index][0], np.array(station.position)),
        )


def build_line_detector() -> LineDetector:
    colour_mask_extractor = contour_extraction.ColourMaskExtractor(
        color=config.LINE_COLOR,
        mask_tolerance=config.MASK_TOLERANCE,
    )
    dilating_mask_extractor = contour_extraction.DilatingMaskExtractor(
        mask_extractor=colour_mask_extractor,
        dilation_size=config.LINE_DILATION_SIZE,
    )
    skeleton_mask_extractor = contour_extraction.SkeletonMaskExtractor(
        mask_extractor=dilating_mask_extractor,
    )
    contour_extractor = contour_extraction.ContourExtractor(
        mask_extractor=skeleton_mask_extractor,
        min_area=0,  # Skeletonized lines are 1px wide and have near-zero area
    )
    return LineDetector(
        contour_extractor=contour_extractor,
    )
