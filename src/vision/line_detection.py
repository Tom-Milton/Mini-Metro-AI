from typing import List
import logging

import cv2
import numpy as np

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import contour_extraction
from src.vision import polyline_extraction
from src.vision import skeleton_extraction


logger = logging.getLogger(__name__)


class LineDetector:
    def __init__(
        self,
        line_polyline_extractor: polyline_extraction.PolylineExtractor,
    ):
        self.line_polyline_extractor = line_polyline_extractor

    def detect(
        self,
        frame: types.Frame,
        stations: List[models.Station],
    ) -> List[models.Line]:
        polylines = self.line_polyline_extractor.extract(frame)
        lines = [self._create_line(polyline, stations) for polyline in polylines]
        logger.info("Detected %d lines", len(lines))
        return lines

    def _create_line(
        self,
        polyline: types.Polyline,
        stations: List[models.Station],
    ) -> models.Line:
        matched_stations = self._find_stations(polyline, stations)
        station_ids = self._order_stations(polyline, matched_stations)
        logger.info("Detected line with %d stations", len(station_ids))
        return models.Line(station_ids=station_ids)

    def _find_stations(
        self,
        polyline: types.Polyline,
        stations: List[models.Station],
    ) -> List[models.Station]:
        """Return the stations that are contained within the given polyline."""
        contour = polyline.reshape(-1, 1, 2)
        return [
            station
            for station in stations
            if abs(cv2.pointPolygonTest(contour, station.position, True))
            <= config.MAX_STATION_LINE_DISTANCE
        ]

    def _order_stations(
        self,
        polyline: types.Polyline,
        stations: List[models.Station],
    ) -> List[types.StationId]:
        """Return station IDs ordered by their position along the polyline."""
        return [
            station.id
            for station in sorted(
                stations,
                key=lambda station: self._polyline_position(polyline, station),
            )
        ]

    def _polyline_position(
        self,
        polyline: types.Polyline,
        station: models.Station,
    ) -> int:
        """Return the index of the polyline point closest to the station."""
        return min(
            range(len(polyline)),
            key=lambda index: cv2.norm(polyline[index], np.array(station.position)),
        )


def build_line_polyline_extractor() -> polyline_extraction.PolylineExtractor:
    colour_mask_extractor = contour_extraction.ColourMaskExtractor(
        colour=config.LINE_COLOUR,
        mask_tolerance=config.MASK_TOLERANCE,
    )
    closing_mask_extractor = contour_extraction.MorphologicalClosingMaskExtractor(
        mask_extractor=colour_mask_extractor,
        kernel_size=config.LINE_CLOSING_SIZE,
    )
    mask_skeleton_extractor = skeleton_extraction.MaskSkeletonExtractor(
        mask_extractor=closing_mask_extractor,
    )
    longest_skeleton_extractor = skeleton_extraction.LongestSkeletonExtraction(
        skeleton_extractor=mask_skeleton_extractor,
    )
    skeleton_polyline_extractor = polyline_extraction.SkeletonPolylineExtractor(
        skeleton_extractor=longest_skeleton_extractor,
    )
    greedy_polyline_extractor = polyline_extraction.GreedyPolylineExtractor(
        polyline_extractor=skeleton_polyline_extractor,
    )
    return greedy_polyline_extractor


def build_line_detector() -> LineDetector:
    return LineDetector(
        line_polyline_extractor=build_line_polyline_extractor(),
    )
