from typing import List
import logging

import cv2
import numpy as np

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import mask_extraction
from src.vision import polyline_extraction
from src.vision import skeleton_extraction


logger = logging.getLogger(__name__)


class LinePolylineExtractor:
    def __init__(
        self,
        station_colour_mask_extractor: mask_extraction.ColourMaskExtractor,
        line_colour_mask_extractor: mask_extraction.ColourMaskExtractor,
        dilating_mask_transformer: mask_extraction.DilatingMaskTransformer,
        morphological_closing_mask_transformer: mask_extraction.MorphologicalClosingMaskTransformer,
        mask_skeleton_extractor: skeleton_extraction.MaskSkeletonExtractor,
        eulerian_polyline_extractor: polyline_extraction.EulerianPolylineExtractor,
    ):
        self.station_colour_mask_extractor = station_colour_mask_extractor
        self.line_colour_mask_extractor = line_colour_mask_extractor
        self.dilating_mask_transformer = dilating_mask_transformer
        self.morphological_closing_mask_transformer = (
            morphological_closing_mask_transformer
        )
        self.mask_skeleton_extractor = mask_skeleton_extractor
        self.eulerian_polyline_extractor = eulerian_polyline_extractor

    def extract(self, frame: types.Frame) -> types.Polyline:
        station_mask = self.station_colour_mask_extractor.extract(frame)
        line_mask = self.line_colour_mask_extractor.extract(frame)
        combined_mask = self.dilating_mask_transformer.transform(
            cv2.bitwise_or(line_mask, station_mask)
        )
        combined_mask = self.morphological_closing_mask_transformer.transform(
            combined_mask
        )
        raw_skeleton = self.mask_skeleton_extractor.extract(combined_mask)
        return self.eulerian_polyline_extractor.extract(raw_skeleton)


class LineDetector:
    def __init__(
        self,
        line_polyline_extractor: LinePolylineExtractor,
    ):
        self.line_polyline_extractor = line_polyline_extractor

    def detect(
        self,
        frame: types.Frame,
        stations: List[models.Station],
    ) -> List[models.Line]:
        try:
            polyline = self.line_polyline_extractor.extract(frame)
        except skeleton_extraction.NoSkeletonError:
            return []
        lines = [self._create_line(polyline, stations)]
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
        """Return station IDs in the order they occur along the polyline."""
        stations = sorted(
            stations,
            key=lambda station: self._polyline_position(polyline, station),
        )
        return [station.id for station in stations]

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


def build_line_polyline_extractor() -> LinePolylineExtractor:
    return LinePolylineExtractor(
        station_colour_mask_extractor=mask_extraction.ColourMaskExtractor(
            colour=config.STATION_COLOUR,
            mask_tolerance=config.STATION_MASK_TOLERANCE,
        ),
        line_colour_mask_extractor=mask_extraction.ColourMaskExtractor(
            colour=config.LINE_COLOUR,
            mask_tolerance=config.LINE_MASK_TOLERANCE,
        ),
        dilating_mask_transformer=mask_extraction.DilatingMaskTransformer(
            dilation_size=config.LINE_DILATION_SIZE,
        ),
        morphological_closing_mask_transformer=(
            mask_extraction.MorphologicalClosingMaskTransformer(
                kernel_size=config.MASK_CLOSING_KERNEL_SIZE,
            )
        ),
        mask_skeleton_extractor=skeleton_extraction.MaskSkeletonExtractor(
            min_component_size=config.MIN_SKELETON_COMPONENT_SIZE,
        ),
        eulerian_polyline_extractor=polyline_extraction.EulerianPolylineExtractor(),
    )


def build_line_detector() -> LineDetector:
    return LineDetector(
        line_polyline_extractor=build_line_polyline_extractor(),
    )
