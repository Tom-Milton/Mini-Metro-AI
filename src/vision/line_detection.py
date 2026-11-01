from typing import List, Sequence
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
        mask_extractor: mask_extraction.MaskExtractor,
        mask_transformer: mask_extraction.MaskTransformer,
        skeleton_extractor: skeleton_extraction.SkeletonExtractor,
        skeleton_transformer: skeleton_extraction.SkeletonTransformer,
        polyline_extractor: polyline_extraction.PolylineExtractor,
        polyline_transformer: polyline_extraction.PolylineTransformer,
    ):
        self.mask_extractor = mask_extractor
        self.mask_transformer = mask_transformer
        self.skeleton_extractor = skeleton_extractor
        self.skeleton_transformer = skeleton_transformer
        self.polyline_extractor = polyline_extractor
        self.polyline_transformer = polyline_transformer

    def extract(self, frame: types.Frame) -> Sequence[types.Polyline]:
        mask = self.mask_transformer.transform(self.mask_extractor.extract(frame))
        try:
            skeleton = self.skeleton_extractor.extract(mask)
        except skeleton_extraction.NoSkeletonError:
            return []
        skeleton = self.skeleton_transformer.transform(skeleton)
        polylines = self.polyline_extractor.extract(skeleton)
        return self.polyline_transformer.transform(polylines)


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


def build_line_polyline_extractor() -> LinePolylineExtractor:
    return LinePolylineExtractor(
        mask_extractor=mask_extraction.ColourMaskExtractor(
            colour=config.LINE_COLOUR,
            mask_tolerance=config.MASK_TOLERANCE,
        ),
        mask_transformer=mask_extraction.DilatingMaskTransformer(
            dilation_size=config.LINE_DILATION_SIZE,
        ),
        skeleton_extractor=skeleton_extraction.MaskSkeletonExtractor(),
        skeleton_transformer=skeleton_extraction.LongestSkeletonTransformer(),
        polyline_extractor=polyline_extraction.SkeletonPolylineExtractor(),
        polyline_transformer=polyline_extraction.GreedyPolylineTransformer(),
    )


def build_line_detector() -> LineDetector:
    return LineDetector(
        line_polyline_extractor=build_line_polyline_extractor(),
    )
