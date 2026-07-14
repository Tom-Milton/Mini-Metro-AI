import numpy as np
import cv2
import pytest
from functools import partial
from typing import Callable

import src.vision.station_detector as station_detector
import src.domain.types as types
import src.domain.models as models
import src.vision.config as config


TEST_FRAME_SIZE = 100
TEST_POLYGON_CENTER = (50, 50)
TEST_POLYGON_RADIUS = 30


def _draw_polygon(img: types.Frame, sides: int) -> None:
    angles = np.linspace(-np.pi / 2, 2 * np.pi - np.pi / 2, sides, endpoint=False)
    points = np.column_stack(
        (
            TEST_POLYGON_CENTER[0] + TEST_POLYGON_RADIUS * np.cos(angles),
            TEST_POLYGON_CENTER[1] + TEST_POLYGON_RADIUS * np.sin(angles),
        )
    ).astype(np.int32)
    cv2.fillPoly(img, [points], config.STATION_COLOR)


TEST_CASES = [
    pytest.param(
        (types.StationShape.CIRCLE, partial(_draw_polygon, sides=100)), id="circle"
    ),
    pytest.param(
        (types.StationShape.TRIANGLE, partial(_draw_polygon, sides=3)), id="triangle"
    ),
    pytest.param(
        (types.StationShape.SQUARE, partial(_draw_polygon, sides=4)), id="square"
    ),
]


class TestStationDetector:
    @pytest.fixture
    def detector(self) -> station_detector.StationDetector:
        return station_detector.StationDetector()

    @pytest.fixture(params=TEST_CASES)
    def test_case(
        self, request: pytest.FixtureRequest
    ) -> tuple[types.StationShape, Callable[[types.Frame], None]]:
        return request.param

    @pytest.fixture
    def shape(
        self, test_case: tuple[types.StationShape, Callable[[types.Frame], None]]
    ) -> types.StationShape:
        return test_case[0]

    @pytest.fixture
    def drawer(
        self, test_case: tuple[types.StationShape, Callable[[types.Frame], None]]
    ) -> Callable[[types.Frame], None]:
        return test_case[1]

    @pytest.fixture
    def input_frame(self, drawer: Callable[[types.Frame], None]) -> types.Frame:
        frame = np.zeros((TEST_FRAME_SIZE, TEST_FRAME_SIZE, 3), dtype=np.uint8)
        drawer(frame)
        return frame

    @pytest.fixture
    def expected_station(self, shape: types.StationShape) -> models.Station:
        return models.Station(position=TEST_POLYGON_CENTER, shape=shape)

    def test_detect(
        self,
        detector: station_detector.StationDetector,
        input_frame: types.Frame,
        expected_station: models.Station,
    ) -> None:
        stations = detector.detect(input_frame)
        observed_station = stations[0]
        assert observed_station.shape == expected_station.shape
