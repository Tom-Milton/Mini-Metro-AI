import pytest
from typing import TypedDict

from src.vision import station_detection
from src.vision import config
from src.domain import types
from src.domain import models
from tests.fixtures import polygons


class StationCase(TypedDict):
    shape: types.StationShape
    sides: int


TEST_CASES = [
    pytest.param(
        StationCase(shape=types.StationShape.CIRCLE, sides=100),
        id="circle",
    ),
    pytest.param(
        StationCase(shape=types.StationShape.TRIANGLE, sides=3),
        id="triangle",
    ),
    pytest.param(
        StationCase(shape=types.StationShape.SQUARE, sides=4),
        id="square",
    ),
]


class TestStationDetector:
    @pytest.fixture
    def detector(self) -> station_detection.StationDetector:
        return station_detection.build_station_detector()

    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request: pytest.FixtureRequest) -> StationCase:
        return request.param

    @pytest.fixture
    def shape(self, test_case: StationCase) -> types.StationShape:
        return test_case["shape"]

    @pytest.fixture
    def sides(self, test_case: StationCase) -> int:
        return test_case["sides"]

    @pytest.fixture
    def centre(self) -> types.Position:
        return types.Position((50, 50))

    @pytest.fixture
    def input_frame(self, sides: int) -> types.Frame:
        return polygons.draw_polygon(sides=sides, color=config.STATION_COLOR)

    @pytest.fixture
    def expected_station(
        self, shape: types.StationShape, centre: types.Position
    ) -> models.Station:
        return models.Station(position=centre, shape=shape)

    def test_detect(
        self,
        detector: station_detection.StationDetector,
        input_frame: types.Frame,
        expected_station: models.Station,
    ) -> None:
        stations = detector.detect(input_frame)
        observed_station = stations[0]
        assert observed_station.shape == expected_station.shape
