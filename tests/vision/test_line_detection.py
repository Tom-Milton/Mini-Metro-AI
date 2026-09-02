import pytest

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import line_detection
from tests.fixtures import polygons


class TestLineDetector:
    @pytest.fixture
    def detector(self) -> line_detection.LineDetector:
        return line_detection.build_line_detector()

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        return polygons.draw_rectangle(
            width=100,
            height=1,
            colour=config.LINE_COLOUR,
        )

    @pytest.fixture
    def stations(self) -> list[models.Station]:
        return [
            models.Station(position=(25, 50), shape=types.StationShape.CIRCLE),
            models.Station(position=(75, 50), shape=types.StationShape.SQUARE),
        ]

    @pytest.fixture
    def expected_line(self) -> models.Line:
        return models.Line(
            station_ids=["circle-25-50", "square-75-50"],
        )

    def test_detect(
        self,
        detector: line_detection.LineDetector,
        input_frame: types.Frame,
        stations: list[models.Station],
        expected_line: models.Line,
    ) -> None:
        lines = detector.detect(input_frame, stations)
        observed_line = lines[0]
        assert observed_line == expected_line
