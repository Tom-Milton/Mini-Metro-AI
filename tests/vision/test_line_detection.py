import pytest

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import line_detection
from tests.fixtures import polygons


class TestLineDetector:
    @pytest.fixture
    def line_id(self) -> types.LineColour:
        # Arbitrary colour; line extraction is colour agnostic
        return next(iter(types.LineColour))

    @pytest.fixture
    def detector(self) -> line_detection.LineDetector:
        return line_detection.build_line_detector()

    @pytest.fixture
    def input_frame(self, line_id: types.LineColour) -> types.Frame:
        return polygons.draw_rectangle(
            width=500, height=1, colour=config.LINE_COLOURS[line_id]
        )

    @pytest.fixture
    def stations(self) -> list[models.Station]:
        return [
            models.Station(position=(125, 0), shape=types.StationShape.CIRCLE),
            models.Station(position=(375, 0), shape=types.StationShape.SQUARE),
        ]

    @pytest.fixture
    def expected_lines(
        self, line_id: types.LineColour
    ) -> dict[types.LineColour, models.Line]:
        return {
            line_id: models.Line(
                id=line_id,
                station_ids=["square-375-0", "circle-125-0"],
            )
        }

    def test_detect(
        self,
        detector: line_detection.LineDetector,
        input_frame: types.Frame,
        stations: list[models.Station],
        expected_lines: dict[types.LineColour, models.Line],
    ) -> None:
        assert detector.detect(input_frame, stations) == expected_lines
