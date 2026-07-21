from unittest.mock import Mock, call

import pytest

from src.automation import station_joiner
from src.domain import models
from src.domain import types


class TestStationJoiner:
    @pytest.fixture
    def mock_mouse_controller(self) -> Mock:
        return Mock()

    @pytest.fixture
    def joiner(self, mock_mouse_controller: Mock) -> station_joiner.StationJoiner:
        return station_joiner.StationJoiner(mock_mouse_controller)

    @pytest.fixture
    def stations(self) -> list[models.Station]:
        return [
            models.Station(position=(100, 200), shape=types.StationShape.CIRCLE),
            models.Station(position=(300, 400), shape=types.StationShape.TRIANGLE),
            models.Station(position=(500, 600), shape=types.StationShape.SQUARE),
        ]

    @pytest.fixture
    def expected_calls(self) -> list[object]:
        return [call([(100, 200), (300, 400), (500, 600)])]

    def test_join(
        self,
        joiner: station_joiner.StationJoiner,
        mock_mouse_controller: Mock,
        stations: list[models.Station],
        expected_calls: list[object],
    ) -> None:
        joiner.join(stations)
        assert mock_mouse_controller.join.call_args_list == expected_calls
