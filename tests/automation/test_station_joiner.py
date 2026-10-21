from unittest.mock import Mock, call
from typing import TypedDict

import pytest

from src.automation import station_joiner
from src.domain import models
from src.domain import types


class JoinCase(TypedDict):
    line_station_indexes: list[int]
    expected_positions: list[types.Position] | None


TEST_CASES = [
    pytest.param(
        JoinCase(
            line_station_indexes=[],
            expected_positions=[(100, 200), (300, 400), (500, 600)],
        ),
        id="joins_all_stations",
    ),
    pytest.param(
        JoinCase(
            line_station_indexes=[0, 1], expected_positions=[(300, 400), (500, 600)]
        ),
        id="joins_remaining_station",
    ),
    pytest.param(
        JoinCase(line_station_indexes=[0, 1, 2], expected_positions=None),
        id="joins_no_stations",
    ),
]


class TestStationJoiner:
    @pytest.fixture(
        params=[
            pytest.param(station_joiner.AppendRouteBuilder, id="append"),
            pytest.param(
                station_joiner.NearestNeighbourRouteBuilder,
                id="nearest-neighbour",
            ),
        ]
    )
    def builder_type(
        self,
        request: pytest.FixtureRequest,
    ) -> type[station_joiner.RouteBuilder]:
        return request.param

    @pytest.fixture
    def mock_mouse_controller(self) -> Mock:
        return Mock()

    @pytest.fixture
    def joiner(
        self,
        builder_type: type[station_joiner.RouteBuilder],
        mock_mouse_controller: Mock,
    ) -> station_joiner.StationJoiner:
        return station_joiner.StationJoiner(
            route_builder=builder_type(),
            mouse=mock_mouse_controller,
        )

    @pytest.fixture
    def stations(self) -> list[models.Station]:
        return [
            models.Station(position=(100, 200), shape=types.StationShape.CIRCLE),
            models.Station(position=(300, 400), shape=types.StationShape.TRIANGLE),
            models.Station(position=(500, 600), shape=types.StationShape.SQUARE),
        ]

    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request: pytest.FixtureRequest) -> JoinCase:
        return request.param

    @pytest.fixture
    def lines(
        self,
        stations: list[models.Station],
        test_case: JoinCase,
    ) -> list[models.Line]:
        if test_case["line_station_indexes"]:
            station_ids = [
                stations[index].id for index in test_case["line_station_indexes"]
            ]
            return [models.Line(station_ids=station_ids)]
        else:
            return []

    @pytest.fixture
    def game_state(
        self,
        stations: list[models.Station],
        lines: list[models.Line],
    ) -> models.GameState:
        return models.GameState(
            stations=stations,
            lines=lines,
        )

    @pytest.fixture
    def expected_calls(self, test_case: JoinCase) -> list[object]:
        expected_positions = test_case["expected_positions"]
        return [call(expected_positions)] if expected_positions else []

    def test_join(
        self,
        joiner: station_joiner.StationJoiner,
        mock_mouse_controller: Mock,
        game_state: models.GameState,
        expected_calls: list[object],
    ) -> None:
        joiner.join(game_state)
        assert mock_mouse_controller.join.call_args_list == expected_calls
