from unittest.mock import Mock, call
from typing import TypedDict

import pytest

from src.automation import route_building
from src.automation import menu_selection
from src.domain import models
from src.domain import types


class JoinCase(TypedDict):
    line_station_indexes: list[int]
    expected_positions: list[types.Position] | None


TEST_CASES = [
    pytest.param(
        JoinCase(
            line_station_indexes=[0],
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


class TestRouteDrawer:
    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request: pytest.FixtureRequest) -> JoinCase:
        return request.param

    @pytest.fixture
    def line_id(self) -> types.LineColour:
        # Arbitrary colour; joining is colour agnostic
        return next(iter(types.LineColour))

    @pytest.fixture(
        params=[
            pytest.param(route_building.AppendingRouteBuilder, id="appending"),
            pytest.param(
                route_building.NearestNeighbourRouteBuilder, id="nearest_neighbour"
            ),
        ]
    )
    def route_builder(
        self, request: pytest.FixtureRequest
    ) -> route_building.RouteBuilder:
        return request.param()

    @pytest.fixture
    def menu_selector(self, mock_mouse_controller: Mock) -> menu_selection.MenuSelector:
        return menu_selection.LineMenuSelector(mouse=mock_mouse_controller)

    @pytest.fixture
    def mock_mouse_controller(self) -> Mock:
        return Mock()

    @pytest.fixture
    def route_drawer(
        self,
        route_builder: route_building.RouteBuilder,
        menu_selector: menu_selection.MenuSelector,
        mock_mouse_controller: Mock,
    ) -> route_building.RouteDrawer:
        return route_building.RouteDrawer(
            route_builder=route_builder,
            menu_selector=menu_selector,
            mouse_controller=mock_mouse_controller,
        )

    @pytest.fixture
    def stations(self) -> dict[types.StationId, models.Station]:
        stations = [
            models.Station(position=(100, 200), shape=types.StationShape.CIRCLE),
            models.Station(position=(300, 400), shape=types.StationShape.TRIANGLE),
            models.Station(position=(500, 600), shape=types.StationShape.SQUARE),
        ]
        return {station.id: station for station in stations}

    @pytest.fixture
    def lines(
        self,
        stations: dict[types.StationId, models.Station],
        test_case: JoinCase,
        line_id: types.LineColour,
    ) -> dict[types.LineColour, models.Line]:
        if test_case["line_station_indexes"]:
            all_station_ids = list(stations)
            station_ids = [
                all_station_ids[index] for index in test_case["line_station_indexes"]
            ]
            return {line_id: models.Line(id=line_id, station_ids=station_ids)}
        else:
            return {}

    @pytest.fixture
    def menus(self, line_id: types.LineColour) -> dict[types.LineColour, models.Menu]:
        return {
            line_id: models.Menu(
                line_id=line_id, position=(50, 100), state=types.MenuState.ACTIVE
            )
        }

    @pytest.fixture
    def game_state(
        self,
        stations: dict[types.StationId, models.Station],
        lines: dict[types.LineColour, models.Line],
        menus: dict[types.LineColour, models.Menu],
    ) -> models.GameState:
        return models.GameState(
            stations=stations,
            lines=lines,
            menus=menus,
        )

    @pytest.fixture
    def expected_calls(
        self,
        test_case: JoinCase,
        menus: dict[types.LineColour, models.Menu],
        line_id: types.LineColour,
    ) -> list[object]:
        menu = menus[line_id]
        expected_positions = test_case["expected_positions"]
        if expected_positions:
            return [
                call.click(menu.position),
                call.join(expected_positions),
                call.click(menu.position),
            ]
        else:
            return []

    def test_draw(
        self,
        route_drawer: route_building.RouteDrawer,
        mock_mouse_controller: Mock,
        game_state: models.GameState,
        expected_calls: list[object],
    ) -> None:
        route_drawer.draw(game_state)
        assert mock_mouse_controller.mock_calls == expected_calls
