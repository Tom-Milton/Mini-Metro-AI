import logging
from abc import ABC, abstractmethod

from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller
from src.automation import menu_selection


logger = logging.getLogger(__name__)


class RouteBuilder(ABC):
    @abstractmethod
    def build(
        self,
        game_state: models.GameState,
        new_stations: list[models.Station],
        line_id: types.LineColour,
    ) -> list[types.Position]:
        pass


class AppendRouteBuilder(RouteBuilder):
    def build(
        self,
        game_state: models.GameState,
        new_stations: list[models.Station],
        line_id: types.LineColour,
    ) -> list[types.Position]:
        new_positions = [station.position for station in new_stations]
        stations_by_id = {station.id: station for station in game_state.stations}
        endpoint_id = game_state.lines[line_id].station_ids[-1]
        return [stations_by_id[endpoint_id].position, *new_positions]


class NearestNeighbourRouteBuilder(RouteBuilder):
    def build(
        self,
        game_state: models.GameState,
        new_stations: list[models.Station],
        line_id: types.LineColour,
    ) -> list[types.Position]:
        new_positions = [station.position for station in new_stations]
        endpoints = self._line_endpoints(game_state.lines[line_id], game_state.stations)
        start = self._closest_endpoint(endpoints, new_positions)
        return [start, *self._nearest_neighbour_order(start, new_positions)]

    def _line_endpoints(
        self, line: models.Line, stations: list[models.Station]
    ) -> list[types.Position]:
        stations_by_id = {station.id: station for station in stations}
        station_ids = line.station_ids
        return [
            stations_by_id[station_id].position
            for station_id in (station_ids[0], station_ids[-1])
        ]

    def _closest_endpoint(
        self,
        endpoints: list[types.Position],
        positions: list[types.Position],
    ) -> types.Position:
        return min(
            endpoints,
            key=lambda endpoint: self._distance(
                endpoint,
                self._closest_position(endpoint, positions),
            ),
        )

    def _nearest_neighbour_order(
        self,
        start: types.Position,
        positions: list[types.Position],
    ) -> list[types.Position]:
        route = []
        remaining = positions.copy()
        current = start
        while remaining:
            next_position = self._closest_position(current, remaining)
            route.append(next_position)
            remaining.remove(next_position)
            current = next_position
        return route

    def _closest_position(
        self,
        position: types.Position,
        candidates: list[types.Position],
    ) -> types.Position:
        return min(
            candidates, key=lambda candidate: self._distance(position, candidate)
        )

    def _distance(
        self,
        first: types.Position,
        second: types.Position,
    ) -> float:
        return float((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


class StationJoiner:
    def __init__(
        self,
        route_builder: RouteBuilder,
        selector: menu_selection.MenuSelector,
        mouse: mouse_controller.MouseController,
    ) -> None:
        self._route_builder = route_builder
        self._mouse_controller = mouse
        self._selector = selector

    def join(self, game_state: models.GameState, line_id: types.LineColour) -> None:
        new_stations = self._find_unjoined_stations(game_state)
        if not new_stations:
            logger.info("No stations to join")
            return

        if line_id in game_state.lines:
            route = self._route_builder.build(game_state, new_stations, line_id)
        else:
            route = [station.position for station in new_stations]

        self._selector.select(game_state.menus[line_id])
        self._mouse_controller.join(route)
        logger.info("Joined %d stations", len(new_stations))

    def _find_unjoined_stations(
        self, game_state: models.GameState
    ) -> list[models.Station]:
        joined_ids = {
            station_id
            for line in game_state.lines.values()
            for station_id in line.station_ids
        }
        return [
            station for station in game_state.stations if station.id not in joined_ids
        ]


def build_station_joiner() -> StationJoiner:
    return StationJoiner(
        selector=menu_selection.build_menu_selector(),
        route_builder=NearestNeighbourRouteBuilder(),
        mouse=mouse_controller.MouseController(),
    )
