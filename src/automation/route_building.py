import logging
from abc import ABC, abstractmethod

from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller
from src.automation import menu_selection


logger = logging.getLogger(__name__)


def unjoined_stations(game_state: models.GameState) -> list[types.StationId]:
    joined_station_ids = {
        station_id
        for line in game_state.lines.values()
        for station_id in line.station_ids
    }
    return [
        station_id
        for station_id in game_state.stations
        if station_id not in joined_station_ids
    ]


class RouteBuilder(ABC):
    @abstractmethod
    def build(self, game_state: models.GameState) -> list[models.Line]:
        pass


class AppendingRouteBuilder(RouteBuilder):
    def build(self, game_state: models.GameState) -> list[models.Line]:
        station_ids = unjoined_stations(game_state)
        if not station_ids or not game_state.lines:
            return []
        line = next(iter(game_state.lines.values()))
        station_ids = [line.station_ids[-1], *station_ids]
        return [models.Line(id=line.id, station_ids=station_ids)]


class RouteDrawer:
    def __init__(
        self,
        route_builder: RouteBuilder,
        menu_selector: menu_selection.MenuSelector,
        mouse_controller: mouse_controller.MouseController,
    ) -> None:
        self._route_builder = route_builder
        self._menu_selector = menu_selector
        self._mouse_controller = mouse_controller

    def draw(self, game_state: models.GameState) -> None:
        for route in self._route_builder.build(game_state):
            self.draw_route(route, game_state.stations, game_state.menus)

    def draw_route(
        self,
        route: models.Line,
        stations: dict[types.StationId, models.Station],
        menus: dict[types.LineColour, models.Menu],
    ) -> None:
        positions = [stations[station_id].position for station_id in route.station_ids]
        self._menu_selector.select(menus[route.id])
        self._mouse_controller.join(positions)
        self._menu_selector.deselect(menus[route.id])
        logger.info("Joined %d stations", len(route.station_ids) - 1)


def build_route_drawer() -> RouteDrawer:
    return RouteDrawer(
        route_builder=AppendingRouteBuilder(),
        menu_selector=menu_selection.build_menu_selector(),
        mouse_controller=mouse_controller.MouseController(),
    )
