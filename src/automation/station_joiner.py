import logging

from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller


logger = logging.getLogger(__name__)


class StationJoiner:
    def __init__(self, mouse: mouse_controller.MouseController | None) -> None:
        self._mouse_controller = mouse or mouse_controller.MouseController()

    def join(self, game_state: models.GameState) -> None:
        new_stations = self._find_unjoined_stations(game_state)
        if not new_stations:
            logger.info("No stations to join")
            return

        route = self._build_route(game_state, new_stations)
        self._mouse_controller.join(route)
        logger.info("Joined %d stations", len(new_stations))

    def _find_unjoined_stations(
        self, game_state: models.GameState
    ) -> list[models.Station]:
        joined_ids = {
            station_id for line in game_state.lines for station_id in line.station_ids
        }
        return [
            station for station in game_state.stations if station.id not in joined_ids
        ]

    def _build_route(
        self,
        game_state: models.GameState,
        new_stations: list[models.Station],
    ) -> list[types.Position]:
        new_positions = [station.position for station in new_stations]
        if not game_state.lines:
            return new_positions

        stations_by_id = {station.id: station for station in game_state.stations}
        last_station_id = game_state.lines[0].station_ids[-1]
        last_position = stations_by_id[last_station_id].position
        return [last_position, *new_positions]
