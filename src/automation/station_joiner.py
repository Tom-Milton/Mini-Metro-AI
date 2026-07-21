import logging

from src.domain import models
from src.infrastructure import mouse_controller


logger = logging.getLogger(__name__)


class StationJoiner:
    def __init__(self, mouse: mouse_controller.MouseController | None) -> None:
        self._mouse_controller = mouse or mouse_controller.MouseController()

    def join(self, stations: list[models.Station]) -> None:
        self._mouse_controller.join([station.position for station in stations])
        logger.info("Joined %d stations", len(stations))
