import numpy as np

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import station_detection
from src.vision import line_detection
from src.vision import menu_detection


class GameStateExtractor:
    def __init__(
        self,
        station_detector: station_detection.StationDetector,
        line_detector: line_detection.LineDetector,
        menu_detector: menu_detection.MenuDetector,
    ) -> None:
        self._station_detector = station_detector
        self._line_detector = line_detector
        self._menu_detector = menu_detector

    def extract(
        self,
        frame: types.Frame,
    ) -> models.GameState:
        game_space, menu_space = self._split_frame(frame)
        menu = self._menu_detector.detect(menu_space)
        stations = self._station_detector.detect(game_space)
        lines = self._line_detector.detect(game_space, stations)
        return models.GameState(
            stations=stations,
            lines=lines,
            menus={menu.line_id: menu},
        )

    def _split_frame(self, frame: types.Frame) -> tuple[types.Frame, types.Frame]:
        """Isolate game and menu spaces while preserving full-frame coordinates."""
        height = int(frame.shape[0] * config.GAME_SPACE_HEIGHT_RATIO)
        return self._mask(frame, 0, height), self._mask(frame, height, frame.shape[0])

    def _mask(self, frame: types.Frame, start: int, stop: int) -> types.Frame:
        masked_frame = np.zeros_like(frame)
        masked_frame[start:stop] = frame[start:stop]
        return masked_frame
