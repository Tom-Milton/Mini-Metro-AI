from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import station_detection
from src.vision import line_detection


class GameStateExtractor:
    def __init__(
        self,
        station_detector: station_detection.StationDetector,
        line_detector: line_detection.LineDetector,
    ) -> None:
        self._station_detector = station_detector
        self._line_detector = line_detector

    def extract(
        self,
        frame: types.Frame,
    ) -> models.GameState:
        game_space = self._game_space(frame)
        stations = self._station_detector.detect(game_space)
        lines = self._line_detector.detect(game_space, stations)
        return models.GameState(
            stations=stations,
            lines=lines,
        )

    def _game_space(self, frame: types.Frame) -> types.Frame:
        """Crop out the bottom UI toolbar before detecting game elements."""
        height = int(frame.shape[0] * config.GAME_SPACE_HEIGHT_RATIO)
        return frame[:height]
