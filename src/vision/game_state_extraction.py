from src.domain import models
from src.domain import types
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
        return models.GameState(
            stations=self._station_detector.detect(frame),
            lines=self._line_detector.detect(frame),
        )
