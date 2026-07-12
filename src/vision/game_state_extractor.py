from src.domain import models
from src.domain import types
from src.vision import station_detector


class GameStateExtractor:
    def __init__(
        self,
        detector: station_detector.StationDetector | None = None,
    ) -> None:
        self._detector = detector or station_detector.StationDetector()

    def extract(
        self,
        frame: types.Frame,
    ) -> models.GameState:
        stations = self._detector.detect(frame)

        return models.GameState(
            stations=stations,
        )
