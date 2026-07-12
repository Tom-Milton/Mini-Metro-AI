import src.domain.types as types
import src.domain.models as models
from src.vision.station_detector import StationDetector


class GameStateExtractor:
    def __init__(
        self,
        detector: StationDetector | None = None,
    ) -> None:
        self._detector = detector or StationDetector()

    def extract(
        self,
        frame: types.Frame,
    ) -> models.GameState:
        stations = self._detector.detect(frame)

        return models.GameState(
            stations=stations,
        )
