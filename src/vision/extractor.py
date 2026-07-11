import src.domain.types as types
import src.domain.models as models
from src.vision import detector


def extract_game_state(frame: types.Frame) -> models.GameState:
    det = detector.StationDetector()
    stations = det.detect(frame)
    return models.GameState(stations=stations)
