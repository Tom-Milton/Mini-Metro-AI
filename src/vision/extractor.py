from src import types, models
from src.vision import detector


def extract_game_state(frame: types.Frame) -> models.GameState:
    det = detector.StationDetector()
    stations = det.detect(frame)
    return models.GameState(stations=stations)
