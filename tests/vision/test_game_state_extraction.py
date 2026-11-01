import cv2
from pathlib import Path
import pytest

from src.vision import game_state_extraction
from src.vision import station_detection
from src.vision import line_detection
from src.vision import menu_detection
from src.domain import models
from src.domain import types
from src.utils import gateways


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestGameStateExtractor:
    @pytest.fixture
    def extractor(self) -> game_state_extraction.GameStateExtractor:
        return game_state_extraction.GameStateExtractor(
            station_detector=station_detection.build_station_detector(),
            line_detector=line_detection.build_line_detector(),
            menu_detector=menu_detection.build_menu_detector(),
        )

    @pytest.fixture(params=list(FIXTURES_DIR.glob("*.png")), ids=lambda p: p.stem)
    def screenshot_path(self, request: pytest.FixtureRequest) -> Path:
        return request.param

    @pytest.fixture
    def input_frame(self, screenshot_path: Path) -> types.Frame:
        frame = cv2.imread(str(screenshot_path))
        assert frame is not None, f"Failed to load image: {screenshot_path}"
        return frame

    @pytest.fixture
    def expected_state(self, screenshot_path: Path) -> models.GameState:
        state_path = screenshot_path.with_suffix("").with_suffix(".state.json")
        return models.GameState.model_validate(gateways.JsonGateway().load(state_path))

    def test_extract(
        self,
        extractor: game_state_extraction.GameStateExtractor,
        input_frame: types.Frame,
        expected_state: models.GameState,
    ) -> None:
        observed_state = extractor.extract(input_frame)
        assert observed_state == expected_state
