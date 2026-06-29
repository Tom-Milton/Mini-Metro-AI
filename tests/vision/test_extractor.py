import cv2
from pathlib import Path
import pytest

from src.vision import extractor
from src import models, types
from src.utils import gateways


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestExtractor:
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

    def test_extract_game_state(
        self, input_frame: types.Frame, expected_state: models.GameState
    ) -> None:
        observed_state = extractor.extract_game_state(input_frame).sorted()
        expected_state = expected_state.sorted()
        assert observed_state == expected_state
