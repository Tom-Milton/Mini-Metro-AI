import cv2
from pathlib import Path
import pytest

from src import mini_metro_vision, game_state
from src.types import Frame
from src.utils import gateways


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestExtractGameState:
    @pytest.fixture(params=list(FIXTURES_DIR.glob("*.png")), ids=lambda p: p.stem)
    def screenshot_path(self, request: pytest.FixtureRequest) -> Path:
        return request.param

    @pytest.fixture
    def input_frame(self, screenshot_path: Path) -> Frame:
        frame = cv2.imread(screenshot_path)
        assert frame is not None, f"Failed to load image: {screenshot_path}"
        return frame

    @pytest.fixture
    def expected_state(self, screenshot_path: Path) -> game_state.GameState:
        state_path = screenshot_path.with_suffix("").with_suffix(".state.json")
        return game_state.GameState.model_validate(gateways.JsonGateway().load(state_path))

    def test_extract_game_state_from_real_screenshot(self, input_frame: Frame, expected_state: game_state.GameState) -> None:
        observed_state = mini_metro_vision.extract_game_state(input_frame)
        assert observed_state.sorted() == expected_state.sorted()
