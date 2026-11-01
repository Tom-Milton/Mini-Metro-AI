import pytest

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import menu_detection
from tests.fixtures import polygons


class TestMenuDetector:
    @pytest.fixture
    def detector(self) -> menu_detection.MenuDetector:
        return menu_detection.build_menu_detector()

    @pytest.fixture
    def input_frame(self, radius: int) -> types.Frame:
        return polygons.draw_polygon(
            sides=100,
            radius=radius,
            colour=config.LINE_COLOUR,
            frame_size=200,
        )

    @pytest.fixture
    def expected_menu(self, state: types.MenuState) -> models.Menu:
        return models.Menu(position=(49, 49), line_id="yellow", state=state)

    @pytest.mark.parametrize(
        "radius, state",
        [
            (3, types.MenuState.AVAILABLE),
            (5, types.MenuState.ACTIVE),
            (10, types.MenuState.SELECTED),
        ],
    )
    def test_detect(
        self,
        detector: menu_detection.MenuDetector,
        input_frame: types.Frame,
        expected_menu: models.Menu,
    ) -> None:
        menu = detector.detect(input_frame)
        assert menu == expected_menu
