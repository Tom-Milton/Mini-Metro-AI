import pytest

from src.domain import models
from src.domain import types
from src.vision import config
from src.vision import menu_detection
from tests.fixtures import polygons


class TestMenuDetector:
    @pytest.fixture
    def line_id(self) -> types.LineColour:
        # Arbitrary colour; menu state classification is colour agnostic
        return next(iter(types.LineColour))

    @pytest.fixture
    def detector(self) -> menu_detection.MenuDetector:
        return menu_detection.build_menu_detector()

    @pytest.fixture
    def colours(self) -> dict[types.LineColour, types.Colour]:
        return config.LINE_COLOURS

    @pytest.fixture
    def input_frame(
        self,
        radius: int,
        line_id: types.LineColour,
        colours: dict[types.LineColour, types.Colour],
    ) -> types.Frame:
        return polygons.draw_polygon(
            sides=100,
            radius=radius,
            colour=colours[line_id],
            frame_size=200,
        )

    @pytest.fixture
    def expected_menu(
        self, state: types.MenuState, line_id: types.LineColour
    ) -> models.Menu:
        return models.Menu(position=(49, 49), line_id=line_id, state=state)

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
        assert menu == {expected_menu.line_id: expected_menu}
