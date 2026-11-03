from unittest.mock import Mock, call

import pytest

from src.automation import menu_selection
from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller


class TestLineMenuSelector:
    @pytest.fixture
    def mock_mouse(self) -> Mock:
        return Mock(spec=mouse_controller.MouseController)

    @pytest.fixture
    def selector(self, mock_mouse: Mock) -> menu_selection.MenuSelector:
        return menu_selection.LineMenuSelector(mouse=mock_mouse)

    @pytest.fixture
    def menu(self, state: types.MenuState) -> models.Menu:
        return models.Menu(
            position=(50, 100), line_id=types.LineColour.YELLOW, state=state
        )

    @pytest.fixture
    def expected_calls(self, state: types.MenuState) -> list[object]:
        clicks = 1 if state == types.MenuState.SELECTED else 2
        return [call.click((50, 100))] * clicks

    @pytest.mark.parametrize("state", list(types.MenuState))
    def test_selection(
        self,
        selector: menu_selection.MenuSelector,
        mock_mouse: Mock,
        menu: models.Menu,
        expected_calls: list[object],
    ) -> None:
        selector.select(menu)
        selector.select(menu)
        selector.deselect(menu)
        selector.deselect(menu)
        assert (mock_mouse.mock_calls, menu.state) == (
            expected_calls,
            types.MenuState.ACTIVE,
        )
