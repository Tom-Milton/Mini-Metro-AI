from unittest.mock import Mock, call

import pytest

from src.automation import menu_selector
from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller


class TestMenuSelector:
    @pytest.fixture
    def mock_mouse(self) -> Mock:
        return Mock(spec=mouse_controller.MouseController)

    @pytest.fixture
    def selector(self, mock_mouse: Mock) -> menu_selector.MenuSelector:
        return menu_selector.MenuSelector(mouse=mock_mouse)

    @pytest.fixture
    def menu(self, state: types.MenuState) -> models.Menu:
        return models.Menu(position=(50, 100), line_id="yellow", state=state)

    @pytest.fixture
    def expected_calls(self, state: types.MenuState) -> list[object]:
        return [] if state == types.MenuState.SELECTED else [call.click((50, 100))]

    @pytest.mark.parametrize("state", list(types.MenuState))
    def test_select(
        self,
        selector: menu_selector.MenuSelector,
        mock_mouse: Mock,
        menu: models.Menu,
        expected_calls: list[object],
    ) -> None:
        selector.select(menu)
        assert mock_mouse.mock_calls == expected_calls
