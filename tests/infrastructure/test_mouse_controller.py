import pynput.mouse as mouse
from unittest.mock import Mock

import pytest

from src.domain import types
from src.infrastructure import mouse_controller


class TestMouseController:
    @pytest.fixture
    def mock_mouse(self) -> Mock:
        mock = Mock()
        mock.position = (0, 0)
        return mock

    @pytest.fixture
    def start_position(self) -> types.Position:
        return types.Position((100, 200))

    @pytest.fixture
    def end_position(self) -> types.Position:
        return types.Position((300, 400))

    @pytest.fixture
    def controller(self, mock_mouse: Mock) -> mouse_controller.MouseController:
        return mouse_controller.MouseController(mock_mouse)

    def test_press(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
    ) -> None:
        controller.press()
        mock_mouse.press.assert_called_once_with(mouse.Button.left)

    def test_release(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
    ) -> None:
        controller.release()
        mock_mouse.release.assert_called_once_with(mouse.Button.left)

    def test_move(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
        end_position: types.Position,
    ) -> None:
        controller.move(end_position)
        assert mock_mouse.position == end_position

    def test_drag(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
        end_position: types.Position,
    ) -> None:
        controller.drag(end_position)
        assert mock_mouse.position == end_position

    def test_join(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
        start_position: types.Position,
        end_position: types.Position,
    ) -> None:
        controller.join([start_position, end_position])
        mock_mouse.press.assert_called_once_with(mouse.Button.left)
        mock_mouse.release.assert_called_once_with(mouse.Button.left)
        assert mock_mouse.position == end_position
