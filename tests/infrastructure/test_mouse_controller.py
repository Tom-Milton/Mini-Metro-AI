from unittest.mock import Mock, call

import pytest

from src.domain import types
from src.infrastructure import mouse_controller


class TestMouseController:
    @pytest.fixture
    def mock_mouse(self) -> Mock:
        mock = Mock()
        mock.position = (0, 0)

        def _track_position(x: int, y: int, *args: object, **kwargs: object) -> None:
            mock.position = (x, y)

        mock.moveTo.side_effect = _track_position
        mock.dragTo.side_effect = _track_position

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

    def test_click(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
        start_position: types.Position,
    ) -> None:
        controller.click(start_position)
        assert mock_mouse.mock_calls == [call.click(*start_position, button="left")]

    def test_join(
        self,
        controller: mouse_controller.MouseController,
        mock_mouse: Mock,
        start_position: types.Position,
        end_position: types.Position,
    ) -> None:
        controller.join([start_position, end_position])
        assert mock_mouse.mock_calls == [
            call.moveTo(*start_position),
            call.mouseDown(button="left"),
            call.dragTo(*end_position, duration=0.1, button="left", mouseDownUp=False),
            call.mouseUp(button="left"),
        ]
        assert mock_mouse.position == end_position
