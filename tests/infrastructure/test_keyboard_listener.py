from unittest.mock import Mock
from pynput import keyboard

import pytest

from src.infrastructure import keyboard_listener


class TestKeyboardListener:
    @pytest.fixture
    def on_escape(self) -> Mock:
        return Mock()

    @pytest.fixture
    def on_space(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_listener(self) -> Mock:
        return Mock(spec=keyboard.Listener)

    @pytest.fixture
    def listener(
        self,
        on_escape: Mock,
        on_space: Mock,
        mock_listener: Mock,
    ) -> keyboard_listener.KeyboardListener:
        return keyboard_listener.KeyboardListener(
            on_escape=on_escape,
            on_space=on_space,
            listener=mock_listener,
        )

    def test_escape(
        self, listener: keyboard_listener.KeyboardListener, on_escape: Mock
    ) -> None:
        listener.on_press(keyboard.Key.esc)
        on_escape.assert_called_once_with()

    def test_space(
        self, listener: keyboard_listener.KeyboardListener, on_space: Mock
    ) -> None:
        listener.on_release(keyboard.Key.space)
        on_space.assert_called_once_with()
