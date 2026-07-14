from unittest.mock import Mock
from pynput import keyboard

import pytest

from src.infrastructure import escape_listener


class TestEscapeListener:
    @pytest.fixture
    def on_escape(self) -> Mock:
        return Mock()

    @pytest.fixture
    def keyboard_listener(
        self,
    ) -> keyboard.Listener:
        return keyboard.Listener(
            on_press=lambda key: None,
        )

    @pytest.fixture
    def listener(
        self,
        on_escape: Mock,
        keyboard_listener: keyboard.Listener,
    ) -> escape_listener.EscapeListener:
        return escape_listener.EscapeListener(
            on_escape=on_escape,
            listener=keyboard_listener,
        )

    def test_esc_triggers_callback(
        self,
        listener: escape_listener.EscapeListener,
        on_escape: Mock,
    ) -> None:
        listener.on_press(keyboard.Key.esc)
        on_escape.assert_called_once_with()

    def test_non_esc_does_not_trigger_callback(
        self,
        listener: escape_listener.EscapeListener,
        on_escape: Mock,
    ) -> None:
        listener.on_press(keyboard.Key.enter)
        on_escape.assert_not_called()
