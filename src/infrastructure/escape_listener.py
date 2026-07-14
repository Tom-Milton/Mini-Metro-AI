from collections.abc import Callable
from typing import TypeAlias

from pynput import keyboard

KeyPress: TypeAlias = keyboard.Key | keyboard.KeyCode | None


class EscapeListener:
    def __init__(
        self,
        on_escape: Callable[[], None],
        listener: keyboard.Listener | None = None,
    ) -> None:
        self._on_escape = on_escape
        self._listener = listener or keyboard.Listener(
            on_press=self.on_press,
        )

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def on_press(
        self,
        key: KeyPress,
    ) -> None:
        if key == keyboard.Key.esc:
            self._on_escape()
