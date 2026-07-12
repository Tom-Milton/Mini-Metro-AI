from collections.abc import Callable

from pynput import keyboard


class EscapeListener:
    def __init__(self, on_escape: Callable[[], None]) -> None:
        self._on_escape = on_escape
        self._listener = keyboard.Listener(
            on_press=self._on_press,
        )

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def _on_press(
        self,
        key: keyboard.Key | keyboard.KeyCode | None,
    ) -> None:
        if key == keyboard.Key.esc:
            self._on_escape()
