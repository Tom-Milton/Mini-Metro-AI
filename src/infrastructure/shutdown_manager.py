import signal
import threading
from collections.abc import Callable
from typing import Any


SignalRegistrar = Callable[..., Any]


class ShutdownManager:
    def __init__(
        self,
        event: threading.Event | None = None,
        register_signal: SignalRegistrar | None = None,
    ) -> None:
        self._event = event or threading.Event()
        self._register_signal = register_signal or signal.signal

    def start(self) -> None:
        self._register_signal(signal.SIGINT, self.stop)
        self._register_signal(signal.SIGTERM, self.stop)

    def stop(self, *_: object) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)
