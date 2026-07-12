import signal
import threading


class ShutdownManager:
    def __init__(self) -> None:
        self._event = threading.Event()

    def start(self) -> None:
        signal.signal(signal.SIGINT, self.stop)

        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_: object) -> None:
        self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)
