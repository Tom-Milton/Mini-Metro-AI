import logging

from src.automation import route_building
from src.infrastructure import escape_listener
from src.infrastructure import screen_capture
from src.infrastructure import shutdown_manager
from src.vision import game_state_extraction

logger = logging.getLogger(__name__)


class CaptureApp:
    def __init__(
        self,
        screen: screen_capture.ScreenCapture,
        extractor: game_state_extraction.GameStateExtractor,
        route_drawer: route_building.RouteDrawer,
        keyboard: escape_listener.EscapeListener,
        shutdown: shutdown_manager.ShutdownManager,
    ) -> None:
        self._screen = screen
        self._extractor = extractor
        self._route_drawer = route_drawer
        self._keyboard = keyboard
        self._shutdown = shutdown

    def __enter__(self) -> "CaptureApp":
        self._shutdown.start()
        self._keyboard.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._keyboard.stop()
        self._shutdown.stop()

    def run(self, interval: float = 5.0) -> None:
        with self:
            logger.info("Capture running")
            self._run(interval)

    def _run(self, interval: float) -> None:
        while not self._shutdown.wait(interval):
            frame = self._screen.capture()
            state = self._extractor.extract(frame)
            self._route_drawer.draw(state)
