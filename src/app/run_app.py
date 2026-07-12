import logging

import src.infrastructure.escape_listener as escape_listener
import src.infrastructure.screen_capture as screen_capture
import src.infrastructure.shutdown_manager as shutdown_manager
import src.vision.game_state_extractor as game_state_extractor

logger = logging.getLogger(__name__)


class CaptureApp:
    def __init__(
        self,
        screen: screen_capture.ScreenCapture,
        extractor: game_state_extractor.GameStateExtractor,
        keyboard: escape_listener.EscapeListener,
        shutdown: shutdown_manager.ShutdownManager,
    ) -> None:
        self._screen = screen
        self._extractor = extractor
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
            logger.info("Captured %d stations", len(state.stations))
