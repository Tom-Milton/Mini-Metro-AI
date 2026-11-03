import logging

from src.automation import route_building
from src.automation import week_navigation
from src.infrastructure import keyboard_listener
from src.infrastructure import screen_capture
from src.infrastructure import shutdown_manager
from src.vision import game_state_extraction
from src.vision import week_detection

logger = logging.getLogger(__name__)


class CaptureApp:
    def __init__(
        self,
        screen: screen_capture.ScreenCapture,
        game_extractor: game_state_extraction.GameStateExtractor,
        week_detector: week_detection.WeekDetector,
        week_navigator: week_navigation.WeekNavigator,
        route_drawer: route_building.RouteDrawer,
        keyboard: keyboard_listener.KeyboardListener,
        shutdown: shutdown_manager.ShutdownManager,
    ) -> None:
        self._screen = screen
        self._game_extractor = game_extractor
        self._week_detector = week_detector
        self._week_navigator = week_navigator
        self._route_drawer = route_drawer
        self._keyboard = keyboard
        self._shutdown = shutdown
        self._paused = False

    def __enter__(self) -> "CaptureApp":
        self._shutdown.start()
        self._keyboard.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._keyboard.stop()
        self._shutdown.stop()

    def run(self, interval: float = 3.0) -> None:
        with self:
            logger.info("Capture running")
            self._run(interval)

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        logger.info("Paused: %s", self._paused)

    def _run(self, interval: float) -> None:
        while not self._shutdown.wait(interval):
            if not self._paused:
                self._capture()

    def _capture(self) -> None:
        frame = self._screen.capture()
        if self._week_detector.detect(frame):
            self._week_navigator.navigate(frame)
        else:
            state = self._game_extractor.extract(frame)
            self._route_drawer.draw(state)
