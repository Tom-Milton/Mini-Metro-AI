import logging

from src.bot import runner
from src.automation import station_joiner
from src.infrastructure import dpi_awareness
from src.infrastructure import escape_listener
from src.infrastructure import screen_capture
from src.infrastructure import shutdown_manager
from src.vision import game_state_extraction
from src.vision import station_detection
from src.vision import line_detection


def run() -> None:
    dpi_awareness.enable_dpi_awareness()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    shutdown = shutdown_manager.ShutdownManager()
    app = runner.CaptureApp(
        screen=screen_capture.ScreenCapture(),
        extractor=game_state_extraction.GameStateExtractor(
            station_detector=station_detection.build_station_detector(),
            line_detector=line_detection.build_line_detector(),
        ),
        joiner=station_joiner.build_station_joiner(),
        keyboard=escape_listener.EscapeListener(on_escape=shutdown.stop),
        shutdown=shutdown,
    )
    app.run()


if __name__ == "__main__":
    run()
