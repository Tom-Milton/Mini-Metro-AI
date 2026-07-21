import logging

from src.bot import runner
from src.automation import station_joiner
from src.infrastructure import dpi_awareness
from src.infrastructure import escape_listener
from src.infrastructure import screen_capture
from src.infrastructure import shutdown_manager
from src.vision import game_state_extractor


def run() -> None:
    dpi_awareness.enable_dpi_awareness()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    shutdown = shutdown_manager.ShutdownManager()
    app = runner.CaptureApp(
        screen=screen_capture.ScreenCapture(),
        extractor=game_state_extractor.GameStateExtractor(),
        joiner=station_joiner.StationJoiner(None),
        keyboard=escape_listener.EscapeListener(on_escape=shutdown.stop),
        shutdown=shutdown,
    )
    app.run()


if __name__ == "__main__":
    run()
