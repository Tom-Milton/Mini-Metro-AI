import logging

from src.app import run_app
from src.infrastructure import escape_listener
from src.infrastructure import screen_capture
from src.infrastructure import shutdown_manager
from src.vision import game_state_extractor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    shutdown = shutdown_manager.ShutdownManager()
    app = run_app.CaptureApp(
        screen=screen_capture.ScreenCapture(),
        extractor=game_state_extractor.GameStateExtractor(),
        keyboard=escape_listener.EscapeListener(on_escape=shutdown.stop),
        shutdown=shutdown,
    )
    app.run()


if __name__ == "__main__":
    main()
