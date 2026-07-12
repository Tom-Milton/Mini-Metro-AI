import logging

import src.app.run_app as run_app
import src.infrastructure.escape_listener as escape_listener
import src.infrastructure.screen_capture as screen_capture
import src.infrastructure.shutdown_manager as shutdown_manager
import src.vision.game_state_extractor as game_state_extractor


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
