import numpy as np
from collections.abc import Callable
from PIL import Image, ImageGrab

from src.domain import types

ScreenGrab = Callable[[], Image.Image]


class ScreenCapture:
    def __init__(self, screen_grab: ScreenGrab | None = None) -> None:
        self._screen_grab = screen_grab or ImageGrab.grab

    def capture(self) -> types.Frame:
        image = self._screen_grab()
        frame = np.asarray(image)
        return frame[..., ::-1]
