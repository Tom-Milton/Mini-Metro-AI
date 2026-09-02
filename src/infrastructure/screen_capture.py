import ctypes
import importlib
import sys
from typing import Any

import numpy as np
from collections.abc import Callable
from PIL import Image, ImageGrab

from src.domain import types

ScreenGrab = Callable[[], Image.Image]


def mac_display_size() -> types.DisplaySize:
    quartz: Any = importlib.import_module("Quartz")
    display_id = quartz.CGMainDisplayID()
    return (
        quartz.CGDisplayPixelsWide(display_id),
        quartz.CGDisplayPixelsHigh(display_id),
    )


def windows_display_size() -> types.DisplaySize:
    user32 = getattr(ctypes, "windll").user32
    return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))


def resolve_display_size() -> types.DisplaySize:
    """Default display-size resolver for the current platform."""
    if sys.platform == "darwin":
        return mac_display_size()
    elif sys.platform == "win32":
        return windows_display_size()
    else:
        raise NotImplementedError("Display size not implemented for this platform.")


class ScreenCapture:
    def __init__(
        self,
        screen_grab: ScreenGrab | None = None,
        display_size: types.DisplaySize | None = None,
    ) -> None:
        self._screen_grab = screen_grab or ImageGrab.grab
        self._display_size = display_size or resolve_display_size()

    def capture(self) -> types.Frame:
        image = self._screen_grab().convert("RGB")
        image = image.resize(self._display_size, Image.Resampling.LANCZOS)
        frame = np.asarray(image)
        return frame[..., ::-1]
