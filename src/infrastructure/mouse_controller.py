import time
from types import ModuleType

import pyautogui

from src.domain import types


class MouseController:
    def __init__(self, controller: ModuleType = pyautogui) -> None:
        self._mouse = controller

    def join(self, positions: list[types.Position]) -> None:
        self._mouse.moveTo(*positions[0])
        self._mouse.mouseDown(button="left")
        for position in positions[1:]:
            self._mouse.dragTo(
                *position, duration=0.1, button="left", mouseDownUp=False
            )
        time.sleep(0.1)
        self._mouse.mouseUp(button="left")
