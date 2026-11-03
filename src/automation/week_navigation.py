from abc import ABC, abstractmethod

from src.domain import types
from src.infrastructure import mouse_controller


class WeekNavigator(ABC):
    @abstractmethod
    def navigate(self, frame: types.Frame) -> None:
        pass


class CentreClickWeekNavigator(WeekNavigator):
    def __init__(self, mouse: mouse_controller.MouseController) -> None:
        self._mouse = mouse

    def navigate(self, frame: types.Frame) -> None:
        height, width = frame.shape[:2]
        centre = (width // 2, height // 2)
        # Click through both screens while reward selection is not implemented.
        self._mouse.click(centre)
        self._mouse.click(centre)


def build_week_navigator() -> WeekNavigator:
    return CentreClickWeekNavigator(mouse=mouse_controller.MouseController())
