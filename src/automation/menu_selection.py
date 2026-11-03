import logging
from abc import ABC, abstractmethod

from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller


logger = logging.getLogger(__name__)


class MenuSelector(ABC):
    @abstractmethod
    def select(self, menu: models.Menu) -> None:
        pass

    @abstractmethod
    def deselect(self, menu: models.Menu) -> None:
        pass


class LineMenuSelector(MenuSelector):
    def __init__(self, mouse: mouse_controller.MouseController):
        self._mouse = mouse

    def select(self, menu: models.Menu) -> None:
        logger.info("Selected %s line", menu.line_id.value)
        if menu.state == types.MenuState.SELECTED:
            return
        self._mouse.click(menu.position)
        menu.state = types.MenuState.SELECTED

    def deselect(self, menu: models.Menu) -> None:
        logger.info("Deselected %s line", menu.line_id.value)
        if menu.state != types.MenuState.SELECTED:
            return
        self._mouse.click(menu.position)
        menu.state = types.MenuState.ACTIVE


def build_menu_selector() -> MenuSelector:
    return LineMenuSelector(mouse=mouse_controller.MouseController())
