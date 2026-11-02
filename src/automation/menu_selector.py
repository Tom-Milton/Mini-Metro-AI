import logging

from src.domain import models
from src.domain import types
from src.infrastructure import mouse_controller


logger = logging.getLogger(__name__)


class MenuSelector:
    def __init__(self, mouse: mouse_controller.MouseController):
        self._mouse = mouse

    def select(self, menu: models.Menu) -> None:
        logger.info("Selected %s line", menu.line_id.value)
        if menu.state == types.MenuState.SELECTED:
            return
        self._mouse.click(menu.position)


def build_menu_selector() -> MenuSelector:
    return MenuSelector(mouse=mouse_controller.MouseController())
