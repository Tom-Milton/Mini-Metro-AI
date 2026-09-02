import time

import pynput.mouse as mouse

from src.domain import types


class MouseController:
    def __init__(self, controller: mouse.Controller | None = None):
        self._mouse = controller or mouse.Controller()

    def press(self) -> None:
        self._mouse.press(mouse.Button.left)

    def release(self) -> None:
        self._mouse.release(mouse.Button.left)

    def move(self, position: types.Position) -> None:
        self._mouse.position = position

    def drag(
        self,
        position: types.Position,
        duration: float = 1.0,
    ) -> None:
        start_x, start_y = self._mouse.position
        end_x, end_y = position

        steps = int(duration * 60)

        for step in range(1, steps + 1):
            progress = step / steps
            intermediate_x = int(start_x + (end_x - start_x) * progress)
            intermediate_y = int(start_y + (end_y - start_y) * progress)
            self.move((intermediate_x, intermediate_y))
            time.sleep(duration / steps)

    def join(self, positions: list[types.Position]) -> None:
        self.move(positions[0])
        self.press()
        for position in positions[1:]:
            self.drag(position, duration=0.5)
        time.sleep(0.5)
        self.release()
