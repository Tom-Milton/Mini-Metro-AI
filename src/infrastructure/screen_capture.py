import numpy as np
from PIL import ImageGrab

from src.domain import types


class ScreenCapture:

    def capture(self) -> types.Frame:
        image = ImageGrab.grab()
        frame = np.asarray(image)
        return frame[..., ::-1]
