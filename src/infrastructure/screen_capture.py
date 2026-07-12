import numpy as np
from PIL import ImageGrab

import src.domain.types as types


class ScreenCapture:

    def capture(self) -> types.Frame:
        image = ImageGrab.grab()
        frame = np.asarray(image)
        return frame[..., ::-1]
