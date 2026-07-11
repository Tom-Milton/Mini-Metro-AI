from enum import Enum
from typing import TypeAlias, NamedTuple

import numpy as np


class ShapeProperties(NamedTuple):
    name: str
    vertices: int


class StationShape(Enum):
    CIRCLE = ShapeProperties(name="circle", vertices=0)
    TRIANGLE = ShapeProperties(name="triangle", vertices=3)
    SQUARE = ShapeProperties(name="square", vertices=4)

    @classmethod
    def from_label(cls, label: str) -> "StationShape":
        return next(m for m in cls if m.value.name == label)


Color: TypeAlias = tuple[int, int, int]  # (B, G, R) - BGR color value
Frame: TypeAlias = np.ndarray  # 3D array (height, width, 3 channels BGR)
Mask: TypeAlias = np.ndarray  # 2D array (height, width) - binary mask
Contour: TypeAlias = np.ndarray  # (N, 1, 2) - contour points
Position: TypeAlias = tuple[int, int]  # (x, y) coordinates
