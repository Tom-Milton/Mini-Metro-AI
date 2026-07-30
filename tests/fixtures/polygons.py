import cv2
import numpy as np

from src.domain import types


def draw_polygon(
    sides: int,
    center: types.Position = (50, 50),
    radius: int = 50,
    color: types.Color = (255, 255, 255),
    frame_size: int = 100,
) -> types.Frame:
    img = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    angles = np.linspace(-np.pi / 2, 2 * np.pi - np.pi / 2, sides, endpoint=False)
    points = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    ).astype(np.int32)
    cv2.fillPoly(img, [points], color)
    return img
