import cv2
import numpy as np

from src.domain import types


def draw_polygon(
    sides: int,
    centre: types.Position = (50, 50),
    radius: int = 50,
    colour: types.Colour = (255, 255, 255),
    frame_size: int = 100,
) -> types.Frame:
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    angles = np.linspace(-np.pi / 2, 2 * np.pi - np.pi / 2, sides, endpoint=False)
    points = np.column_stack(
        (
            centre[0] + radius * np.cos(angles),
            centre[1] + radius * np.sin(angles),
        )
    ).astype(np.int32)
    frame = cv2.fillPoly(frame, [points], colour)
    return frame


def draw_rectangle(
    width: int,
    height: int,
    colour: types.Colour = (255, 255, 255),
    padding: int = 0,
) -> types.Frame:
    frame_width = width + 2 * padding
    frame_height = height + 2 * padding
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[padding : padding + height, padding : padding + width] = colour
    return frame
