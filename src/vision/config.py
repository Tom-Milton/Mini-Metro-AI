from src.domain import types

GAME_SPACE_HEIGHT_RATIO: float = 0.9  # Ignore the bottom UI toolbar
MASK_TOLERANCE: int = 5  # Colour tolerance for masking (+/- from exact colour)
MIN_CONTOUR_AREA: float = 5.0  # Discard noisy contours smaller than this
STATION_COLOUR: types.Colour = (248, 248, 248)  # Station detection colour (BGR)
LINE_COLOUR: types.Colour = (5, 204, 249)  # Yellow line colour (BGR)
LINE_DILATION_SIZE: int = 25  # Join fragmented line segments
MAX_STATION_LINE_DISTANCE: int = 50  # Match stations near the skeleton line
POLYGON_EPSILON_RATIO: float = (
    0.02  # Polygon approximation epsilon ratio (epsilon * perimeter)
)
