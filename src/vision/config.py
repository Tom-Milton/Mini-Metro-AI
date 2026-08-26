from src.domain import types

GAME_SPACE_HEIGHT_RATIO: float = 0.9  # Ignore the bottom UI toolbar
MASK_TOLERANCE: int = 3  # Color tolerance for masking (+/- from exact color)
STATION_COLOR: types.Color = (248, 248, 248)  # Station detection color (BGR)
LINE_COLOR: types.Color = (5, 204, 249)  # Yellow line color (BGR)
LINE_DILATION_SIZE: int = 65  # Join fragmented line segments
MAX_STATION_LINE_DISTANCE: int = 15  # Match stations near the skeleton line
POLYGON_EPSILON_RATIO: float = (
    0.02  # Polygon approximation epsilon ratio (epsilon * perimeter)
)
