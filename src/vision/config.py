from src.domain import types

STATION_COLOR: types.Color = (248, 248, 248)  # Station detection color (BGR)
LINE_COLOR: types.Color = (5, 207, 252)  # Yellow line color (BGR)
MASK_TOLERANCE: int = 3  # Color tolerance for masking (+/- from exact color)
POLYGON_EPSILON_RATIO: float = (
    0.02  # Polygon approximation epsilon ratio (epsilon * perimeter)
)
