from src.domain import types

# General
GAME_SPACE_HEIGHT_RATIO: float = 0.9  # Ignore the bottom UI toolbar
MIN_CONTOUR_AREA: float = 5.0  # Discard noisy contours smaller than this

# Station detection
STATION_COLOUR: types.Colour = (248, 248, 248)  # Station detection colour (BGR)
STATION_MASK_TOLERANCE: int = 5  # Colour tolerance for station masking
POLYGON_EPSILON_RATIO: float = (
    0.02  # Polygon approximation epsilon ratio (epsilon * perimeter)
)

# Line detection
LINE_COLOURS: dict[types.LineColour, types.Colour] = {
    types.LineColour.YELLOW: (5, 204, 249),
    types.LineColour.RED: (35, 45, 233),
    types.LineColour.BLUE: (146, 63, 28),
}
LINE_MASK_TOLERANCE: int = 15  # Colour tolerance for line masking
LINE_DILATION_SIZE: int = 25  # Expand the mask to connect nearby line segments
MASK_OPENING_KERNEL_SIZE: int = 3  # Remove thin anti-aliasing artefacts
MASK_CLOSING_KERNEL_SIZE: int = 50  # Fill small gaps and holes in the mask
MAX_STATION_LINE_DISTANCE: int = 10  # Match stations near the skeleton line

# Menu detection
MENU_DIAMETER_RATIOS: dict[types.MenuState, float] = {
    types.MenuState.AVAILABLE: 0.027,
    types.MenuState.ACTIVE: 0.053,
    types.MenuState.SELECTED: 0.095,
}
