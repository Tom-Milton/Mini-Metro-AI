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
LINE_COLOUR: types.Colour = (5, 204, 249)  # Yellow line colour (BGR)
LINE_MASK_TOLERANCE: int = 15  # Colour tolerance for line masking
LINE_DILATION_SIZE: int = 25  # Expand the mask to connect nearby line segments
MASK_CLOSING_KERNEL_SIZE: int = 50  # Fill small gaps and holes in the mask
MIN_SKELETON_COMPONENT_SIZE: int = 250  # Minimum size of retained components
MAX_STATION_LINE_DISTANCE: int = 5  # Match stations near the skeleton line

# Menu detection
MENU_AVAILABLE_DIAMETER_RATIO: float = 0.027  # Available button diameter / frame height
MENU_ACTIVE_DIAMETER_RATIO: float = 0.053  # Active button diameter / frame height
MENU_SELECTED_DIAMETER_RATIO: float = 0.095  # Selected button diameter / frame height
