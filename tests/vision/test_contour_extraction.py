import pytest
from typing import TypedDict

from src.domain import types
from src.vision import config
from src.vision import contour_extraction
from tests.fixtures import polygons


class ContourCase(TypedDict):
    sides: int


TEST_CASES = [
    pytest.param(ContourCase(sides=100), id="circle"),
    pytest.param(ContourCase(sides=3), id="triangle"),
    pytest.param(ContourCase(sides=4), id="square"),
]


class TestContourExtractor:
    @pytest.fixture
    def extractor(self) -> contour_extraction.ContourExtractor:
        mask_extractor = contour_extraction.ColourMaskExtractor(
            color=config.STATION_COLOR,
            mask_tolerance=config.MASK_TOLERANCE,
        )
        return contour_extraction.ContourExtractor(
            mask_extractor=mask_extractor,
        )

    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request: pytest.FixtureRequest) -> ContourCase:
        return request.param

    @pytest.fixture
    def sides(self, test_case: ContourCase) -> int:
        return test_case["sides"]

    @pytest.fixture
    def input_frame(self, sides: int) -> types.Frame:
        return polygons.draw_polygon(sides=sides, color=config.STATION_COLOR)

    def test_extract(
        self,
        extractor: contour_extraction.ContourExtractor,
        input_frame: types.Frame,
    ) -> None:
        observed_contours = extractor.extract(input_frame)
        assert len(observed_contours) == 1
