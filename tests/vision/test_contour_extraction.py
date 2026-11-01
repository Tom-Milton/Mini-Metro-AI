import pytest
from typing import TypedDict

from src.domain import types
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
    def extractor(self) -> contour_extraction.MaskContourExtractor:
        return contour_extraction.MaskContourExtractor(min_area=5.0)

    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request: pytest.FixtureRequest) -> ContourCase:
        return request.param

    @pytest.fixture
    def sides(self, test_case: ContourCase) -> int:
        return test_case["sides"]

    @pytest.fixture
    def input_mask(self, sides: int) -> types.Mask:
        return polygons.draw_polygon(sides=sides, colour=(255, 255, 255))[:, :, 0]

    def test_extract(
        self,
        extractor: contour_extraction.MaskContourExtractor,
        input_mask: types.Mask,
    ) -> None:
        observed_contours = extractor.extract(input_mask)
        assert len(observed_contours) == 1
