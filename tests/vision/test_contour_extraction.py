import numpy as np
import pytest
from typing import TypedDict

from src.domain import types
from src.vision import contour_extraction
from tests.fixtures import polygons


class TestColourMaskExtractor:
    @pytest.fixture
    def extractor(self) -> contour_extraction.ColourMaskExtractor:
        return contour_extraction.ColourMaskExtractor(
            colour=(100, 100, 100),
            mask_tolerance=0,
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        return polygons.draw_rectangle(
            width=3,
            height=3,
            colour=(100, 100, 100),
            padding=3,
        )

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=3,
            height=3,
            colour=(255, 255, 255),
            padding=3,
        )[:, :, 0]

    def test_extract(
        self,
        extractor: contour_extraction.ColourMaskExtractor,
        input_frame: types.Frame,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = extractor.extract(input_frame)
        assert np.array_equal(observed_mask, expected_mask)


class TestDilatingMaskExtractor:
    @pytest.fixture
    def extractor(self) -> contour_extraction.DilatingMaskExtractor:
        return contour_extraction.DilatingMaskExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=(100, 100, 100),
                mask_tolerance=0,
            ),
            dilation_size=3,
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        return polygons.draw_rectangle(
            width=3,
            height=3,
            colour=(100, 100, 100),
            padding=3,
        )

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=5,
            height=5,
            colour=(255, 255, 255),
            padding=2,
        )[:, :, 0]

    def test_extract(
        self,
        extractor: contour_extraction.DilatingMaskExtractor,
        input_frame: types.Frame,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = extractor.extract(input_frame)
        assert np.array_equal(observed_mask, expected_mask)


class TestMorphologicalClosingMaskExtractor:
    @pytest.fixture
    def extractor(self) -> contour_extraction.MorphologicalClosingMaskExtractor:
        return contour_extraction.MorphologicalClosingMaskExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=(100, 100, 100),
                mask_tolerance=0,
            ),
            kernel_size=3,
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        rectangle = polygons.draw_rectangle(
            width=3,
            height=3,
            colour=(100, 100, 100),
        )
        gap = np.zeros((3, 1, 3), dtype=np.uint8)
        return np.hstack((rectangle, gap, rectangle))

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=7,
            height=3,
            colour=(255, 255, 255),
        )[:, :, 0]

    def test_extract(
        self,
        extractor: contour_extraction.MorphologicalClosingMaskExtractor,
        input_frame: types.Frame,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = extractor.extract(input_frame)
        assert np.array_equal(observed_mask, expected_mask)


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
        return contour_extraction.MaskContourExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=(100, 100, 100),
                mask_tolerance=5,
            ),
            min_area=5.0,
        )

    @pytest.fixture(params=TEST_CASES)
    def test_case(self, request: pytest.FixtureRequest) -> ContourCase:
        return request.param

    @pytest.fixture
    def sides(self, test_case: ContourCase) -> int:
        return test_case["sides"]

    @pytest.fixture
    def input_frame(self, sides: int) -> types.Frame:
        return polygons.draw_polygon(sides=sides, colour=(100, 100, 100))

    def test_extract(
        self,
        extractor: contour_extraction.MaskContourExtractor,
        input_frame: types.Frame,
    ) -> None:
        observed_contours = extractor.extract(input_frame)
        assert len(observed_contours) == 1
