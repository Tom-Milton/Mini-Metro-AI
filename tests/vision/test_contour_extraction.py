import numpy as np
import pytest
from typing import TypedDict

from src.domain import types
from src.vision import config
from src.vision import contour_extraction
from tests.fixtures import polygons


class TestMaskExtractor:
    __test__ = False

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        # Small regular polygons produce unintuitive skeletons, so use a 5x1 rectangle.
        return polygons.draw_rectangle(
            width=5,
            height=1,
            colour=config.LINE_COLOUR,
            frame_size=9,
        )

    def test_extract(
        self,
        extractor: contour_extraction.MaskExtractor,
        input_frame: types.Frame,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = extractor.extract(input_frame)
        assert np.array_equal(observed_mask, expected_mask)


class TestColourMaskExtractor(TestMaskExtractor):
    __test__ = True

    @pytest.fixture
    def extractor(self) -> contour_extraction.ColourMaskExtractor:
        return contour_extraction.ColourMaskExtractor(
            colour=config.LINE_COLOUR,
            mask_tolerance=0,
        )

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(255, 255, 255),
            frame_size=9,
        )[:, :, 0]


class TestDilatingMaskExtractor(TestMaskExtractor):
    __test__ = True

    @pytest.fixture
    def extractor(self) -> contour_extraction.DilatingMaskExtractor:
        return contour_extraction.DilatingMaskExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=config.LINE_COLOUR,
                mask_tolerance=0,
            ),
            dilation_size=3,
        )

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=7,
            height=3,
            colour=(255, 255, 255),
            frame_size=9,
        )[:, :, 0]


class TestSkeletonMaskExtractor(TestMaskExtractor):
    __test__ = True

    @pytest.fixture
    def extractor(self) -> contour_extraction.SkeletonMaskExtractor:
        return contour_extraction.SkeletonMaskExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=config.LINE_COLOUR,
                mask_tolerance=0,
            ),
        )

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(255, 255, 255),
            frame_size=9,
        )[:, :, 0]


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
            colour=config.STATION_COLOUR,
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
        return polygons.draw_polygon(sides=sides, colour=config.STATION_COLOUR)

    def test_extract(
        self,
        extractor: contour_extraction.ContourExtractor,
        input_frame: types.Frame,
    ) -> None:
        observed_contours = extractor.extract(input_frame)
        assert len(observed_contours) == 1
