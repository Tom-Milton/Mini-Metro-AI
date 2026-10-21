import numpy as np
import pytest
import skan

from src.domain import types
from src.vision import contour_extraction
from src.vision import skeleton_extraction
from tests.fixtures import polygons


class TestMaskSkeletonExtractor:
    @pytest.fixture
    def extractor(self) -> skeleton_extraction.MaskSkeletonExtractor:
        return skeleton_extraction.MaskSkeletonExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=(100, 100, 100),
                mask_tolerance=0,
            ),
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        # Rectangles do not skeletonise predictably, so use a one-pixel-wide case.
        return polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(100, 100, 100),
            padding=3,
        )

    @pytest.fixture
    def expected_skeleton(self) -> skan.Skeleton:
        expected_image = polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(255, 255, 255),
            padding=3,
        )[:, :, 0]
        return skan.Skeleton(expected_image > 0)

    def test_extract(
        self,
        extractor: skeleton_extraction.MaskSkeletonExtractor,
        input_frame: types.Frame,
        expected_skeleton: skan.Skeleton,
    ) -> None:
        skeleton = extractor.extract(input_frame)
        assert np.array_equal(skeleton.coordinates, expected_skeleton.coordinates)


class TestLongestSkeletonExtraction:
    @pytest.fixture
    def extractor(self) -> skeleton_extraction.LongestSkeletonExtraction:
        return skeleton_extraction.LongestSkeletonExtraction(
            skeleton_extractor=skeleton_extraction.MaskSkeletonExtractor(
                mask_extractor=contour_extraction.ColourMaskExtractor(
                    colour=(100, 100, 100),
                    mask_tolerance=0,
                ),
            ),
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        rectangle = polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(100, 100, 100),
            padding=3,
        )
        rectangle[2, 5] = (100, 100, 100)
        return rectangle

    @pytest.fixture
    def expected_skeleton(self) -> skan.Skeleton:
        expected_image = polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(255, 255, 255),
            padding=3,
        )[:, :, 0]
        return skan.Skeleton(expected_image > 0)

    def test_extract(
        self,
        extractor: skeleton_extraction.LongestSkeletonExtraction,
        input_frame: types.Frame,
        expected_skeleton: skan.Skeleton,
    ) -> None:
        skeleton = extractor.extract(input_frame)
        assert np.array_equal(skeleton.coordinates, expected_skeleton.coordinates)
