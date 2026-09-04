import numpy as np
import pytest
import skan

from src.domain import types
from src.vision import contour_extraction
from src.vision import polyline_extraction
from tests.fixtures import polygons


class TestMaskSkeletonExtractor:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.MaskSkeletonExtractor:
        return polyline_extraction.MaskSkeletonExtractor(
            mask_extractor=contour_extraction.ColourMaskExtractor(
                colour=(100, 100, 100),
                mask_tolerance=0,
            ),
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        # Rectangles do not skeletonise predictably, so we use a case that
        # is already one pixel wide and requires no skeletonisation.
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
        extractor: polyline_extraction.MaskSkeletonExtractor,
        input_frame: types.Frame,
        expected_skeleton: skan.Skeleton,
    ) -> None:
        skeleton = extractor.extract(input_frame)
        assert np.array_equal(skeleton.coordinates, expected_skeleton.coordinates)


class TestLongestSkeletonExtraction:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.LongestSkeletonExtraction:
        return polyline_extraction.LongestSkeletonExtraction(
            skeleton_extractor=polyline_extraction.MaskSkeletonExtractor(
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
        extractor: polyline_extraction.LongestSkeletonExtraction,
        input_frame: types.Frame,
        expected_skeleton: skan.Skeleton,
    ) -> None:
        skeleton = extractor.extract(input_frame)
        assert np.array_equal(skeleton.coordinates, expected_skeleton.coordinates)


class TestPolylineExtractor:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.SkeletonPolylineExtractor:
        return polyline_extraction.SkeletonPolylineExtractor(
            skeleton_extractor=polyline_extraction.MaskSkeletonExtractor(
                mask_extractor=contour_extraction.ColourMaskExtractor(
                    colour=(100, 100, 100),
                    mask_tolerance=0,
                ),
            ),
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        return polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(100, 100, 100),
            padding=3,
        )

    @pytest.fixture
    def expected_polyline(self) -> types.Polyline:
        return np.array([[3, 3], [4, 3], [5, 3], [6, 3], [7, 3]])

    def test_extract(
        self,
        extractor: polyline_extraction.SkeletonPolylineExtractor,
        input_frame: types.Frame,
        expected_polyline: types.Polyline,
    ) -> None:
        polylines = extractor.extract(input_frame)
        assert np.array_equal(polylines[0], expected_polyline)


class TestGreedyPolylineExtractor:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.GreedyPolylineExtractor:
        return polyline_extraction.GreedyPolylineExtractor(
            polyline_extractor=polyline_extraction.SkeletonPolylineExtractor(
                skeleton_extractor=polyline_extraction.MaskSkeletonExtractor(
                    mask_extractor=contour_extraction.ColourMaskExtractor(
                        colour=(100, 100, 100),
                        mask_tolerance=0,
                    ),
                ),
            ),
        )

    @pytest.fixture
    def input_frame(self) -> types.Frame:
        segment = polygons.draw_rectangle(
            width=5,
            height=1,
            colour=(100, 100, 100),
        )
        gap = np.zeros((1, 1, 3), dtype=np.uint8)
        return np.hstack((segment, gap, segment))

    @pytest.fixture
    def expected_polyline(self) -> types.Polyline:
        return np.array([[index, 0] for index in range(11)])

    def test_extract(
        self,
        extractor: polyline_extraction.GreedyPolylineExtractor,
        input_frame: types.Frame,
        expected_polyline: types.Polyline,
    ) -> None:
        polylines = extractor.extract(input_frame)
        assert np.array_equal(polylines[0], expected_polyline) or np.array_equal(
            polylines[0], expected_polyline[::-1]
        )
