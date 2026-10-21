import numpy as np
import pytest

from src.domain import types
from src.vision import contour_extraction
from src.vision import polyline_extraction
from src.vision import skeleton_extraction
from tests.fixtures import polygons


class TestPolylineExtractor:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.SkeletonPolylineExtractor:
        return polyline_extraction.SkeletonPolylineExtractor(
            skeleton_extractor=skeleton_extraction.MaskSkeletonExtractor(
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
                skeleton_extractor=skeleton_extraction.MaskSkeletonExtractor(
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
