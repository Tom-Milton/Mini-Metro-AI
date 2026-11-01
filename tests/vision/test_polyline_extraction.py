import numpy as np
import pytest
import skan

from src.domain import types
from src.vision import polyline_extraction
from tests.fixtures import polygons


class TestPolylineExtractor:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.SkeletonPolylineExtractor:
        return polyline_extraction.SkeletonPolylineExtractor()

    @pytest.fixture
    def input_skeleton(self) -> skan.Skeleton:
        frame = polygons.draw_t_shape(bar_length=5, spur_length=2)
        return skan.Skeleton(frame[:, :, 0] > 0)

    @pytest.fixture
    def expected_polylines(self) -> list[types.Polyline]:
        return [
            np.array([[2, 0], [2, 1], [2, 2]]),  # Spur.
            np.array([[0, 2], [1, 2], [2, 2]]),  # Left bar.
            np.array([[2, 2], [3, 2], [4, 2]]),  # Right bar.
        ]

    def test_extract(
        self,
        extractor: polyline_extraction.SkeletonPolylineExtractor,
        input_skeleton: skan.Skeleton,
        expected_polylines: list[types.Polyline],
    ) -> None:
        polylines = extractor.extract(input_skeleton)
        assert np.array_equal(polylines, expected_polylines)


class TestGreedyPolylineTransformer:
    @pytest.fixture
    def transform(self) -> polyline_extraction.GreedyPolylineTransformer:
        return polyline_extraction.GreedyPolylineTransformer()

    @pytest.fixture
    def input_polylines(self) -> list[types.Polyline]:
        return [
            np.array([[index, 0] for index in range(0, 5)]),
            np.array([[index, 0] for index in range(6, 11)]),
        ]

    @pytest.fixture
    def expected_polyline(self) -> types.Polyline:
        return np.array([[index, 0] for index in range(11)])

    def test_transform(
        self,
        transform: polyline_extraction.GreedyPolylineTransformer,
        input_polylines: list[types.Polyline],
        expected_polyline: types.Polyline,
    ) -> None:
        polylines = transform.transform(input_polylines)
        assert np.array_equal(polylines[0], expected_polyline) or np.array_equal(
            polylines[0], expected_polyline[::-1]
        )  # Either direction.
