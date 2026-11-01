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
        return skan.Skeleton(np.ones((1, 5), dtype=bool))

    @pytest.fixture
    def expected_polyline(self) -> types.Polyline:
        return np.array([[index, 0] for index in range(5)])

    def test_extract(
        self,
        extractor: polyline_extraction.SkeletonPolylineExtractor,
        input_skeleton: skan.Skeleton,
        expected_polyline: types.Polyline,
    ) -> None:
        polyline = extractor.extract(input_skeleton)
        assert np.array_equal(polyline, expected_polyline)


class TestEulerianPolylineExtractor:
    @pytest.fixture
    def extractor(self) -> polyline_extraction.EulerianPolylineExtractor:
        return polyline_extraction.EulerianPolylineExtractor()

    @pytest.fixture
    def input_skeleton(self) -> skan.Skeleton:
        frame = polygons.draw_t_shape(bar_length=7, spur_length=2)
        return skan.Skeleton(frame[:, :, 0] > 0)

    @pytest.fixture
    def expected_polyline(self) -> types.Polyline:
        bar1 = np.array([[x, 2] for x in range(0, 3)])
        spur = np.array([[3, y] for y in (2, 1, 0, 1, 2)])
        bar2 = np.array([[x, 2] for x in range(4, 7)])
        return np.concatenate((bar1, spur, bar2))

    def test_extract(
        self,
        extractor: polyline_extraction.EulerianPolylineExtractor,
        input_skeleton: skan.Skeleton,
        expected_polyline: types.Polyline,
    ) -> None:
        polyline = extractor.extract(input_skeleton)
        assert np.array_equal(polyline, expected_polyline)


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
        polyline = transform.transform(input_polylines)
        assert np.array_equal(polyline, expected_polyline)
