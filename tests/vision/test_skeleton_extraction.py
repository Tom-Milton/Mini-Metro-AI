import numpy as np
import pytest
import skan

from src.domain import types
from src.vision import skeleton_extraction
from tests.fixtures import polygons


class TestMaskSkeletonExtractor:
    @pytest.fixture
    def extractor(self) -> skeleton_extraction.MaskSkeletonExtractor:
        return skeleton_extraction.MaskSkeletonExtractor()

    @pytest.fixture
    def input_mask(self) -> types.Mask:
        return polygons.draw_t_shape(bar_length=7, spur_length=2)[:, :, 0]

    @pytest.fixture
    def expected_skeleton(self) -> skan.Skeleton:
        frame = polygons.draw_t_shape(bar_length=7, spur_length=2)
        return skan.Skeleton(frame[:, :, 0] > 0)

    def test_extract(
        self,
        extractor: skeleton_extraction.MaskSkeletonExtractor,
        input_mask: types.Mask,
        expected_skeleton: skan.Skeleton,
    ) -> None:
        skeleton = extractor.extract(input_mask)
        assert np.array_equal(skeleton.coordinates, expected_skeleton.coordinates)


class TestLongestSkeletonTransformer:
    @pytest.fixture
    def transform(self) -> skeleton_extraction.LongestSkeletonTransformer:
        return skeleton_extraction.LongestSkeletonTransformer()

    @pytest.fixture
    def input_skeleton(self) -> skan.Skeleton:
        frame = polygons.draw_t_shape(bar_length=7, spur_length=2)
        return skan.Skeleton(frame[:, :, 0] > 0)

    @pytest.fixture
    def expected_skeleton(self) -> skan.Skeleton:
        frame = polygons.draw_t_shape(bar_length=7, spur_length=2)
        frame[:2, 7 // 2] = 0  # Remove the spur.
        return skan.Skeleton(frame[:, :, 0] > 0)

    def test_transform(
        self,
        transform: skeleton_extraction.LongestSkeletonTransformer,
        input_skeleton: skan.Skeleton,
        expected_skeleton: skan.Skeleton,
    ) -> None:
        skeleton = transform.transform(input_skeleton)
        assert np.array_equal(skeleton.coordinates, expected_skeleton.coordinates)
