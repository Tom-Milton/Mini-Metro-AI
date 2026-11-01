import numpy as np
import pytest

from src.domain import types
from src.vision import mask_extraction
from tests.fixtures import polygons


class TestColourMaskExtractor:
    @pytest.fixture
    def extractor(self) -> mask_extraction.ColourMaskExtractor:
        return mask_extraction.ColourMaskExtractor(
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
        extractor: mask_extraction.ColourMaskExtractor,
        input_frame: types.Frame,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = extractor.extract(input_frame)
        assert np.array_equal(observed_mask, expected_mask)


class TestDilatingMaskTransformer:
    @pytest.fixture
    def transform(self) -> mask_extraction.DilatingMaskTransformer:
        return mask_extraction.DilatingMaskTransformer(dilation_size=3)

    @pytest.fixture
    def input_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=3,
            height=3,
            colour=(255, 255, 255),
            padding=3,
        )[:, :, 0]

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=5,
            height=5,
            colour=(255, 255, 255),
            padding=2,
        )[:, :, 0]

    def test_transform(
        self,
        transform: mask_extraction.DilatingMaskTransformer,
        input_mask: types.Mask,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = transform.transform(input_mask)
        assert np.array_equal(observed_mask, expected_mask)


class TestMorphologicalClosingMaskTransformer:
    @pytest.fixture
    def transform(self) -> mask_extraction.MorphologicalClosingMaskTransformer:
        return mask_extraction.MorphologicalClosingMaskTransformer(kernel_size=3)

    @pytest.fixture
    def input_mask(self) -> types.Mask:
        rectangle = polygons.draw_rectangle(
            width=3,
            height=3,
            colour=(255, 255, 255),
        )[:, :, 0]
        gap = np.zeros((3, 1), dtype=np.uint8)
        return np.hstack((rectangle, gap, rectangle))

    @pytest.fixture
    def expected_mask(self) -> types.Mask:
        return polygons.draw_rectangle(
            width=7,
            height=3,
            colour=(255, 255, 255),
        )[:, :, 0]

    def test_transform(
        self,
        transform: mask_extraction.MorphologicalClosingMaskTransformer,
        input_mask: types.Mask,
        expected_mask: types.Mask,
    ) -> None:
        observed_mask = transform.transform(input_mask)
        assert np.array_equal(observed_mask, expected_mask)
