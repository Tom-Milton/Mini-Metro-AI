import pytest
import numpy as np
from PIL import Image

from src.infrastructure import screen_capture
from src.domain import types


class TestScreenCapture:
    @pytest.fixture
    def input_rgb_image(self) -> Image.Image:
        return Image.new("RGB", (1, 1), (10, 20, 30))

    @pytest.fixture
    def display_size(self, input_rgb_image: Image.Image) -> types.DisplaySize:
        return input_rgb_image.size

    @pytest.fixture
    def expected_bgr_array(self, display_size: types.DisplaySize) -> types.Frame:
        pixel = np.array([30, 20, 10], dtype=np.uint8)
        shape = (display_size[1], display_size[0], 3)
        return np.full(shape, pixel)

    @pytest.fixture
    def capture(
        self, input_rgb_image: Image.Image, display_size: types.DisplaySize
    ) -> screen_capture.ScreenCapture:
        return screen_capture.ScreenCapture(
            screen_grab=lambda: input_rgb_image,
            display_size=display_size,
        )

    @pytest.mark.parametrize("display_size", [(1, 1), (2, 2)])
    def test_capture(
        self, capture: screen_capture.ScreenCapture, expected_bgr_array: types.Frame
    ) -> None:
        observed_frame = capture.capture()
        assert np.array_equal(observed_frame, expected_bgr_array)
