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
    def expected_bgr_array(self) -> types.Frame:
        return np.array([[[30, 20, 10]]], dtype=np.uint8)

    @pytest.fixture
    def capture(self, input_rgb_image: Image.Image) -> screen_capture.ScreenCapture:
        return screen_capture.ScreenCapture(screen_grab=lambda: input_rgb_image)

    def test_capture_converts_rgb_image_to_bgr_array(
        self, capture: screen_capture.ScreenCapture, expected_bgr_array: types.Frame
    ) -> None:
        observed_frame = capture.capture()
        assert np.array_equal(observed_frame, expected_bgr_array)
