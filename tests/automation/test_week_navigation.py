from unittest.mock import Mock, call

import numpy as np
import pytest

from src.automation import week_navigation
from src.domain import types
from src.infrastructure import mouse_controller


class TestCentreClickWeekNavigator:
    @pytest.fixture
    def mock_mouse(self) -> Mock:
        return Mock(spec=mouse_controller.MouseController)

    @pytest.fixture
    def navigator(self, mock_mouse: Mock) -> week_navigation.WeekNavigator:
        return week_navigation.CentreClickWeekNavigator(mouse=mock_mouse)

    @pytest.fixture
    def frame(self) -> types.Frame:
        return np.zeros((100, 200, 3), dtype=np.uint8)

    @pytest.fixture
    def expected_calls(self) -> list[object]:
        return [call.click((100, 50)), call.click((100, 50))]

    def test_navigate(
        self,
        navigator: week_navigation.WeekNavigator,
        mock_mouse: Mock,
        frame: types.Frame,
        expected_calls: list[object],
    ) -> None:
        navigator.navigate(frame)
        assert mock_mouse.mock_calls == expected_calls
