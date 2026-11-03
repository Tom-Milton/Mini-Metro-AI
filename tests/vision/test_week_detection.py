from pathlib import Path

import cv2
import pytest

from src.domain import types
from src.vision import week_detection


class TestTemplateWeekDetector:
    @pytest.fixture
    def detector(self) -> week_detection.WeekDetector:
        return week_detection.build_week_detector()

    @pytest.fixture
    def frame(self) -> types.Frame:
        fixtures = Path(__file__).parent.parent / "fixtures"
        frame = cv2.imread(str(fixtures / "week.png"))
        assert frame is not None
        return frame

    def test_detect(
        self,
        detector: week_detection.WeekDetector,
        frame: types.Frame,
    ) -> None:
        assert detector.detect(frame)
