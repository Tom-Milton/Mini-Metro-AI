from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from src.domain import types
from src.vision import config


class WeekDetector(ABC):
    @abstractmethod
    def detect(self, frame: types.Frame) -> bool:
        pass


class TemplateWeekDetector(WeekDetector):
    def __init__(
        self,
        template: types.Mask,
        region: types.Region,
        match_threshold: float,
    ) -> None:
        self.template = template
        self.region = region
        self.match_threshold = match_threshold

    def detect(self, frame: types.Frame) -> bool:
        window = self._prepare_window(self._crop_window(frame))
        score = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED)
        return score.item() >= self.match_threshold

    def _crop_window(self, frame: types.Frame) -> types.Frame:
        height, width = frame.shape[:2]
        left, top, right, bottom = self.region
        return frame[
            round(top * height) : round(bottom * height),
            round(left * width) : round(right * width),
        ]

    def _prepare_window(self, window: types.Frame) -> types.Mask:
        grey_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grey_window, self.template.shape[::-1])


def build_week_detector() -> WeekDetector:
    template_path = Path(__file__).parent / "templates" / "week.png"
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    assert template is not None
    return TemplateWeekDetector(
        template=template,
        region=config.WEEK_REGION,
        match_threshold=config.WEEK_MATCH_THRESHOLD,
    )
