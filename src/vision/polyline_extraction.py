from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Sequence

import numpy as np

from src.domain import types
from src.vision import skeleton_extraction


class PolylineExtractor(ABC):
    @abstractmethod
    def extract(self, frame: types.Frame) -> Sequence[types.Polyline]:
        pass


class SkeletonPolylineExtractor(PolylineExtractor):
    def __init__(self, skeleton_extractor: skeleton_extraction.SkeletonExtractor):
        self.skeleton_extractor = skeleton_extractor

    def extract(self, frame: types.Frame) -> Sequence[types.Polyline]:
        try:
            skeleton = self.skeleton_extractor.extract(frame)
        except skeleton_extraction.NoSkeletonError:
            return []
        paths = (skeleton.path_coordinates(index) for index in range(skeleton.n_paths))
        # Convert Skan's (row, column) coordinates to the project's (x, y) order.
        return [path[:, ::-1] for path in paths]


class GreedyPolylineExtractor(PolylineExtractor):
    def __init__(self, polyline_extractor: PolylineExtractor):
        self.polyline_extractor = polyline_extractor

    def extract(self, frame: types.Frame) -> Sequence[types.Polyline]:
        lines = list(self.polyline_extractor.extract(frame))
        while len(lines) > 1:
            first_index, second_index = self._find_closest_pair(lines)
            second = lines.pop(second_index)
            first = lines.pop(first_index)
            lines.append(self._merge_lines(first, second))
        return lines

    def _find_closest_pair(
        self,
        lines: list[types.Polyline],
    ) -> tuple[int, int]:
        return min(
            combinations(range(len(lines)), 2),
            key=lambda pair: self._calculate_pair_distance(pair, lines),
        )

    def _calculate_pair_distance(
        self,
        pair: tuple[int, int],
        lines: list[types.Polyline],
    ) -> float:
        first, second = self._orient_pair(
            lines[pair[0]],
            lines[pair[1]],
        )
        return self._distance(first[-1], second[0])

    def _orient_pair(
        self,
        first: types.Polyline,
        second: types.Polyline,
    ) -> tuple[types.Polyline, types.Polyline]:
        return min(
            product(self._orientations(first), self._orientations(second)),
            key=lambda lines: (
                self._distance(lines[0][-1], lines[1][0]),
                tuple(lines[0][0]),
                tuple(lines[1][-1]),
            ),
        )

    def _merge_lines(
        self,
        first: types.Polyline,
        second: types.Polyline,
    ) -> types.Polyline:
        first, second = self._orient_pair(first, second)
        bridge = self._bridge(first[-1], second[0])
        return np.vstack((first, bridge, second))

    def _orientations(
        self,
        line: types.Polyline,
    ) -> tuple[types.Polyline, types.Polyline]:
        return line, line[::-1]

    def _bridge(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        distance = int(self._distance(start, end))
        return np.linspace(start, end, distance + 1, dtype=np.int32)[1:-1]

    def _distance(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second))
