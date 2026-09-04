from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Callable, Sequence, cast

import networkx as nx
import numpy as np
import skan
import skimage.morphology

from src.domain import types
from src.vision import contour_extraction


# scikit-image does not expose a complete type signature for this function.
_typed_skeletonize = cast(
    Callable[[np.ndarray], np.ndarray],
    getattr(skimage.morphology, "skeletonize"),
)


class NoSkeletonError(Exception):
    """Raised when a mask has too few pixels to form a skan.Skeleton."""


class SkeletonExtractor(ABC):
    @abstractmethod
    def extract(self, frame: types.Frame) -> skan.Skeleton:
        pass


class MaskSkeletonExtractor(SkeletonExtractor):
    def __init__(self, mask_extractor: contour_extraction.MaskExtractor):
        self.mask_extractor = mask_extractor

    def extract(self, frame: types.Frame) -> skan.Skeleton:
        mask = self.mask_extractor.extract(frame)
        skeleton_image = _typed_skeletonize(mask > 0)
        if np.count_nonzero(skeleton_image) == 0:
            raise NoSkeletonError
        return skan.Skeleton(skeleton_image)


class LongestSkeletonExtraction(SkeletonExtractor):
    def __init__(self, skeleton_extractor: SkeletonExtractor):
        self.skeleton_extractor = skeleton_extractor

    def extract(self, frame: types.Frame) -> skan.Skeleton:
        skeleton = self.skeleton_extractor.extract(frame)
        graph = nx.from_scipy_sparse_array(skeleton.graph)
        longest_path_nodes = self._longest_path_nodes(graph)
        spur_paths = self._spur_paths(skeleton, longest_path_nodes)
        return skeleton.prune_paths(spur_paths)

    def _longest_path_nodes(self, graph: nx.Graph) -> set[int]:
        components = nx.connected_components(graph)
        path_sets = (self._longest_path(graph, component) for component in components)
        return set().union(*path_sets)

    def _spur_paths(
        self, skeleton: skan.Skeleton, longest_path_nodes: set[int]
    ) -> list[int]:
        return [
            i
            for i in range(skeleton.n_paths)
            if not longest_path_nodes.issuperset(skeleton.path(i))
        ]

    def _longest_path(self, graph: nx.Graph, component: set[int]) -> set[int]:
        subgraph = graph.subgraph(component)
        first_end = self._farthest_node(subgraph, next(iter(component)))
        second_end = self._farthest_node(subgraph, first_end)
        return set(nx.shortest_path(subgraph, first_end, second_end, weight="weight"))

    def _farthest_node(self, graph: nx.Graph, source: int) -> int:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        return max(lengths, key=lambda node: lengths[node])


class PolylineExtractor(ABC):
    @abstractmethod
    def extract(self, frame: types.Frame) -> Sequence[types.Polyline]:
        pass


class SkeletonPolylineExtractor(PolylineExtractor):
    def __init__(self, skeleton_extractor: SkeletonExtractor):
        self.skeleton_extractor = skeleton_extractor

    def extract(self, frame: types.Frame) -> Sequence[types.Polyline]:
        try:
            skeleton = self.skeleton_extractor.extract(frame)
        except NoSkeletonError:
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
