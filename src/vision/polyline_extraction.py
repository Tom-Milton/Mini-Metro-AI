from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Sequence

import networkx as nx
import numpy as np
import skan

from src.domain import types


class PolylineExtractor(ABC):
    @abstractmethod
    def extract(self, skeleton: skan.Skeleton) -> types.Polyline:
        pass


class SkeletonPolylineExtractor(PolylineExtractor):
    def extract(self, skeleton: skan.Skeleton) -> types.Polyline:
        # Convert Skan's (row, column) coordinates to the project's (x, y) order.
        return skeleton.path_coordinates(0)[:, ::-1]


class EulerianPolylineExtractor(PolylineExtractor):
    def extract(self, skeleton: skan.Skeleton) -> types.Polyline:
        graph = nx.from_scipy_sparse_array(skeleton.graph)
        start_node, end_node = self._end_nodes(graph)
        route = self._eulerian_route(graph, start_node, end_node)
        coordinates = np.round(skeleton.coordinates[route]).astype(np.int32)
        return coordinates[:, ::-1]

    def _eulerian_route(
        self, graph: nx.Graph, start_node: int, end_node: int
    ) -> list[int]:
        eulerian_graph = nx.eulerize(graph)
        main_path = nx.shortest_path(graph, start_node, end_node, weight="weight")
        self._remove_main_edges(eulerian_graph, main_path)
        edges = list(nx.eulerian_path(eulerian_graph, source=start_node))
        return [edges[0][0]] + [end for _, end in edges]

    def _remove_main_edges(
        self, eulerian_graph: nx.MultiGraph, main_path: list[int]
    ) -> None:
        """Remove the doubled main path so the route can have two endpoints."""
        for start, end in zip(main_path, main_path[1:]):
            eulerian_graph.remove_edge(start, end)

    def _end_nodes(self, graph: nx.Graph) -> tuple[int, int]:
        first_end = self._farthest_node(graph, next(iter(graph)))
        second_end = self._farthest_node(graph, first_end)
        return first_end, second_end

    def _farthest_node(self, graph: nx.Graph, source: int) -> int:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        return max(lengths, key=lambda node: lengths[node])


class PolylineTransformer(ABC):
    @abstractmethod
    def transform(self, polylines: Sequence[types.Polyline]) -> types.Polyline:
        pass


class GreedyPolylineTransformer(PolylineTransformer):
    def transform(self, polylines: Sequence[types.Polyline]) -> types.Polyline:
        lines = list(polylines)
        while len(lines) > 1:
            first_index, second_index = self._find_closest_pair(lines)
            second = lines.pop(second_index)
            first = lines.pop(first_index)
            lines.append(self._merge_lines(first, second))
        return lines[0]

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
