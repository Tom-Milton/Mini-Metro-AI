from abc import ABC, abstractmethod
from typing import Callable, cast

import networkx as nx
import numpy as np
import skan
import skimage.morphology

from src.domain import types


# scikit-image does not expose a complete type signature for this function.
_typed_skeletonize = cast(
    Callable[[np.ndarray], np.ndarray],
    getattr(skimage.morphology, "skeletonize"),
)


class NoSkeletonError(Exception):
    """Raised when a mask has too few pixels to form a skan.Skeleton."""


class MultipleSkeletonError(Exception):
    """Raised when a skeleton contains more than one connected component."""


class SkeletonExtractor(ABC):
    @abstractmethod
    def extract(self, mask: types.Mask) -> skan.Skeleton:
        pass


class MaskSkeletonExtractor(SkeletonExtractor):
    def __init__(self, min_component_size: int):
        self.min_component_size = min_component_size

    def extract(self, mask: types.Mask) -> skan.Skeleton:
        skeleton_image = _typed_skeletonize(mask > 0)
        skeleton_image = skimage.morphology.remove_small_objects(
            skeleton_image,
            max_size=self.min_component_size - 1,
            connectivity=2,
        )
        components = skimage.morphology.label(skeleton_image, connectivity=2)
        component_count = int(np.max(components))
        if component_count == 0:
            raise NoSkeletonError
        if component_count > 1:
            raise MultipleSkeletonError
        return skan.Skeleton(skeleton_image)


class SkeletonTransformer(ABC):
    @abstractmethod
    def transform(self, skeleton: skan.Skeleton) -> skan.Skeleton:
        pass


class LongestSkeletonTransformer(SkeletonTransformer):
    def transform(self, skeleton: skan.Skeleton) -> skan.Skeleton:
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
