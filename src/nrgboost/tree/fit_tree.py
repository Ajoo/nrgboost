from __future__ import annotations

from typing import Any
from numpy.typing import ArrayLike

import numpy as np

from dataclasses import dataclass, field
from collections import namedtuple
from operator import itemgetter

from nrgboost.tree.build_tree import BaseTreeFitter
from nrgboost.tree.partitioner import DataPartitioner
from nrgboost.tree.eval import EvalTreeBuilder, EvalNode


@dataclass(slots=True)
class FitNode:
    left: FitNode | int
    right: FitNode | int
    dim: int
    value: Any

    @property
    def left_is_leaf(self):
        return isinstance(self.left, int)

    @property
    def right_is_leaf(self):
        return isinstance(self.right, int)


def build_node_list(root: FitNode):
    nodes = []
    _build_node_list(root, nodes, 0)
    return [n[1] for n in sorted(nodes, key=itemgetter(0))]


def _build_node_list(node: FitNode, nodes, index):

    num_internal_nodes = 1 if np.isscalar(node.value) else len(node.value)

    start = len(nodes)

    if node.left_is_leaf:
        left_index = -node.left - 1
    else:
        left_index = index + num_internal_nodes
        _build_node_list(node.left, nodes, left_index)

    if node.right_is_leaf:
        right_index = -node.right - 1
    else:
        right_index = index + len(nodes) - start + num_internal_nodes
        _build_node_list(node.right, nodes, right_index)

    if np.isscalar(node.value):
        nodes.append((
            index,
            EvalNode(left_index, right_index,
                     node.dim, node.value)
        ))
        return

    for i, val in enumerate(node.value[:-1]):
        nodes.append((
            index + i,
            EvalNode(index + i + 1, right_index,
                     node.dim, val)
        ))
    nodes.append((
        index + num_internal_nodes - 1,
        EvalNode(left_index, right_index,
                 node.dim, node.value[-1])
    ))


class NoSplitPossibleException(Exception):
    pass


@dataclass
class TreeFitter(BaseTreeFitter):
    root: FitNode = field(init=False, default=None)
    leaf_parents: dict[int, FitNode] = field(init=False, default_factory=dict)

    def fit(self, data: DataPartitioner):
        self.data = data
        self.root = None
        self.leaf_parents = dict()

        self.build_tree()

        if self.root is None:
            raise NoSplitPossibleException(
                "Could not find a split on root node! Consider decreasing min_gain or relaxing partitioner constraints.")

        nodes = build_node_list(self.root)
        return EvalTreeBuilder(self.data.domain, nodes, self.data.get_partition_values())

    def find_split(self, leaf: int) -> tuple[float, int, Any]:
        return self.data.find_best_split(leaf)

    def split(self, leaf: int, dim: int, value: Any):
        left_leaf, right_leaf = self.data.split_leaf(leaf, dim, value)

        node = FitNode(left_leaf, right_leaf, dim, value)
        if self.leaf_parents:
            parent_node = self.leaf_parents[leaf]
            if parent_node.left == leaf:
                parent_node.left = node
            elif parent_node.right == leaf:
                parent_node.right = node
            else:
                raise AssertionError(
                    f"Parent node {parent_node} should have had leaf {leaf} as either left or right child.")
        else:
            self.root = node

        self.leaf_parents[left_leaf] = node
        self.leaf_parents[right_leaf] = node
        return left_leaf, right_leaf


Range = namedtuple('Range', ['start', 'stop'])


class ScalarTreeFitter(BaseTreeFitter):

    def __init__(self, gain_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain_function = gain_function
        self.bins = []

    def __post_init__(self):
        self.seed = np.random.default_rng(self.seed)

    def fit(self, *data: ArrayLike):
        # TODO: generalize so not cumulative necessarily
        self.cdata = [np.insert(np.cumsum(d), 0, 0) for d in data]
        leaves = self.build_tree(Range(0, data[0].size))
        self.bins = np.sort(self.bins)
        return leaves

    def find_split(self, leaf: Range) -> tuple[float, int]:
        if leaf.stop - leaf.start <= 1:
            return -np.inf, -1
        cdata = [d[leaf.start+1:leaf.stop+1] - d[leaf.start]
                 for d in self.cdata]
        gains = self.gain_function(*cdata)
        # TODO: consider random sampling from same values
        split_value = np.argmax(gains)
        return gains[split_value], leaf.start + split_value + 1

    def split(self, leaf: Range, value: Any):
        self.bins.append(value)

        return Range(leaf.start, value), Range(value, leaf.stop)

    @property
    def num_leaves(self):
        return len(self.bins) + 1
