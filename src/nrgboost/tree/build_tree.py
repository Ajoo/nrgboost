from __future__ import annotations
from typing import Any, Optional, ClassVar

import numpy as np

import sys
from heapq import heappush, heappop
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseTreeFitter:
    """ Base class for a tree.

    This only defines how the tree is built but is agnostic to what is the objective
    to pursue. This can be defined through the find_split and split methods.

    """
    max_leaves: int
    min_gain: Optional[float] = 0.
    splitter: Optional[str] = "best"
    seed: Optional[int | np.random.Generator] = None
    max_iters: ClassVar[int] = sys.maxsize

    def __post_init__(self):
        self.seed = np.random.default_rng(self.seed)

    def find_split(self, leaf: int) -> tuple:
        """Finds the best split parameters for the current node.

        Returns:
            tuple: The first element is the gain for the best split and the remaining
            elements are the parameters of the best split
        """
        raise NotImplementedError

    def split(self, leaf: int, *params: tuple) -> tuple[Any]:
        """Split the node into two children nodes.

        Args:
            leaf: The leaf index.
            params: The parameters of the split.

        Returns:
            tuple[int]: Tuple of children indexes
        """
        raise NotImplementedError

    def build_tree(self, root: Any = 0):
        if self.splitter.lower().startswith('best'):
            return build_tree_best_first(self, root, self.max_leaves, self.min_gain, self.max_iters)
        elif self.splitter.lower().startswith('depth'):
            return build_tree_breadth_first(self, root, self.max_leaves, self.min_gain, self.max_iters)
        elif self.splitter.lower().startswith('random'):
            return build_tree_random(self, root, self.max_leaves, self.min_gain, self.max_iters, self.seed)

        raise ValueError(
            f'Splitter value {self.splitter} not supported. Options are "best", "depth" or "random".')


@dataclass(order=True)
class PrioritizedLeaf:
    negative_gain: float
    depth: int
    leaf: Any = field(compare=False)
    split_params: tuple[Any] = field(compare=False)


def build_tree_best_first(tree: BaseTreeFitter, root: Any, max_leaves, min_gain=0., max_iters=sys.maxsize):
    max_negative_gain = - min_gain
    fringe = []

    gain, *split_params = tree.find_split(root)
    heappush(fringe, PrioritizedLeaf(-gain, 0, root, split_params))

    for _ in range(max_iters):
        # break if we would go over max_leaves
        if max_leaves > 0 and len(fringe) >= max_leaves:
            break

        leaf_to_split = heappop(fringe)
        # break if fringe has no more splittable leaves
        if leaf_to_split.negative_gain > max_negative_gain:
            fringe.insert(0, leaf_to_split)       # put it back
            break

        children = tree.split(leaf_to_split.leaf, *leaf_to_split.split_params)
        for child in children:
            gain, *split_params = tree.find_split(child)
            heappush(fringe,
                     PrioritizedLeaf(-gain, leaf_to_split.depth +
                                     1, child, split_params)
                     )

    return [leaf.leaf for leaf in fringe]


def build_tree_breadth_first(tree: BaseTreeFitter, root: Any, max_leaves, min_gain=0., max_iters=sys.maxsize):
    fringe = deque([root])

    terminal = []
    for _ in range(max_iters):
        # break if we would go over max_leaves
        if max_leaves > 0 and len(fringe) >= max_leaves:
            break

        try:
            leaf_to_split = fringe.popleft()
        except IndexError:
            break

        gain, *split_params = tree.find_split(leaf_to_split)

        # if leaf cannot be split any further
        if gain < min_gain:
            terminal.append(leaf_to_split)
            continue

        children = tree.split(leaf_to_split, *split_params)
        for leaf in children:
            fringe.append(leaf)

    return terminal + list(fringe)


def build_tree_random(tree: BaseTreeFitter, max_leaves, min_gain=0., max_iters=sys.maxsize, seed=None):
    prng = np.random.default_rng(seed)

    raise NotImplementedError
