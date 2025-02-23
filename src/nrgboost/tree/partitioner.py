from typing import Any, Callable, Optional
import numpy as np

from nrgboost.domain import ProductDomain
from nrgboost.distribution import MultivariateDistribution
from nrgboost.tree._partitioner import make_find_best_split, maximize, minimize, st

# TODO: get fastpath for encoded marginals:
#  - create array of size sum(decoded_marginals) to store decoded marginals
#  - create encoder indexer into this array to get array or marginals directly

# TODO: get rid of cumulative. This should always be the case so it's better to push
# cumsum to the function that receives the non-cumulative marginals for the ordered dims

# TODO: instead of writting a single general Partitioner
# and mask away features in subclasses
# should implement simpler versions of this class
# e.g.,
# - a specialized implementation without any leaf_dims
# - a specialized implementation with a single leaf_dim
# these would already cover all immediate use cases...


def argument_ge(arg_num: int, rvalue):
    def picked_argument(*args):
        return args[arg_num] >= rvalue

    return picked_argument


@staticmethod
def not_implemented(*args):
    raise NotImplementedError


def _split_ordered(*dist):
    parent = tuple(d[-1] for d in dist)
    left = tuple(d[:-1] for d in dist)
    right = tuple(p - l for p, l in zip(parent, left))
    return left, right, parent


def _split_categorical(*right):
    parent = [np.sum(d, 0) for d in right]
    left = [p - r for p, r in zip(parent, right)]
    return tuple(left), tuple(right), tuple(parent)


def _expand_dims_right(a, ndim):
    if a.ndim == ndim:
        return a
    return a[(Ellipsis,) + (None,)*(ndim - a.ndim)]


class DataPartitioner:
    objective = not_implemented
    value_function = not_implemented

    def sort_function(self, *dist):
        return self.value_function(*dist)

    # partition dimensions are all not included in leaves
    def __init__(self,
                 constraints: list[Callable] = [],
                 feature_fraction: float = 1,
                 cumulative: bool = True,
                 domain: Optional[ProductDomain] = None,
                 categorical_split_one_vs_all: bool = False,
                 use_fastpath: bool = True,
                 jit_unordered: bool = False,
                 seed: Optional[int] = None
                 ):
        self.constraints = st(constraints) if len(constraints) else None

        # TODO: this might fail if objective is not a static method
        # should just pass objective to partitioner
        # or otherside fazctorize objective + constraint into its own
        # self.find_dim_split = make_find_dim_split(
        #    self.objective, self.constraints)

        self.feature_fraction = feature_fraction
        self.cumulative = cumulative
        self.split_1va = categorical_split_one_vs_all
        self.use_fastpath = use_fastpath
        self.jit_unordered = jit_unordered
        self.domain = domain
        self.prng = np.random.default_rng(seed)
        self._find_best_split = None
        # self.split_finder = None

    def set_data(self, *data: MultivariateDistribution):
        self._check_domains(data)
        self.leaf_domains = [self.domain]

        self.partition_dims = np.arange(self.domain.ndim)
        self.data = [d.as_partitioned(self.leaf_domains) for d in data]
        # TODO: wtach out for sort_function not being value function
        if self.use_fastpath and (self._find_best_split is None):
            if self.split_1va: # No need for sort function
                sort_function = None
            elif not hasattr(self.sort_function, '__self__'):
                # self.sort_function is not an instance method so use it directly
                sort_function = self.sort_function
            elif self.sort_function.__func__ is DataPartitioner.sort_function:
                # self.sort_function is the default instance method -> use value_function
                sort_function = self.value_function
            else:
                raise NotImplementedError(
                    'Sort_function is instance method but not the default one. '
                    'This is not supported with use_fastpath= True.'
                )
            try:
                self._find_best_split = make_find_best_split(
                    len(data),
                    self.domain.ordered,
                    self.objective,
                    self.constraints,
                    sort_function,
                    jit_unordered=self.jit_unordered,
                    with_feature_frac=self.feature_fraction < 1
                )
            except Exception as e:
                print(
                    "Could not find fast path implementation for best split finder.", e)

    def reset_data(self, tree, *data: MultivariateDistribution):
        self.data = [d.tree_partition(tree, self.leaf_domains) for d in data]

    def set_seed(self, seed):
        self.prng = np.random.default_rng(seed)
        return self

    def _check_domains(self, data: list[MultivariateDistribution]):
        if self.domain is None:
            self.domain = data[0].domain
        assert all(d.domain == self.domain for d in data)

    def get_partition_values(self, return_leaf_sums=False):
        # TODO: improve this with a specialized method for partitioned datasets
        # TODO: this should be a marginal when leaf_dims is not empty
        leaf_sums = [
            # if ordered this will be only marginal
            np.array([partition.sum() for partition in data.partitions])
            for data in self.data
        ]

        # TODO: remove value function from partitioner! make this return leaf sums
        if return_leaf_sums:
            return leaf_sums

        return self.value_function(*leaf_sums)

    def find_best_split(self, leaf):
        if self.feature_fraction < 1:
            num_features = max(1, round(self.feature_fraction *
                                        len(self.partition_dims)))
            partition_dims = self.prng.choice(
                self.partition_dims, num_features, replace=False)
            partition_dims = np.sort(partition_dims)

            assert len(partition_dims) > 0
        else:
            partition_dims = None

        # fastpath available
        if self._find_best_split is not None:
            return self.__find_best_split(leaf, partition_dims=partition_dims)

        if partition_dims is None:
            partition_dims = self.partition_dims
        gains, split_values = zip(*self.get_leaf_gains(leaf, partition_dims))
        dim_idx = np.argmax(gains)
        return gains[dim_idx], partition_dims[dim_idx], split_values[dim_idx]

    def __find_best_split(self, leaf, partition_dims=None):
        leaf_domain = self.leaf_domains[leaf]

        if partition_dims is not None:
            leaf_domain = leaf_domain[partition_dims]
        encoder = leaf_domain.measure_encoder(as_mask=True)

        marginals = (d.partitions[leaf].full_encoded_marginals(
            partition_dims, encoder=encoder) for d in self.data)

        if partition_dims is None:
            gain, split_dim, split_index = self._find_best_split(
                leaf_domain.cardinalities, *marginals)
            split_value = leaf_domain[split_dim].decode(split_index)
        else:
            gain, split_dim, split_index = self._find_best_split(
                partition_dims, leaf_domain.cardinalities, *marginals)
            split_value = self.leaf_domains[leaf][split_dim].decode(
                split_index)

        return gain, split_dim, split_value

    # TODO: This functions should be jitted (and parallelized over partition_dims) somehow!!!
    def get_leaf_gains(self, leaf, partition_dims=None):
        if partition_dims is None:
            partition_dims = self.partition_dims
        # dim_mask = np.isin(self.partition_dims, partition_dims)

        leaf_domain = self.leaf_domains[leaf][partition_dims]
        leaf_marginals = [
            data.partitions[leaf].encoded_marginals(
                dims=partition_dims,
                cumulative=self.cumulative,
            )
            for data in self.data
        ]

        for dim_domain, *marginals in zip(leaf_domain, *leaf_marginals):
            if dim_domain.cardinality == 1:
                yield -np.inf, -1
                continue

            # marginals = [m.astype(np.float64) for m in marginals]
            if dim_domain.ordered:
                splits = _split_ordered(*marginals)
                split_index, gain = self.find_dim_split(*splits)
                split_index += 1
            elif self.split_1va:
                splits = _split_categorical(*marginals)
                split_index, gain = self.find_dim_split(*splits)
            else:
                values = self.sort_function(*marginals)
                sort_index = np.argsort(values, axis=0)
                marginals = [np.cumsum(marginal[sort_index], axis=0)
                             for marginal in marginals]

                splits = _split_ordered(*marginals)
                split_index, gain = self.find_dim_split(*splits)
                if sort_index.ndim <= 1:  # only 1 ordering
                    split_index = sort_index[:split_index+1]
                else:  # multiple orderings, best is chosen
                    split_index = np.unravel_index(
                        split_index, sort_index.shape)
                    split_index = sort_index[(
                        slice(split_index[0]+1), *split_index[1:])]

            yield gain, dim_domain.decode(split_index)

    def find_dim_split(self, *splits):
        gain = self.objective(*splits)
        if self.constraints is not None:
            constraints = self.constraints(*splits)
            constraints = _expand_dims_right(constraints, gain.ndim)
            gain = np.where(constraints, gain, -np.inf)
        idx = np.argmax(gain)
        return idx, gain.flat[idx]

    @property
    def num_leaves(self):
        return len(self.leaf_domains)

    def split_leaf(self, leaf: int, dim: int, value: Any) -> tuple[int, int]:
        left_domain, right_domain = self.leaf_domains[leaf].partition(
            dim, value)
        self.leaf_domains[leaf] = left_domain
        self.leaf_domains.append(right_domain)

        for data in self.data:
            data.split(leaf, dim, value)

        return leaf, self.num_leaves - 1
