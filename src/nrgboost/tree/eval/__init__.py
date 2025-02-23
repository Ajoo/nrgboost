# TODO: consider having only a relative offset for right and left leaf
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from itertools import islice, accumulate
from numpy.typing import ArrayLike
from typing import Optional
from tqdm import tqdm
import warnings


from nrgboost.domain import ProductDomain, FiniteDomain
from _eval import ffi, lib
import pickle


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# TODO: change this to produce a numpy array of nodes (represented as int64)
EvalNode = namedtuple(
    "EvalNode", ["left", "right", "dim", "value"])


def to_eval_node(node):
    return EvalNode(node.left, node.right, node.dim, node.value)


# Serializeable version of an EvalTree without cffi types
# TODO: replace nodes with a numpy array instead of EvalNodes
# TODO: think of a better way to do this that doesn't involve having these separate classes
@dataclass
class EvalTreeBuilder:
    domain: ProductDomain
    nodes: list[EvalNode]
    leaf_values: Optional[ArrayLike] = None

    def build(self):
        return EvalTree(self.domain, self.nodes, self.leaf_values)


def make_support(domain: FiniteDomain):
    if domain.ordered:
        return ffi.new("Range64 *", {"start": domain.start, "stop": domain.stop})

    return ffi.new("Set64 *", {
        "max_value": domain.decoded_cardinality,
        # No need to keep track of this since it isn't going anywhere
        "mask": ffi.from_buffer(domain.support)
    })


def cumsumexp(logits):
    logits = np.copy(logits.astype(np.double), 'C')
    lib.cumsumexp(ffi.from_buffer("double *", logits), logits.size)
    return logits


def sample_cat(probs, size=None, seed=None):
    prng = np.random.default_rng(seed)
    probs = np.ascontiguousarray(probs.astype(np.double))

    u = prng.random(size)
    if np.isscalar(u):
        return lib.sample_cat(u, ffi.from_buffer("double *", probs), probs.size)

    samples = np.empty(u.shape, np.intc)
    for i, ui in enumerate(u.flat):
        samples[i] = lib.sample_cat(
            ui, ffi.from_buffer("double *", probs), probs.size)
    return samples


class EvalDomain:
    def __init__(self, domain: ProductDomain):
        self.domain = domain
        self.is_dim_ordered = ffi.new("int[]", [d.ordered for d in domain])
        self.support = [make_support(d) for d in domain]
        self._support = ffi.new(
            "void *[]", [ffi.cast("void *", s) for s in self.support])

        self.cdata = ffi.new("ProductDomain *", {
            "num_dims": domain.ndim,
            "is_dim_ordered": self.is_dim_ordered,
            "support": self._support
        })

    def print(self):
        lib.ProductDomain_print(self.cdata)


# TODO: clean this up so that EvalDomain has everything required for running evaluations
# and we don't need an explicit link in EvalTree
class EvalTree:
    """Class that wraps a binary tree to be used for evaluation.
    """

    def __init__(
        self,
        domain: ProductDomain,
        nodes: list[EvalNode],
        leaf_values: ArrayLike = None,
        leaf_domains: None = None,  # just here for backwards compatibility
    ):
        if isinstance(domain, EvalDomain):
            self.domain = domain.domain
            self._domain = domain
        else:
            self.domain = domain
            self._domain = EvalDomain(domain)

        self.leaf_values = leaf_values

        num_nodes = len(nodes)
        num_leaves = leaf_values.shape[0]
        assert num_nodes < 2**15, "Too many interior nodes."
        assert num_leaves <= 2**15, "Too many leaves."

        if isinstance(nodes, np.ndarray):
            self._nodes = nodes
            nodes = ffi.from_buffer("Node64 *", nodes)
            # num_leaves = - nodes.view(np.int16).reshape((-1, 4))[:, :2].min()
        else:
            self._nodes = ffi.new("Node64[]", nodes)
            nodes = self._nodes
            # num_leaves = -min(min(node.left, node.right)
            #                   for node in nodes)

        self._tree = ffi.new("Tree64*", {
            "domain": self._domain.cdata,
            "nodes": nodes,
            "values": ffi.from_buffer("double *", leaf_values),
            "num_nodes": num_nodes,
            "num_leaves": num_leaves,
        })

    def serialize(self):
        nodes = np.frombuffer(ffi.buffer(self._nodes),
                              dtype=np.int64, count=self.num_nodes)

        return {
            'domain': self.domain,
            'nodes': nodes,
            'leaf_values': self.leaf_values,
        }

    @staticmethod
    def deserialize(dict):
        return EvalTree(**dict)

    @property
    def num_nodes(self):
        return self._tree.num_nodes

    @property
    def num_leaves(self):
        return self._tree.num_leaves

    @property
    def ndim(self):
        return self.domain.ndim

    def print(self):
        lib.Tree64_print(self._tree)

    def evaluate(self, data: ArrayLike, slice_dims: list[int] = [], only_leaf_index: bool = False):
        num_samples, num_dims = data.shape
        assert num_dims == self.ndim, ("Data doesn't have the right number of dimensions."
                                       f"Expected {self.ndim}, was {num_dims}")

        data = np.ascontiguousarray(data.astype(np.uint8))
        assert data.flags["C_CONTIGUOUS"]

        if len(slice_dims) == 0:
            leaf_index = np.empty(num_samples, dtype=np.int16)
            assert leaf_index.flags["C_CONTIGUOUS"]

            lib.Tree64_eval_uint8(
                self._tree,
                ffi.from_buffer("int16_t *", leaf_index),
                # ffi.cast("int16_t *", ffi.from_buffer(leaf_index)),
                ffi.from_buffer(data),
                num_samples
            )
        elif len(slice_dims) == 1:
            dim = slice_dims[0]
            slice_domain = self.domain[dim]
            leaf_index = np.full((num_samples,
                                  slice_domain.decoded_cardinality), -1, dtype=np.int16)
            assert leaf_index.flags["C_CONTIGUOUS"]

            lib.Tree64_eval_slice_uint8(
                self._tree,
                ffi.from_buffer("int16_t *", leaf_index),
                ffi.from_buffer(data),
                num_samples,
                dim,
            )
        else:
            raise NotImplementedError

        if only_leaf_index or self.leaf_values is None:
            return leaf_index

        return self.leaf_values[leaf_index]

    @property
    def leaf_domains(self):
        nodes = self._tree.nodes
        domains = {0: self.domain}
        i = 0
        while i < self.num_nodes:
            node = nodes[i]
            d = domains.pop(i)
            i += 1

            if d[node.dim].ordered:
                values = node.value
            else:
                values = [node.value]
                # composite split
                while i < self.num_nodes and nodes[i].right == node.right:
                    node = nodes[i]
                    values.append(node.value)
                    i += 1
            domains[node.left], domains[node.right] = d.partition(
                node.dim, values)

        leaf_indexes = ~np.arange(self.num_leaves, dtype=np.int16)
        return [domains[idx] for idx in leaf_indexes]

    def sample_points(self, num_points, seed):
        from nrgboost.distribution import UniformDistribution

        prng = np.random.default_rng(seed)

        if self.leaf_values.ndim < 2:
            warnings.warn("Only one log density recorded")
            leaf_values = self.leaf_values
        else:
            leaf_values = self.leaf_values[..., 0]
        leaf_probabilities = np.exp(
            leaf_values - np.logaddexp.reduce(leaf_values))

        leaf_distributions = [UniformDistribution(
            d) for d in self.leaf_domains]
        num_samples_per_leaf = prng.multinomial(num_points, leaf_probabilities)
        samples = []
        for dist, nleaf in zip(leaf_distributions, num_samples_per_leaf):
            samples.append(dist.sample_points(nleaf, seed=prng))

        return np.concatenate(samples, 0)[prng.permutation(num_points)]

    def scale(self, factor: float):
        self.leaf_values *= factor


@dataclass
class EvalTrees:
    trees: list[EvalTree]

    @property
    def num_trees(self):
        return len(self.trees)

    @property
    def domain(self):
        return self.trees[0].domain

    def __getitem__(self, index):
        return self.trees[index]

    def add(self, tree: EvalTree):
        self.trees.append(tree)

    def serialize(self):
        return [tree.serialize() for tree in self.trees]

    @classmethod
    def deserialize(cls, trees):
        return cls([EvalTree.deserialize(tree) for tree in trees])

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.serialize(), file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            trees = pickle.load(file)
            return cls.deserialize(trees)


def _spawn_seed_seqs(prng, num_spawn, num_state=2, dtype=np.uint64):
    seed_seqs = [s.generate_state(num_state, dtype)
                 for s in prng.bit_generator.seed_seq.spawn(num_spawn)]
    return np.stack(seed_seqs, 0)
    # return [r.bit_generator.seed_seq.generate_state(num_state, dtype) for r in prng.spawn(num_spawn)]


class EvalTreeSum(EvalTrees):
    # void Tree64_gibbs_uint8(const Tree64* trees, int num_trees, uint8_t* output, int num_samples, uint8_t* sample, unsigned int seed)
    def sample_points(
        self,
        num_samples_per_chain,
        num_chains=1,
        initial_logits=[],
        stop_round=None,
        temperature=1.0,
        initial_samples=None,
        burn_in=0,
        seed=None,
        num_threads=0
    ):
        """_summary_

        Args:
            num_samples_per_chain (_type_): _description_
            num_chains (int, optional): _description_. Defaults to 1.
            initial_logits (list, optional): _description_. Defaults to [].
            initial_sample (_type_, optional): _description_. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.
            num_threads (int, optional): Number of threads used to run different chains in parallel. 
                0 (default) means use the default number of 

        Returns:
            _type_: _description_
        """
        from nrgboost.distribution import UniformDistribution

        prng = np.random.default_rng(seed)

        if initial_samples is None:
            initial_samples = UniformDistribution(
                self.domain).sample(num_chains, seed=prng).data
            initial_samples = np.ascontiguousarray(initial_samples)
        else:
            initial_samples = np.copy(initial_samples, 'C')

        total_samples_per_chain = num_samples_per_chain + burn_in
        trees = [t._tree for t in self.trees[:stop_round]]
        output = np.empty((num_chains, total_samples_per_chain, self.domain.ndim),
                          dtype=np.uint8, order='C')
        initial_logits = [ffi.from_buffer('double *', l)
                          for l in initial_logits]

        seed_seqs = _spawn_seed_seqs(prng, num_chains)
        # print(seed_seqs)
        # seed_seqs = [ffi.from_buffer('uint64_t*', seed_seq)
        #             for seed_seq in seed_seqs]

        if num_chains == 1:
            if num_threads != 0:
                warnings.warn(
                    "Trying to use more than 1 thread while running a single chain. This setting will be ignored.")

            lib.Tree64_gibbs_uint8(
                ffi.new('Tree64*[]', trees),
                len(trees),
                ffi.from_buffer('uint8_t*', output),
                total_samples_per_chain,
                ffi.from_buffer('uint8_t*', initial_samples),
                initial_logits if initial_logits else ffi.NULL,
                temperature,
                ffi.from_buffer('uint64_t*', seed_seqs),
            )
        else:
            lib.Tree64_gibbs_uint8_p(
                ffi.new('Tree64*[]', trees),
                len(trees),
                ffi.from_buffer('uint8_t*', output),
                total_samples_per_chain,
                num_chains,
                ffi.from_buffer('uint8_t*', initial_samples),
                initial_logits if initial_logits else ffi.NULL,
                temperature,
                ffi.from_buffer('uint64_t*', seed_seqs),
                num_threads
            )

        return output[:, burn_in:].reshape((-1, self.domain.ndim))

    # TODO: parallelize this (in C)
    def evaluate(self, data: ArrayLike, slice_dims: list[int] = [], only_leaf_index: bool = False, cumulative: bool = False, initial=None, start=None, stop=None):
        tree_evals = (t.evaluate(data, slice_dims=slice_dims, only_leaf_index=only_leaf_index)
                      for t in self.trees[start:stop])
        if only_leaf_index:
            return np.stack(tuple(tree_evals), 0)

        if cumulative:
            return accumulate(tree_evals, initial=initial)

        return sum(tree_evals, start=initial if initial is not None else 0)


class EvalTreeLogSumExp(EvalTrees):

    def sample_points(self, num_points, seed=None):
        prng = np.random.default_rng(seed)
        tree_assignments = prng.choice(self.num_trees, num_points)
        num_samples_per_tree = np.bincount(
            tree_assignments, minlength=self.num_trees)

        samples = np.empty((num_points, self.domain.ndim),
                           dtype=np.uint8, order='C')
        for t, (tree, ntree) in tqdm(enumerate(zip(self, num_samples_per_tree))):
            # print(f'Sampling {ntree} samples from tree {t}')
            samples[tree_assignments == t] = tree.sample_points(
                ntree, prng)  # shuffle

        return samples

    def evaluate(self, data: ArrayLike, slice_dims: list[int] = [], only_leaf_index: bool = False, cumulative: bool = False, chunksize=None, start=None, stop=None):
        num_slice_dims = len(slice_dims)

        def eval_tree(t):
            e = t.evaluate(data, slice_dims=slice_dims,
                           only_leaf_index=only_leaf_index)
            if e.ndim == 1 + num_slice_dims:
                return e

            return e[..., 0] - np.sum(e[..., 1:], -1)

        tree_evals = (eval_tree(t) for t in self.trees[slice(start, stop)])

        if only_leaf_index:
            return np.stack(tuple(tree_evals), 0)

        if chunksize is None:
            reduction = np.logaddexp.accumulate if cumulative else np.logaddexp.reduce
            return reduction(tuple(tree_evals))

        batches = iter(batched(tree_evals, chunksize))

        if cumulative:  # single output per chunk
            output = [np.logaddexp.reduce(batch) for batch in batches]
            return np.logaddexp.accumulate(output)

        output = np.logaddexp.reduce(next(batches))
        for batch in batches:
            output = np.logaddexp.reduce((output,) + batch)

        return output


def bincount(data, weights=None, minlengths=0, num_parallel=0):
    assert data.dtype == np.uint8, f"Only uint8 data is supported, got {data.dtype}"

    num_dims, num_samples = data.shape
    if np.isscalar(minlengths):
        if minlengths == 0:
            minlengths = np.max(data, -1) + 1
        else:
            minlengths = np.broadcast_to(minlengths, num_dims)
    counts = [np.zeros(l, dtype=np.int64) for l in minlengths]

    data = np.ascontiguousarray(data)
    if weights is None:
        lib.bincount_p(
            ffi.from_buffer('uint8_t*', data),
            num_dims,
            num_samples,
            [ffi.from_buffer('int64_t*', cnt) for cnt in counts],
            num_parallel
        )
    else:
        weights = np.ascontiguousarray(weights.astype(np.float64))
        lib.weighted_bincount_p(
            ffi.from_buffer('uint8_t*', data),
            ffi.from_buffer('double*', weights),
            num_dims,
            num_samples,
            [ffi.from_buffer('int64_t*', cnt) for cnt in counts],
            num_parallel
        )
    return counts


def sample(num_samples, bits=32, seed=123):
    output = np.empty(num_samples, dtype=np.double, order='C')
    if bits == 32:
        lib.sample32(ffi.from_buffer('double*', output), num_samples, seed)
    elif bits == 64:
        lib.sample64(ffi.from_buffer('double*', output), num_samples, seed)
    else:
        raise NotImplementedError
    return output
