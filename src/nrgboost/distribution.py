from __future__ import annotations
from numba import jit

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Optional
from numbers import Number
from functools import reduce
from copy import copy

from nrgboost.domain import Domain, FiniteDomain, ProductDomain
from nrgboost.utils import dump_memmap, _FixedLengthIndexer, _VariableLengthIndexer


@dataclass
class Distribution:
    domain: Domain
    is_empirical: ClassVar[bool] = False

    def sample(self, num_samples: int, *args, seed: Optional[int] = None, **kwargs) -> Distribution:
        points = self.sample_points(num_samples, *args, seed=seed, **kwargs)
        return EmpiricalMultivariateDistribution(self.domain, points.T)

    def sample_points(self, num_samples: int, seed: Optional[int] = None) -> ArrayLike:
        raise NotImplementedError

    def encoded(self, cumulative=False) -> ArrayLike:
        raise NotImplementedError

    def predict(self, data: ArrayLike, slice_dims: list[int] = [], log: bool = False) -> ArrayLike:
        raise NotImplementedError


# TODO: is this needed? Or should this just be any distribution over a scalar domain?
@dataclass
class UnivariateDistribution(Distribution):
    domain: FiniteDomain

    def sum(self) -> Number:
        raise NotImplementedError


DimArg = int | list[int]


@dataclass
class MultivariateDistribution(Distribution):
    domain: ProductDomain

    @property
    def ndim(self):
        return self.domain.ndim

    def encoded(self):
        raise NotImplementedError

    def encoded_sum(self, dim: Optional[DimArg] = None, cumulative=False) -> ArrayLike:
        raise NotImplementedError

    def log_sum_all(self):  # default
        return np.log(self.sum_all())

    # Default implementation
    def sum(self, dims: Optional[DimArg] = None, log: bool = False) -> Distribution:
        if dims is None:
            return self.log_sum_all() if log else self.sum_all()

        assert not log, "log marginalize not implemented"
        return self.marginalize(self._not_dims(dims))

    def marginalize_encoded(self, dims: DimArg, cumulative=False) -> ArrayLike:
        raise NotImplementedError

    def _process_dims(self, dims: Optional[list[int]], include_dims: list[int]):
        if dims is None:
            return (i for i in range(self.ndim) if i not in include_dims)
        return dims

    def encoded_marginals(
        self,
        dims: Optional[list[int]] = None,
        include_dims: DimArg = [],
        cumulative=False,
    ) -> Iterable[ArrayLike]:
        raise NotImplementedError

    # Default implementation
    def marginalize(self, dim: DimArg) -> Distribution:
        domain = self.domain[dim]
        distribution = self.marginalize_encoded(dim)

        if isinstance(domain, FiniteDomain):
            return FullUnivariateDistribution(domain, distribution)
        return FullDistribution(domain, distribution)

    @property
    def marginals(self) -> Iterable[FullUnivariateDistribution]:
        return _FixedLengthIndexer(self.marginalize, self.ndim)

    def _not_dims(self, dims: DimArg):
        if np.isscalar(dims):
            dims = [dims]

        not_dims = set(range(self.ndim)) - set(dims)
        return sorted(list(not_dims))

    def memmap(self):
        return

    def as_partitioned(self, partition_domains: Optional[list[Domain]] = None):
        raise NotImplementedError

    def tree_partition(self, tree, partition_domains: list[Domain]):
        raise NotImplementedError


class PartitionedByDomainDistribution:

    def __init__(self, distribution: PartitionedByDomainMixin, partitions: Optional[list[Domain]] = None):
        if partitions is None:
            partitions = [distribution.domain]
            self._owner = True
        else:
            self._owner = False

        self.distribution = distribution
        self.partition_domains = partitions

    def get_partition(self, index):
        return self.distribution.restrict(self.partition_domains[index])

    def num_partitions(self):
        return len(self.partition_domains)

    # TODO: should this really be variable? Can't it have a length defined at the
    # point this attribute is indexed?
    # Alternativelly, should this just be a cached property or
    # created at initialization?
    @property
    def partitions(self):
        return _VariableLengthIndexer(self.get_partition, self.num_partitions)

    def split(self, index, dim, value):
        if not self._owner:
            return  # No need to do anything since not owner

        left, right = self.partition_domains[index].partition(dim, value)
        self.partition_domains[index] = left
        self.partition_domains.append(right)


class PartitionedByDomainMixin:

    def restrict(self, domain: Domain):
        raise NotImplementedError

    def as_partitioned(self, partition_domains: Optional[list[Domain]] = None):
        return PartitionedByDomainDistribution(self, partition_domains)

    def tree_partition(self, tree, partition_domains: list[Domain]):
        return PartitionedByDomainDistribution(self, partition_domains)


@dataclass(kw_only=True)
class ScalarWeightMixin:
    weight: Number = 1

    def reweight(self, weight):
        reweighted = copy(self)
        reweighted.weight = weight
        return reweighted

    def rescale(self, scale):
        return self.reweight(self.weight*scale)


@dataclass
class UnivariateUniformDistribution(ScalarWeightMixin, PartitionedByDomainMixin, UnivariateDistribution):

    def encoded(self, cumulative=False):
        if cumulative:
            return np.linspace(0, self.weight, self.domain.cardinality + 1)[1:]
        C = self.domain.cardinality
        return (self.weight / C) * np.ones(C)

    def restrict(self, domain: FiniteDomain):
        weight = self.weight * domain.cardinality / self.domain.cardinality
        return UnivariateUniformDistribution(domain, weight=weight)

    def sum(self):
        return self.weight


@dataclass
class ProductDistribution(ScalarWeightMixin, MultivariateDistribution):

    def sum_all(self):
        return self.weight

    def _marginalize(self, dim: int) -> ArrayLike:
        raise NotImplementedError

    def _marginalize_cumulative(self, dim: int) -> ArrayLike:
        return np.cumsum(self._marginalize(dim))

    def __marginalize(self, dim: int, cumulative: bool = False) -> ArrayLike:
        if cumulative and self.domain[dim].ordered:
            return self._marginalize_cumulative(dim)
        return self._marginalize(dim)

    def marginalize_encoded(self, dims: DimArg, cumulative: bool = False):
        if np.isscalar(dims):
            return self.weight * self.__marginalize(dims, cumulative)

        marginals = [self.__marginalize(dim, cumulative) for dim in dims]
        marginals[0] *= self.weight
        return reduce(np.multiply.outer, marginals)

    def encoded_marginals(
        self,
        dims: Optional[list[int]] = None,
        include_dims: list[int] = [],
        cumulative: bool = False,
    ):
        dims = self._process_dims(dims, include_dims)

        if len(include_dims) > 0:
            include_marginals = self.marginalize_encoded(
                include_dims, cumulative=cumulative)
        else:
            include_marginals = self.weight

        for dim in dims:
            assert dim not in include_dims
            yield np.multiply.outer(self.__marginalize(dim, cumulative), include_marginals)

    def encoded(self, cumulative=False):
        return self.marginalize_encoded(range(self.ndim), cumulative=cumulative)


@dataclass
class UniformDistribution(PartitionedByDomainMixin, ProductDistribution):
    
    def predict(self, data: ArrayLike, slice_dims: list[int] = [], log: bool = False) -> ArrayLike:
        shape = self.domain[slice_dims].cardinalities
        value = np.log(self.weight) - self.domain.log_cardinality
        if not log:
            value = np.exp(log)
            
        return np.full((data.shape[0], *shape), value)

    def _marginalize(self, dim: int):
        C = self.domain[dim].cardinality
        return np.ones(C) / C

    def _marginalize_cumulative(self, dim: int):
        return np.linspace(0, 1, self.domain[dim].cardinality + 1)[1:]

    def marginalize(self, dims: DimArg):
        if np.isscalar(dims):
            return UnivariateUniformDistribution(self.domain[dims], weight=self.weight)
        return UniformDistribution(self.domain[dims], weight=self.weight)

    def full_encoded_marginals(self, dims: Optional[list[int]] = None, encoder: ArrayLike = None):
        if dims is None:
            dims = slice(None)

        cardinalities = self.domain.cardinalities[dims]
        num_dims = cardinalities.size
        sum_cardinalities = np.zeros(num_dims + 1, dtype=np.intp)
        sum_cardinalities[1:] = np.cumsum(cardinalities)
        marginals = np.empty(sum_cardinalities[-1], dtype=np.float64)
        for C, start, stop in zip(cardinalities, sum_cardinalities[:-1], sum_cardinalities[1:]):
            marginals[start:stop] = self.weight/C

        return marginals

    def restrict(self, domain: Domain):
        weight = self.weight * \
            np.prod(domain.cardinalities / self.domain.cardinalities)
        return UniformDistribution(domain, weight=weight)

    def sample_points(self, num_samples: int, seed: int | None = None) -> ArrayLike:
        prng = np.random.default_rng(seed)
        points = prng.integers(self.domain.cardinalities,
                               size=(num_samples, self.ndim))
        return self.domain.decode(points)


@dataclass
class ProductOfMarginalsDistribution(PartitionedByDomainMixin, ProductDistribution):
    # TODO: variant that stores cumulative marginals instead
    _marginals: list[ArrayLike]
    _sum_marginals: ArrayLike | None = None

    def __post_init__(self):
        if self._sum_marginals is not None:
            return

        sum_marginals = np.array([np.sum(m) for m in self._marginals])
        assert np.allclose(sum_marginals[1:], sum_marginals[0])
        weight = self.weight * sum_marginals[0]

        object.__setattr__(self, '_sum_marginals', sum_marginals)
        object.__setattr__(self, 'weight', weight)

    def predict(self, data: ArrayLike, slice_dims: list[int] = [], log: bool = False) -> ArrayLike:
        logits = np.log(self.weight) - np.sum(np.log(self._sum_marginals))
        logits = np.full(data.shape[0], logits)

        for dim, (m, col) in enumerate(zip(self._marginals, data.T)):
            if dim in slice_dims:
                logits = np.add.outer(np.log(m), logits)
            else:
                logits += np.log(m)[col]  # np.take(np.log(m), col)

        logits = np.transpose(logits, (len(slice_dims),) +  # sample axis is last
                              tuple(len(slice_dims) - 1 - np.argsort(np.argsort(slice_dims))))  # TODO: figure out a better way...

        return logits if log else np.exp(logits)

    def _marginalize(self, dim: int):
        return self._marginals[dim] / self._sum_marginals[dim]

    def full_encoded_marginals(self, dims: Optional[list[int]] = None, encoder: ArrayLike = None):
        if dims is None:
            marginals_and_sums = zip(self._marginals, self._sum_marginals)
        else:
            marginals_and_sums = (
                (self._marginals[dim], self._sum_marginals[dim]) for dim in dims)

        return np.concatenate([m/M for m, M in marginals_and_sums]) * float(self.weight)

    def marginalize(self, dims: DimArg):
        if np.isscalar(dims):
            marginal = self._marginals[dims] * \
                (self.weight / self._sum_marginals[dims])
            return FullUnivariateDistribution(self.domain[dims], marginal)
        marginals = [self._marginals[dim] for dim in dims]
        sum_marginals = self._sum_marginals[np.asarray(dims)]
        return ProductOfMarginalsDistribution(self.domain[dims], marginals, sum_marginals, weight=self.weight)

    def restrict(self, domain: Domain):
        marginals = [d.encode_measure(m, from_domain=from_d)
                     for d, from_d, m in zip(domain, self.domain, self._marginals)]

        sum_marginals = np.copy(self._sum_marginals)
        weight = self.weight
        for dim, (current_marginal, restricted_marginal) in enumerate(zip(self._marginals, marginals)):
            if current_marginal.size > restricted_marginal.size:  # dim is restricted
                new_sum = np.sum(restricted_marginal)
                weight *= new_sum/sum_marginals[dim]
                sum_marginals[dim] = new_sum
        return ProductOfMarginalsDistribution(domain, marginals, sum_marginals, weight=weight)

    @staticmethod
    def from_distribution(distribution: MultivariateDistribution, uniform_mixture: float = 0):
        marginals = distribution.encoded_marginals()
        if uniform_mixture > 0:
            assert uniform_mixture <= 1
            total = distribution.sum_all()
            marginals = [(1 - uniform_mixture)*m +
                         uniform_mixture*total/m.size for m in marginals]
        else:
            marginals = list(marginals)
        return ProductOfMarginalsDistribution(distribution.domain, marginals)

    def sample_points(self, num_samples: int, seed: int | None = None) -> ArrayLike:
        prng = np.random.default_rng(seed)
        points = [prng.choice(m.size, p=m/np.sum(m), size=num_samples)
                  for m in self._marginals]
        return np.stack(points, -1)


def sample_discrete(probs, num_samples, prng):
    raise NotImplementedError


@dataclass
class FullUnivariateDistribution(PartitionedByDomainMixin, UnivariateDistribution):
    _distribution: ArrayLike

    def sum(self):
        return np.sum(self._distribution)

    def encoded(self, cumulative=False):
        if cumulative:
            return np.cumsum(self._distribution)
        return self._distribution

    def restrict(self, domain: FiniteDomain):
        encoded = domain.encode_measure(
            self._distribution, from_domain=self.domain)
        return FullUnivariateDistribution(domain, encoded)


@dataclass
class FullDistribution(PartitionedByDomainMixin, MultivariateDistribution):
    _distribution: ArrayLike

    def sum(self, dims: Optional[DimArg] = None):
        if dims is None:
            return np.sum(self._distribution)

        raise NotImplementedError

    def sum_all(self):
        return np.sum(self._distribution)

    def __post_init__(self):
        assert self.ndim == self._distribution.ndim
        assert self.domain.shape == self._distribution.shape

    def encoded(self, cumulative=False):
        if cumulative:
            encoded = np.copy(self._distribution)
            for i in range(self.ndim):
                encoded = np.cumsum(encoded, i, out=encoded)
            return encoded
        return self._distribution

    def marginalize_encoded(self, dims: DimArg, cumulative: bool = False):
        sum_dims = self._not_dims(dims)

        marginal = np.sum(self._distribution, axis=tuple(sum_dims))
        if cumulative:
            if np.isscalar(dims):
                dims = [dims]
            for i, domain in enumerate(self.domain[dims]):
                if domain.ordered:
                    marginal = np.cumsum(marginal, axis=i)
        return marginal

    def encoded_marginals(
        self,
        dims: Optional[list[int]] = None,
        include_dims: list[int] = [],
        cumulative: bool = False,
    ):

        dims = self._process_dims(dims, include_dims)
        if include_dims:
            for dim in dims:
                assert dim not in include_dims
                yield self.marginalize_encoded([dim, *include_dims], cumulative=cumulative)
        else:
            for dim in dims:
                yield self.marginalize_encoded(dim, cumulative)

    def marginalize(self, dims: int | list[int]):
        marginals = self.marginalize_encoded(dims)
        if np.isscalar(dims):
            return FullUnivariateDistribution(self.domain[dims], marginals)
        return FullDistribution(self.domain[dims], marginals)

    def restrict(self, domain: Domain):
        encoded = domain.encode_measure(
            self._distribution, from_domain=self.domain)
        return FullDistribution(domain, encoded)

    def sample_points(self, num_samples, seed=None):
        prng = np.random.default_rng(seed)
        p = np.reshape(self._distribution, -1)
        sum_p = np.sum(p)
        if not np.isclose(sum_p, 1):
            p /= sum_p
    
        samples = prng.choice(p.size, num_samples, p=p)
        return np.unravel_index(samples, self._distribution.shape)

    def sample(self, num_samples, relative=False, seed=None):
        prng = np.random.default_rng(seed)
        p = np.reshape(self._distribution, -1)
        sum_p = np.sum(p)
        if not np.isclose(sum_p, 1):
            p /= sum_p

        samples = prng.multinomial(num_samples, p)
        if relative:
            samples /= num_samples
        return FullDistribution(self.domain, np.reshape(samples, self._distribution.shape))

    @staticmethod
    def from_distribution(distribution: MultivariateDistribution):
        return FullDistribution(distribution.domain, distribution.encoded())


@jit(nopython=True)
def _empirical_decoded_full_marginalizer(data, weight, decoded_cardinalities):
    # print(data.shape, sum_cardinalities.shape)
    sum_cardinalities = np.zeros(
        decoded_cardinalities.size + 1, dtype=decoded_cardinalities.dtype)
    sum_cardinalities[1:] = np.cumsum(decoded_cardinalities)
    data_ = data + sum_cardinalities[:-1, np.newaxis]

    # weights_ = np.repeat(
    #    weights, data.shape[0]) if weights is not None else None
    if weight == 1.0:
        return np.bincount(data_.reshape(-1), minlength=sum_cardinalities[-1]).astype(np.float64)
    return weight * np.bincount(data_.reshape(-1), minlength=sum_cardinalities[-1])


@jit(nopython=True)
def _weighted_empirical_decoded_full_marginalizer(data, weights, decoded_cardinalities):
    # print(data.shape, sum_cardinalities.shape)
    sum_cardinalities = np.zeros(
        decoded_cardinalities.size + 1, dtype=decoded_cardinalities.dtype)
    sum_cardinalities[1:] = np.cumsum(decoded_cardinalities)
    data_ = data + sum_cardinalities[:-1, np.newaxis]

    weights = np.repeat(weights, decoded_cardinalities.size)

    return np.bincount(data_.reshape(-1), weights=weights, minlength=sum_cardinalities[-1])


# TODO: Consider changing the orientation of data
# for encoded_marginals also vectorize so that all marginals are computed in a single pass
# of the array getting rid of the python loop
# Actually this doesn't seem to be a significan bottleneck!
@dataclass
class EmpiricalMultivariateDistribution(ScalarWeightMixin, MultivariateDistribution):
    data: ArrayLike
    mask_based_partitioning: bool = field(default=False, kw_only=True)
    copy_ok: bool = field(default=True, kw_only=True)

    def __post_init__(self):
        self.data = np.ascontiguousarray(self.data)

    def sum_all(self):
        return self.weight * self.num_samples

    @property
    def num_samples(self):
        return self.data.shape[-1]

    def mask(self, mask: ArrayLike, domain: Optional[ProductDomain] = None):
        if domain is None:
            domain = self.domain
        if not self.copy_ok:
            return WeightedEmpiricalMultivariateDistribution(domain, self.data, weights=mask, weight=self.weight)

        return EmpiricalMultivariateDistribution(domain, data=self.data[:, mask], weight=self.weight)

    # Another option would be to encode data first then bincount
    # TODO: profile best option
    def marginalize_encoded_scalar(self, dim: int, cumulative=False):
        dim_domain = self.domain[dim]
        decoded_marginal = np.bincount(self.data[dim], weights=self.weights,
                                       minlength=dim_domain.decoded_cardinality)
        marginal = self.weight * dim_domain.encode_measure(decoded_marginal)
        if cumulative and dim_domain.ordered:
            return np.cumsum(marginal)
        return marginal

    def marginalize_encoded(self, dims: Optional[DimArg] = None, cumulative=False) -> ArrayLike:
        if np.isscalar(dims):
            return self.marginalize_encoded_scalar(dims, cumulative=cumulative)

        marginalized_domain = self.domain[dims]
        shape = marginalized_domain.decoded_cardinalities
        # TODO: try setting clip mode with +1 of margin maybe?
        # This way could avoid keeping track of the decoded cardinality
        # (i.e., only track stop)
        # Another option is to just filter data outside of domain
        # or bincount encoded data
        flat_data = np.ravel_multi_index(self.data[dims], shape)
        decoded_marginal = np.bincount(
            flat_data, weights=self.weights, minlength=np.prod(shape))
        decoded_marginal = np.reshape(decoded_marginal, shape)
        marginal = self.weight * \
            marginalized_domain.encode_measure(decoded_marginal)

        if cumulative:
            for i, domain in enumerate(marginalized_domain):
                if domain.ordered:
                    marginal = np.cumsum(marginal, axis=i)
        return marginal

    def full_encoded_marginals(self, dims: Optional[list[int]] = None, encoder: ArrayLike = None):
        if dims is not None:
            data = self.data[dims]
            domain = self.domain[dims]
        else:
            data = self.data
            domain = self.domain

        decoded_cardinalities = domain.decoded_cardinalities
        if encoder is None:
            encoder = domain.measure_encoder(as_mask=True)
        return _empirical_decoded_full_marginalizer(data, float(self.weight), decoded_cardinalities)[encoder]

    def encoded_marginals(
        self,
        dims: Optional[list[int]] = None,
        include_dims: DimArg = [],
        cumulative=False,
    ) -> Iterable[ArrayLike]:

        dims = self._process_dims(dims, include_dims)
        if len(include_dims) > 0:
            for dim in dims:
                assert dim not in include_dims
                yield self.marginalize_encoded([dim] + include_dims, cumulative=cumulative)
            return

        for dim in dims:
            yield self.marginalize_encoded_scalar(dim, cumulative=cumulative)

    def marginalize(self, dims: DimArg) -> Distribution:
        domain = self.domain[dims]
        if np.isscalar(dims):
            encoded = self.marginalize_encoded_scalar(dims)
            return FullUnivariateDistribution(domain, encoded)

        return EmpiricalMultivariateDistribution(domain, self.data[dims], weight=self.weight)

    def as_partitioned(self, partition_domains: Optional[list[Domain]] = None, partitions: Optional[ArrayLike] = None):
        if self.mask_based_partitioning:
            return MaskBasedPartitionedEmpiricalDistribution(self, partition_domains, partitions)
        return PartitionedEmpiricalDistribution(self, partition_domains, partitions)

    def tree_partition(self, tree, partition_domains: list[Domain]):
        partitions = tree.evaluate(self.data.T, only_leaf_index=True)
        return self.as_partitioned(self, partition_domains, partitions=partitions)

    # TODO: consider if want these to return an EmpiricalMultivariateDistribution instead
    # TODO: consider whether to rescale weights to match original num_samples
    # TODO: remove the non-booststrap options

    def sample(self, num_samples: int, method='bootstrap', seed=None):
        prng = np.random.default_rng(seed)

        if method == 'bootstrap':
            idx = prng.choice(self.num_samples, int(num_samples), replace=True)
            weights = np.bincount(idx, minlength=self.num_samples)
        elif method == 'choice':
            idx = prng.choice(self.num_samples, int(
                num_samples), replace=False)
            weights = np.zeros(self.num_samples, dtype=np.int_)
            weights[idx] = 1
        elif method == 'bernoulli':
            weights = prng.random(
                self.num_samples) < num_samples/self.num_samples

        return WeightedEmpiricalMultivariateDistribution(self.domain, self.data, weights=weights, weight=self.weight)

    def sample_points(self, num_samples: int, seed=None):
        prng = np.random.default_rng(seed)
        idx = prng.integers(self.num_samples, size=num_samples)
        return self.data.T[idx]

    def memmap(self):
        self.data = dump_memmap(self.data)

    def rescale(self, scale):
        return EmpiricalMultivariateDistribution(self.domain, self.data, weight=scale*self.weight)


EmpiricalMultivariateDistribution.weights = None


@dataclass
class WeightedEmpiricalMultivariateDistribution(EmpiricalMultivariateDistribution):
    weights: ArrayLike

    def __post_init__(self):
        super().__post_init__()
        self.weights = self.weights * self.weight
        self.weight = 1

    def sum_all(self):
        return np.sum(self.weights)

    def mask(self, mask: ArrayLike, domain: Optional[ProductDomain] = None):
        if domain is None:
            domain = self.domain
        if not self.copy_ok:
            return WeightedEmpiricalMultivariateDistribution(domain, data=self.data, weights=self.weights * mask)

        return WeightedEmpiricalMultivariateDistribution(domain, data=self.data[:, mask], weights=self.weights[mask])

    def encoded(self):
        raise NotImplementedError

    def encoded_sum(self, dim: Optional[DimArg] = None, cumulative=False) -> ArrayLike:
        raise NotImplementedError

    def marginalize(self, dims: DimArg) -> Distribution:
        domain = self.domain[dims]
        if np.isscalar(dims):
            encoded = self.marginalize_encoded_scalar(dims)
            return FullUnivariateDistribution(domain, encoded)

        return WeightedEmpiricalMultivariateDistribution(domain, self.data[dims], weights=self.weights)

    def full_encoded_marginals(self, dims: Optional[list[int]] = None, encoder: ArrayLike = None):
        if dims is not None:
            data = self.data[dims]
            domain = self.domain[dims]
        else:
            data = self.data
            domain = self.domain

        decoded_cardinalities = domain.decoded_cardinalities
        if encoder is None:
            encoder = domain.measure_encoder(as_mask=True)
        return _weighted_empirical_decoded_full_marginalizer(data, self.weights, decoded_cardinalities)[encoder]

    def sample(self, num_samples: int, seed=None):
        prng = np.random.default_rng(seed)

        pvals = self.weights/self.sum_all()
        weights = prng.multinomial(num_samples, pvals)
        return WeightedEmpiricalMultivariateDistribution(self.domain, self.data, weights)

    def sample_points(self, num_samples: int, seed=None):
        raise NotImplementedError

    def reweight(self, weight):
        raise NotImplementedError  # TODO: how to interpret reweight?

    def rescale(self, scale):
        return WeightedEmpiricalMultivariateDistribution(self.domain, self.data, weights=scale*self.weights)


class PartitionedEmpiricalDistribution:

    def __init__(self,
                 distribution: EmpiricalMultivariateDistribution,
                 partition_domains: Optional[list[Domain]] = None,
                 partitions: Optional[ArrayLike] = None,
                 ):
        if partition_domains is None:
            partition_domains = [distribution.domain]
            assert partitions is None

            self._owns_domains = True
        else:
            self._owns_domains = False

        if partitions is None:
            if len(partition_domains) == 1:  # single partition
                partition_distributions = [distribution]
            else:
                # TODO: implement method to check if array in domain
                raise NotImplementedError
        else:
            # TODO: check
            partition_distributions = [
                distribution.mask(partitions == i, d) for i, d in enumerate(partition_domains)]

        self.partitions = partition_distributions
        self.partition_domains = partition_domains

    @property
    def num_partitions(self):
        return len(self.partitions)

    def split(self, index, dim, value):
        if self._owns_domains:
            # TODO: factor this into base class
            left_domain, right_domain = self.partition_domains[index].partition(
                dim, value)
            self.partition_domains[index] = left_domain
            self.partition_domains.append(right_domain)

        partition_distribution = self.partitions[index]
        col = partition_distribution.data[dim]
        if self.partition_domains[index][dim].ordered:
            right_mask = col >= value
        else:
            right_mask = np.isin(col, value)

        self.partitions[index] = partition_distribution.mask(
            ~right_mask, self.partition_domains[index])
        self.partitions.append(
            partition_distribution.mask(right_mask, self.partition_domains[-1]))


class MaskBasedPartitionedEmpiricalDistribution:

    def __init__(self,
                 distribution: EmpiricalMultivariateDistribution,
                 partition_domains: Optional[list[Domain]] = None,
                 partitions: Optional[ArrayLike] = None,
                 ):
        if partition_domains is None:
            partition_domains = [distribution.domain]
            assert partitions is None

            partitions = np.zeros(distribution.num_samples, dtype=np.int_)
            self._owns_domains = True
            self._owns_partitions = True
        else:
            if partitions is None:
                if len(partition_domains) == 1:  # single partition
                    partitions = np.zeros(
                        distribution.num_samples, dtype=np.int_)
                else:
                    # TODO: implement method to check if array in domain
                    raise NotImplementedError
                self._owns_partitions = True
            else:
                self._owns_partitions = False
            self._owns_domains = False

        self.distribution = distribution
        self._partitions = partitions
        self.partition_domains = partition_domains
        self._num_partitions = len(partition_domains)

    def num_partitions(self):
        return self._num_partitions

    def get_partition(self, index):
        if self._num_partitions == 1:
            return self.distribution

        mask = self._partitions == index
        domain = self.partition_domains[index]
        return self.distribution.mask(mask, domain)

    @property
    def partitions(self):
        return _VariableLengthIndexer(self.get_partition, self.num_partitions)

    def split(self, index, dim, value):
        if self._owns_domains:
            # TODO: factor this into base class
            left, right = self.partition_domains[index].partition(dim, value)
            self.partition_domains[index] = left
            self.partition_domains.append(right)
        if self._owns_partitions:
            col = self.distribution.data[dim]
            partition_mask = self._partitions == index
            if self.partition_domains[index][dim].ordered:
                right_mask = col >= value
            else:
                right_mask = np.isin(col, value)

            self._partitions[partition_mask &
                             right_mask] = self._num_partitions

        self._num_partitions += 1

######################################################## FACTORIZED ######################################################################


@dataclass
class FactorizedEmpiricalDistribution(EmpiricalMultivariateDistribution):
    conditional_data: ArrayLike
    conditional_dim: int

    def mask(self, mask: ArrayLike, domain: ProductDomain):
        if domain is None:
            domain = self.domain

        # TODO: make sure this works
        conditional_data = domain[self.conditional_dim].encode_measure(
            self.conditional_data, from_domain=self.domain[self.conditional_dim])
        if not self.copy_ok:
            return self._Weighted(
                domain,
                data=self.data,
                conditional_data=conditional_data,
                conditional_dim=self.conditional_dim,
                weights=mask)

        return self._Self(
            domain,
            data=self.data[:, mask],
            conditional_data=conditional_data[..., mask],
            conditional_dim=self.conditional_dim)

    @property
    def _conditional_marginal(self):
        return np.sum(self._conditional_weights, -1)

    @property
    def _conditional_weights(self):
        if self.weights is None:
            return self.conditional_data

        return self.conditional_data * self.weights

    def marginalize_encoded_scalar(self, dim: int, cumulative=False):
        if dim != self.conditional_dim:
            return super().marginalize_encoded_scalar(dim, cumulative=cumulative)

        # already encoded
        marginal = self._conditional_marginal
        if cumulative and self.domain[dim].ordered:
            return np.cumsum(marginal)
        return marginal

    # TODO: make this work with cdim being any dimension
    def marginalize_encoded(self, dims: Optional[DimArg] = None, cumulative=False) -> ArrayLike:
        if np.isscalar(dims):
            return self.marginalize_encoded_scalar(dims, cumulative=cumulative)
        if len(dims) == 1:
            return self.marginalize_encoded_scalar(dims[0], cumulative=cumulative)

        if self.conditional_dim not in dims:
            return super().marginalize_encoded(dims, cumulative=cumulative)

        if dims[-1] != self.conditional_dim:
            raise NotImplementedError(
                "For now conditional_dim must be last dim")

        cdim = dims[-1]
        dims = dims[:-1]
        marginalized_domain = self.domain[dims]
        shape = tuple(marginalized_domain.decoded_cardinalities)

        flat_data = np.ravel_multi_index(self.data[dims], shape)

        decoded_marginal = [np.bincount(
            flat_data, weights=w, minlength=np.prod(shape)) for w in self._conditional_weights]
        decoded_marginal = np.stack(decoded_marginal, -1)
        decoded_marginal = np.reshape(decoded_marginal, shape + (-1,))
        marginal = marginalized_domain.encode_measure(
            decoded_marginal)

        if cumulative:
            for i, domain in enumerate(self.domain[dims + [cdim]]):
                if domain.ordered:
                    marginal = np.cumsum(marginal, axis=i)
        return marginal

    def marginalize(self, dims: DimArg) -> Distribution:
        domain = self.domain[dims]
        if np.isscalar(dims):
            encoded = self.marginalize_encoded_scalar(dims)
            return FullUnivariateDistribution(domain, encoded)

        if self.conditional_dim in dims:
            conditional_dim = dims.index(self.conditional_dim)
            return self._Self(domain, self.data[dims], self.conditional_data, conditional_dim)

        return EmpiricalMultivariateDistribution(domain, self.data[dims])

    def as_partitioned(self, partition_domains: Optional[list[Domain]] = None, partitions: Optional[ArrayLike] = None):
        return FactorizedPartitionedEmpiricalDistribution(self, partition_domains, partitions)

    def tree_partition(self, tree, partition_domains: list[Domain]):
        raise NotImplementedError
        # partitions = tree.evaluate(self.data.T, only_leaf_index=True)
        # return PartitionedEmpiricalDistribution(self, partition_domains, partitions, copy=copy)


class LogitFactorizedEmpiricalDistribution(FactorizedEmpiricalDistribution):
    @property
    def _conditional_marginal(self):
        if self.num_samples == 0:
            return np.zeros(self.conditional_data.shape[0])
        return np.exp(logsumexp(self.conditional_data, -1, self.weights))

    @property
    def _conditional_weights(self):
        if self.weights is None:
            return np.exp(self.conditional_data)

        return np.exp(self.conditional_data) * self.weights

    def add_conditional(self, delta):
        self.conditional_data = self.conditional_data + delta
        # renormalize
        self.conditional_data -= logsumexp(self.conditional_data, 0)


@dataclass
class WeightedFactorizedEmpiricalDistribution(FactorizedEmpiricalDistribution):
    weights: ArrayLike

    def mask(self, mask: ArrayLike, domain: Optional[ProductDomain] = None):
        if domain is None:
            domain = self.domain

        # TODO: make sure this works
        conditional_data = domain[self.conditional_dim].encode_measure(
            self.conditional_data, from_domain=self.domain[self.conditional_dim])
        if not self.copy_ok:
            return self._Weighted(
                domain,
                data=self.data,
                conditional_data=conditional_data,
                conditional_dim=self.conditional_dim,
                weights=self.weights * mask)

        return self._Weighted(
            domain,
            data=self.data[:, mask],
            conditional_data=conditional_data[..., mask],
            conditional_dim=self.conditional_dim,
            weights=self.weights[mask])

    def marginalize(self, dims: DimArg) -> Distribution:
        domain = self.domain[dims]
        if np.isscalar(dims):
            encoded = self.marginalize_encoded_scalar(dims)
            return FullUnivariateDistribution(domain, encoded)

        if self.conditional_dim in dims:
            conditional_dim = dims.index(self.conditional_dim)
            return self._Weighted(domain, self.data[dims], self.conditional_data, conditional_dim, weights=self.weights)

        return WeightedEmpiricalMultivariateDistribution(domain, self.data[dims], weights=self.weights)


# TODO: replace this with a Mixin
FactorizedEmpiricalDistribution.weights = None
FactorizedEmpiricalDistribution._Self = FactorizedEmpiricalDistribution
FactorizedEmpiricalDistribution._Weighted = WeightedFactorizedEmpiricalDistribution


class WeightedLogitFactorizedEmpiricalDistribution(WeightedFactorizedEmpiricalDistribution):
    pass


LogitFactorizedEmpiricalDistribution._Self = LogitFactorizedEmpiricalDistribution
LogitFactorizedEmpiricalDistribution._Weighted = WeightedLogitFactorizedEmpiricalDistribution
WeightedLogitFactorizedEmpiricalDistribution._Self = LogitFactorizedEmpiricalDistribution
WeightedLogitFactorizedEmpiricalDistribution._Weighted = WeightedLogitFactorizedEmpiricalDistribution


class FactorizedPartitionedEmpiricalDistribution(MaskBasedPartitionedEmpiricalDistribution):
    def split(self, index, dim, value):
        if dim == self.distribution.conditional_dim:
            raise NotImplementedError(
                "Current Partitioner Implementation does not suppoert splitting on conditional dimension.")
        # TODO: Need to keep one mask per partition because when splitting on
        # conditional dim, a datapoint may belong to multiple partitions (disjoint over conditional_dim)

        super().split(index, dim, value)
