from __future__ import annotations
from functools import reduce

import math
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import ClassVar, Optional


class Domain:
    ndim: ClassVar[int] = 1


class ScalarDomain(Domain):
    ordered: ClassVar[bool]
    discrete: ClassVar[bool]
    finite: ClassVar[bool]


class FiniteDomain(ScalarDomain):
    """A finite domain.

    Holds the metadata necessary to encode and decode into a range of ints.
    """
    discrete: ClassVar[bool] = True
    finite: ClassVar[bool] = True

    @property
    def cardinality(self):
        raise NotImplementedError

    @property
    def decoded_cardinality(self):
        raise NotImplementedError

    def encode(self, values: ArrayLike) -> ArrayLike:
        '''Encodes a discrete domain into a 0-based int range.

        The encoded domain is [0..cardinality[

        If the domain is unordered, the encoded domain will be an int set instead.
        '''
        raise NotImplementedError

    def decode(self, values: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def measure_encoder(self, from_domain: Optional[FiniteDomain] = None, as_mask: bool = False):
        '''Returns object that can be used to encode a decoded measure.

        The measure should be over the cannonical domain (i.e., decoded) 
        or encoded over a `from_domain`.

        '''
        raise NotImplementedError

    def encode_measure(
        self, measure: ArrayLike,
        from_domain: Optional[FiniteDomain] = None,
        axis: Optional[int] = None
    ) -> ArrayLike:
        encoder = self.measure_encoder(from_domain)
        if axis is not None:
            encoder = (slice(None),) * (axis - 1) + (encoder,)
        return measure[encoder]

    # TODO: can't imagine this being useful
    def decode_measure(self, measure: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
        raise NotImplementedError

    def contains(self, points: ArrayLike):
        raise NotImplementedError

    @property
    def dtype(self):
        return int_dtype(self.cardinality)


@dataclass(repr=False, frozen=True)
class FiniteIntRange(FiniteDomain):
    ordered: ClassVar[bool] = True
    start: int
    stop: Optional[int] = None
    max: Optional[int] = None

    def __post_init__(self):
        if self.stop is not None:
            return
        object.__setattr__(self, 'stop', self.start)
        object.__setattr__(self, 'start', 0)

        if self.max is None:
            object.__setattr__(self, 'max', self.stop)

    @property
    def cardinality(self):
        return self.stop - self.start

    @property
    def decoded_cardinality(self):
        return self.max

    @property
    def encode(self, values):
        return values - self.start

    def decode(self, values):
        return values + self.start

    def measure_encoder(self, from_domain: Optional[FiniteIntRange] = None, as_mask: bool = False):
        if from_domain is None:
            if as_mask:
                mask = np.zeros(self.decoded_cardinality, dtype=np.bool_)
                mask[self.start:self.stop] = 1
                return mask
            return slice(self.start, self.stop)

        offset = from_domain.start
        assert self.start >= offset
        assert self.stop <= from_domain.stop
        if as_mask:
            mask = np.zeros(self.decoded_cardinality, dtype=np.bool_)
            mask[self.start - offset:self.stop - offset] = 1
            return mask
        return slice(self.start - offset, self.stop - offset)

    def decode_measure(self, measure: ArrayLike, max: Optional[int] = None):
        if max is None:
            max = self.max

        output = np.zeros(max, dtype=measure.dtype)
        output[self.start:self.stop] = measure
        return output

    def contains(self, points: ArrayLike):
        return np.logical_and(points >= self.start, points < self.stop)

    def partition(self, value: int) -> tuple[FiniteIntRange]:
        """Partition range into two ranges.

        Args:
            value (int): The value to partition on.

        Raises

        Returns:
            tuple[FiniteIntRange]: Tuple with left and right ranges.
                value will be included in the right range (x >= value).
        """
        assert value > self.start and value < self.stop, \
            f"value ({value}) should be between start ({self.start}) and stop ({self.stop})."
        return FiniteIntRange(self.start, value, self.max), FiniteIntRange(value, self.stop, self.max)

    def __repr__(self):
        return f'[{self.start}..{self.stop}['


@dataclass(repr=False, frozen=True)
class FiniteIntSet(FiniteDomain):
    ordered: ClassVar[bool] = False
    max: int
    support: Optional[ArrayLike[np.uint8]] = None  # bitpacked available bits

    def __post_init__(self):
        if self.support is not None:
            return

        bytes, bits = divmod(self.max, 8)
        if bits == 0:
            support = np.empty(bytes, dtype=np.uint8)
            support[:] = 255
        else:
            support = np.empty(bytes + 1, dtype=np.uint8)
            support[:-1] = 255
            support[-1] = np.int8(-(1 << 8 - bits)).astype(np.uint8)

        object.__setattr__(self, 'support', support)

    # TODO: these properties should probably be precomputed or cached
    @property
    def cardinality(self):
        return int(np.unpackbits(self.support).sum())

    @property
    def decoded_cardinality(self):
        return self.max

    # TODO: dtype depends on cardinality
    @property
    def _encoder(self):
        return np.cumsum(np.unpackbits(self.support)[:self.max], dtype=np.uint8) - 1

    # TODO: dtype depends on cardinality
    @property
    def _decoder(self):
        return np.where(np.unpackbits(self.support)[:self.max])[0]

    @property
    def _support_mask(self):
        return np.unpackbits(self.support)[:self.max].view(np.bool_)

    def encode(self, values):
        return self._encoder[np.asarray(values)]

    def decode(self, values):
        return self._decoder[np.asarray(values)]

    def measure_encoder(self, from_domain: Optional[FiniteIntSet] = None, as_mask: bool = False):
        if from_domain is None:
            return self._support_mask

        # TODO: handle case where self is a FiniteIntSet with a smaller max than from_domain?
        return self._support_mask[from_domain._support_mask]

    def decode_measure(self, measure: ArrayLike):
        output = np.zeros(self.max, dtype=measure.dtype)
        output[self._support_mask] = measure
        return output

    # TODO: consider a special point domain for right is np.isscalar(values)
    def partition(self, values: int | ArrayLike[int]) -> tuple[FiniteIntSet]:
        support_left, support_right = self._partition_values(values)
        return FiniteIntSet(self.max, support_left), FiniteIntSet(self.max, support_right)

    def _partition_values(self, values: ArrayLike[int]):
        bytes, bits = np.divmod(values, 8)
        values = (np.uint8(0x80) >> bits.astype(np.uint8))

        support_left = np.copy(self.support)
        support_right = np.zeros_like(self.support)

        for byte in np.unique(bytes):
            value = np.bitwise_or.reduce(values[bytes == byte])
            support_left[byte] &= ~value
            support_right[byte] = value

        return support_left, support_right

    # We don't check bounds because we assume points will be always less than max
    # TODO: But for general use should check this
    def contains(self, points: ArrayLike):
        return self._support_mask[points]

    def __repr__(self):
        if self.cardinality == self.max:
            return '{' + f'0..{self.max-1}' + '}'

        values = np.where(np.unpackbits(self.support))[0]
        if values.size == 1:
            return '{' + str(values[0]) + '}'

        return '{' + ', '.join([str(v) for v in values]) + '}'


def ix_(indexes):
    is_simple = tuple(isinstance(idx, slice) for idx in indexes)
    if all(is_simple):
        return indexes

    advanced_indexes = (np.arange(idx.start, idx.stop, idx.step)
                        if simple else idx for idx, simple in zip(indexes, is_simple))
    return np.ix_(*advanced_indexes)


@dataclass(frozen=True)
class ProductDomain(Domain):
    domains: list[FiniteDomain]

    @property
    def ndim(self):
        return len(self.domains)

    @property
    def shape(self):
        return tuple(d.cardinality for d in self.domains)

    @property
    def cardinalities(self):
        return np.array([d.cardinality for d in self.domains])

    @property
    def decoded_cardinalities(self):
        return np.array([d.decoded_cardinality for d in self.domains])

    @property
    def ordered(self):
        return np.array([d.ordered for d in self.domains])

    @property
    def log_cardinality(self):
        return np.sum(np.log(self.cardinalities))

    def __getitem__(self, dim: int | list[int]) -> ScalarDomain:
        if np.isscalar(dim):
            return self.domains[dim]

        domains = self.domains
        return ProductDomain([domains[i] for i in dim])

    def decode(self, values: ArrayLike):
        return np.stack([d.decode(v) for d, v in zip(self.domains, values.T)], -1)

    def measure_encoder(self, from_domain: Optional[ProductDomain] = None, as_mask: bool = False):
        '''Returns object that can be used to encode a decoded measure.

        The measure should be over the cannonical domain (i.e., decoded) 
        or encoded over a `from_domain`.

        '''
        if from_domain is None:
            encoder = tuple(d.measure_encoder(as_mask=as_mask)
                            for d in self.domains)
        else:
            encoder = tuple(d.measure_encoder(from_domain=from_d, as_mask=as_mask)
                            for d, from_d in zip(self.domains, from_domain.domains))
        return np.concatenate(encoder) if as_mask else encoder

    def encode_measure(
        self, measure: ArrayLike,
        from_domain: Optional[FiniteDomain] = None
    ) -> ArrayLike:
        encoder = self.measure_encoder(from_domain)
        return measure[ix_(encoder)]

    def partition(self, dim, value):
        """Partition product domain into two domains.

        Args:
            dim (int): The dimension to split on.
            value (int): The value to partition on.

        Raises

        Returns:
            tuple[FiniteIntRange]: Tuple with left and right domains.
        """
        domains_left, domains_right = list(self.domains), list(self.domains)
        domains_left[dim], domains_right[dim] = self.domains[dim].partition(
            value)

        return ProductDomain(domains_left), ProductDomain(domains_right)

    def contains(self, points: ArrayLike):
        contained = (d.contains(p) for d, p in zip(self.domains, points.T))
        return reduce(np.logical_and, contained)

    @property
    def dtypes(self):
        return tuple(d.dtype for d in self.domains)

    def __repr__(self):
        return ' x '.join(repr(d) for d in self.domains)


def int_dtype(cardinality):
    INT_DTYPES = (np.uint8, np.uint16, np.uint32, np.uint64)

    bit_length = (cardinality - 1).bit_length()
    index = math.ceil(bit_length / 8) - 1

    if index < len(INT_DTYPES):
        return INT_DTYPES[index]
    return np.uint64
