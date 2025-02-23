from typing import Optional
from nrgboost.distribution import MultivariateDistribution, UniformDistribution
import numpy as np
import operator

from nrgboost.domain import ProductDomain
from nrgboost.tree.partitioner import DataPartitioner, minimize, maximize


def neg_relative_entropy(p, q, *_):
    return np.where(p == 0, 0, p*np.log(q))


def neg_entropy(p, *_):
    return neg_relative_entropy(p, p)


def kl(p, q, *_):
    return np.where(p == 0, 0, p*(np.log(p) - np.log(q)))
    # return neg_entropy(p) - neg_relative_entropy(p, q)


def neg_gini(p, *_):
    return p**2


def chisq_m(p, *q):
    return p**2 / sum(q)


def chisq(p, q):
    return p**2 / q


max_entropy = minimize(neg_entropy)
max_kl = maximize(kl)
max_chisq = maximize(chisq)


def density_value(p, *_):
    return p


def log_density_value(p, *_):
    return np.log(p)


def density_ratio_m(p, *q):
    return p / sum(q)


def density_ratio(p, *q):
    return p / q


def density_ratio_m1_m(p, *q):
    return p / sum(q) - 1.0


def density_ratio_m1(p, q):
    return p / q - 1.0


def reg_density_ratio_m1(rho):
    def density_ratio_m1(p, *q):
        return p / (sum(q) + rho) - 1.0

    return density_ratio_m1


def log_density_ratio_m(p, *q):
    return np.log(p) - np.log(sum(q))


def log_densities_m(p, *q):
    return np.log(np.stack([p, *q], -1))


def log_density_ratio(p, q):
    return np.log(p) - np.log(q)


def log_densities(p, q):
    return np.log(np.stack([p, q], -1))


def log_half_density_ratio_p1(p, *q):
    return np.log(p/sum(q)/2 + 1)


def min_data_in_leaf(rvalue, allow_equal=True):
    comparison = operator.ge if allow_equal else operator.gt

    def constraint_func(p, *_):
        return comparison(p, rvalue)

    return constraint_func


def min_ref_in_leaf(rvalue, allow_equal=True):
    comparison = operator.ge if allow_equal else operator.gt

    def constraint_func(p, r, *_):
        return comparison(r, rvalue)

    return constraint_func


def max_ratio_in_leaf(rvalue, allow_both_zero=False):
    """Constrains ratio p(X) / q(X) in a leaf X to be smaller than a constant

    Args:
        rvalue (float): max value for ratio
        allow_both_zero (bool, optional): If True allows leaves where both p and q are 0. Defaults to False.

    Returns:
        ArrayLike: mask
    """
    def constraint_func(p, q, *_):
        # max ratio over leaf dim
        ratio = p / q
        mask = ratio <= rvalue
        if allow_both_zero:
            # in case all are 0
            return np.logical_or(mask, np.isnan(ratio))
        return mask

    return constraint_func


class constraints:
    min_data_in_leaf = min_data_in_leaf
    min_ref_in_leaf = min_ref_in_leaf
    max_ratio_in_leaf = max_ratio_in_leaf


class GenerativeDataPartitioner(DataPartitioner):
    def __init__(
        self,
        constraints: list = [],
        feature_fraction: float = 1,
        domain: Optional[ProductDomain] = None,
        categorical_split_one_vs_all: bool = False,
        use_fastpath: bool = True,
        jit_unordered: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            constraints=constraints,
            feature_fraction=feature_fraction,
            cumulative=True,
            domain=domain,
            categorical_split_one_vs_all=categorical_split_one_vs_all,
            use_fastpath=use_fastpath,
            jit_unordered=jit_unordered,
            seed=seed
        )


class EnergyDataPartitioner(GenerativeDataPartitioner):
    value_function = staticmethod(log_densities)
    sort_function = staticmethod(log_density_ratio)

    def set_data(self, data: MultivariateDistribution, reference: Optional[MultivariateDistribution] = None):
        if reference is None:
            reference = UniformDistribution(data.domain)

        return super().set_data(data, reference)


class MaxLikelihoodDataPartitioner(EnergyDataPartitioner):
    """Data Partitioner for maximizing Likelihood.
    """
    # maximize kl from data to reference distribution
    objective = staticmethod(max_kl)


class MinBrierScoreDataPartitioner(EnergyDataPartitioner):
    """Data Partitioner for minimizing the Brier Score.
    I.e., the squared error between data and model distributions.
    """
    # maximize Chi squared from data to reference distribution
    objective = staticmethod(max_chisq)


class MidpointRandomPartitioner(GenerativeDataPartitioner):
    value_function = staticmethod(log_density_ratio)

    def __init__(
        self,
        constraints: list = [],
        feature_fraction: float = 1,
        domain: Optional[ProductDomain] = None,
        categorical_split_one_vs_all: bool = False,
        seed: Optional[int] = None,
    ):
        assert len(constraints) == 0
        assert feature_fraction == 1
        super().__init__(domain=domain,
                         categorical_split_one_vs_all=categorical_split_one_vs_all, seed=seed)

    def find_best_split(self, leaf):
        leaf_domain = self.leaf_domains[leaf]
        split_dim = self.prng.choice(
            [d for d in self.partition_dims if leaf_domain[d].cardinality > 1])
        dim_domain = leaf_domain[split_dim]

        if dim_domain.ordered:
            split_val, remainder = divmod(dim_domain.cardinality, 2)
            if remainder:
                split_val += self.prng.random() < 0.5
        elif self.split_1va:  # pick random
            split_val = self.prng.integers(dim_domain.cardinality)
        else:
            split_val = self.prng.permutation(dim_domain.cardinality)[
                :dim_domain.cardinality//2]

        return 0, split_dim, dim_domain.decode(split_val)


class NRGBoostDataPartitioner(GenerativeDataPartitioner):
    """Data Partitioner for minimizing quadratic approximation to likelihood
    w.r.t. energy function at an existing model.
    """
    objective = staticmethod(max_chisq)
    value_function = staticmethod(density_ratio_m1)
