from nrgboost.tree.eval import EvalTrees, EvalTreeSum, EvalTreeLogSumExp, EvalTree
from nrgboost.tree.generative import (
    MaxLikelihoodDataPartitioner,
    MinBrierScoreDataPartitioner,
    MidpointRandomPartitioner,
    NRGBoostDataPartitioner,
)
from nrgboost.tree.partitioner import DataPartitioner
from nrgboost.tree.fit_tree import TreeFitter, NoSplitPossibleException
from nrgboost.distribution import EmpiricalMultivariateDistribution, MultivariateDistribution, UniformDistribution, ProductOfMarginalsDistribution
from typing import Optional
from joblib import Parallel, delayed
import numpy as np
from scipy.special import logsumexp
from numpy.typing import ArrayLike
from tqdm import tqdm


class LogitBaggingTreeEstimator(EvalTreeLogSumExp):
    def predict(self, data: ArrayLike, slice_dims: list[int] = [], chunksize=None, cumulative: bool = False, start=None, stop=None):
        logits = self.evaluate(
            data, slice_dims, chunksize=chunksize, cumulative=cumulative, start=start, stop=stop)
        if slice_dims:  # normalize over slice_dims
            logits -= np.logaddexp.reduce(logits, axis=-1, keepdims=True)

        return logits


class ProbabilityBaggingTreeEstimator(EvalTreeSum):
    def predict(self, data: ArrayLike, slice_dims: list[int] = [], cumulative: bool = False, start=None, stop=None):
        preds = self.evaluate(
            data, slice_dims, cumulative=cumulative, start=start, stop=stop)
        return preds/self.num_trees


def fit_generative_bagging_estimator(
    data: MultivariateDistribution,
    reference_distribution: Optional[MultivariateDistribution] = None,
    num_trees: int = 100,
    criterion: str = 'likelihood',
    max_leaves: int = 32,
    constraints: list = [],
    min_gain: float = 0.,
    splitter: Optional[str] = "best",
    bagging_frac: float = 1,
    feature_frac: float = 1,
    categorical_split_one_vs_all: bool = False,
    seed: int = None,
    num_jobs: int = 1,
    max_num_bytes: str | None = '1M',
):
    prng = np.random.default_rng(seed)

    if criterion == 'likelihood':
        Partitioner = MaxLikelihoodDataPartitioner
    elif criterion == 'brier':
        Partitioner = MinBrierScoreDataPartitioner
    elif criterion == 'midpointrandom':
        Partitioner = MidpointRandomPartitioner
        splitter = 'depth'
    else:
        raise ValueError(
            f"criterion can be 'likelihood' or 'brier' but was {criterion}")

    partitioner = Partitioner(
        constraints=constraints,
        feature_fraction=feature_frac,
        categorical_split_one_vs_all=categorical_split_one_vs_all,
        use_fastpath=False,
        seed=prng
    )

    return fit_bagging_estimator(
        partitioner, data, reference_distribution, num_trees, max_leaves, min_gain, splitter, bagging_frac, prng,
        num_jobs=num_jobs, max_num_bytes=max_num_bytes,
        estimator_class=LogitBaggingTreeEstimator
    )


def fit_bagging_estimator(
    partitioner: DataPartitioner,
    data: MultivariateDistribution,
    reference_distribution: Optional[MultivariateDistribution] = None,
    num_trees: int = 100,
    max_leaves: int = 32,
    min_gain: float = 0.,
    splitter: Optional[str] = "best",
    bagging_frac: Optional[float] = 1,
    seed: int = None,
    num_jobs: int = 1,
    max_num_bytes: str | None = '1M',
    estimator_class: type = EvalTrees,
):
    prng = np.random.default_rng(seed)
    if bagging_frac is not None:
        num_samples = bagging_frac * data.sum_all()

    fitter = TreeFitter(max_leaves, min_gain=min_gain,
                        splitter=splitter, seed=prng)

    def fit_tree(pt, prng):
        # TODO: need to set fitter rng to the rng for this run
        # TODO: maybe leave this outside and iterate over bootstrapped versions of data
        bootstrap_data = data.sample(
            num_samples, seed=prng) if bagging_frac is not None else data

        pt.set_seed(prng)
        # partitioner.prng = prng
        if reference_distribution is None:
            pt.set_data(bootstrap_data)
        else:
            pt.set_data(bootstrap_data, reference_distribution)

        try:
            return fitter.fit(partitioner)
        except NoSplitPossibleException:
            return None

    prngs = prng.spawn(num_trees)
    trees = Parallel(n_jobs=num_jobs, max_nbytes=max_num_bytes, verbose=10)(
        delayed(fit_tree)(partitioner, prng) for prng in prngs
    ) if num_jobs > 0 else tqdm(fit_tree(partitioner, prng) for prng in prngs)

    print("Finished fitting trees")
    built_trees = []
    for tree in trees:
        try:
            assert tree is not None
            built_trees.append(tree.build())
        except AssertionError:
            pass  # either initial splitting failed or too many nodes
    return estimator_class(built_trees)


def delta_likelihood(leaf_sums, leaf_values):
    P, Q, *_ = leaf_sums
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    return np.dot(P, leaf_values) - logsumexp(leaf_values, b=Q)


class GenerativeBoostingTreeEstimator(EvalTreeSum):
    def __init__(self, trees, initial_model=None):
        super().__init__(trees)
        self.initial_model = initial_model

    @property
    def domain(self):
        if self.initial_model is not None:
            return self.initial_model.domain

        return super().domain

    def predict(self, data: ArrayLike, slice_dims: list[int] = [], cumulative: bool = False, normalize: bool = True, start=None, stop=None):
        initial = self.initial_model.predict(
            data, slice_dims=slice_dims, log=True) if self.initial_model is not None else None

        logits = self.evaluate(
            data, slice_dims, cumulative=cumulative, initial=initial, start=start, stop=stop)

        if slice_dims and normalize:  # normalize over slice_dims
            if cumulative:
                logits = (l - np.logaddexp.reduce(l,
                          axis=-1, keepdims=True) for l in logits)
            else:
                logits -= np.logaddexp.reduce(logits, axis=-1, keepdims=True)

        return logits

    line_search_step_sizes = np.logspace(-3, 1, 101)

    def line_search(self, leaf_sums, leaf_values):
        P, Q, *_ = leaf_sums
        P = P / np.sum(P)
        Q = Q / np.sum(Q)

        dL = self.line_search_step_sizes * np.dot(P, leaf_values)
        dL -= logsumexp(self.line_search_step_sizes[:,
                        None] * leaf_values, axis=-1, b=Q)

        idx = np.argmax(dL)
        return self.line_search_step_sizes[idx], dL

    def fit(
        self,
        data: EmpiricalMultivariateDistribution,
        num_trees: int = 200,
        shrinkage: float = 0.15,
        max_leaves: int = 256,
        p_refresh: float = 0.1,
        num_model_samples: int = 80_000,
        num_chains: int = 16,
        line_search: bool = True,
        temperature=1.0,
        initial_samples: str = 'data',
        burn_in=100,
        regularization: MultivariateDistribution = None,
        constraints: list = [],
        min_gain: float = 0.,
        splitter: Optional[str] = "best",
        feature_frac: float = 1,
        categorical_split_one_vs_all: bool = False,
        jit_all: bool = False,
        num_threads: int = 0,
        seed: int = None,
    ):
        if not isinstance(data, EmpiricalMultivariateDistribution):
            raise NotImplementedError
        regularization = [] if regularization is None else [regularization]

        prng = np.random.default_rng(seed)

        data_domain = data.domain
        # TODO: make this work with initial model as None
        if self.initial_model is None:
            self.initial_model = UniformDistribution(data_domain)

        fitter = TreeFitter(max_leaves, min_gain=min_gain,
                            splitter=splitter, seed=prng)

        partitioner = NRGBoostDataPartitioner(
            constraints=constraints,
            feature_fraction=feature_frac,
            domain=data_domain,
            categorical_split_one_vs_all=categorical_split_one_vs_all,
            jit_unordered=jit_all,
            seed=prng
        )
        
        data_samples = data.sum_all()
        dLs = []

        model = None
        log_p_keep = np.log(1 - p_refresh)
        for i in tqdm(range(num_trees)):
            if self.num_trees == 0:
                model = self.initial_model.reweight(
                    num_model_samples)
                assert model.sum_all() == num_model_samples
            elif self.num_trees == 1:  # TODO: handle the case where initial model is uniform differently
                tree = self.trees[0]  # to handle case where i=0
                max_tree = np.max(tree.leaf_values)
                samples = []
                while sum([s.shape[0] for s in samples]) < num_model_samples:
                    new_samples = self.initial_model.sample_points(
                        num_model_samples, seed=prng).astype(np.uint8)
                    log_accept = tree.evaluate(new_samples) - max_tree
                    keep_samples = prng.random(
                        size=num_model_samples) < np.exp(log_accept)
                    samples.append(new_samples[keep_samples])
                samples = np.concatenate(samples, 0)[:num_model_samples]
                model = EmpiricalMultivariateDistribution(
                    self.domain, samples.T)
            elif p_refresh < 1:
                assert isinstance(model, EmpiricalMultivariateDistribution)
                tree = self.trees[-1]  # just to make sure
                log_accept = tree.evaluate(
                    model.data.T) - np.max(tree.leaf_values)
                keep_samples = prng.random(
                    size=model.num_samples) < np.exp(log_accept + log_p_keep)
                samples = model.data[:, keep_samples]
                num_new_samples = num_model_samples - samples.shape[-1]
                # print(f'Sampling {num_new_samples} new samples')
                num_samples_per_chain = num_new_samples // num_chains
                new_samples = self.sample_points(
                    num_samples_per_chain, num_chains=num_chains, temperature=temperature,
                    initial_samples=initial_samples, burn_in=burn_in,
                    data=data, seed=prng, num_threads=num_threads)
                samples = np.concatenate(
                    (model.data[:, keep_samples], new_samples[-num_new_samples:].T), -1)
                model = EmpiricalMultivariateDistribution(
                    self.domain, samples)
            else:  # refresh all
                num_samples_per_chain = num_model_samples // num_chains
                samples = self.sample_points(
                    num_samples_per_chain, num_chains=num_chains, temperature=temperature,
                    initial_samples=initial_samples, burn_in=burn_in,
                    data=data, seed=prng, num_threads=num_threads)[-num_model_samples:]
                model = EmpiricalMultivariateDistribution(
                    self.domain, samples.T)
            # assert model_samples == model.sum_all()
            partitioner.set_data(data, model.rescale(
                data_samples/num_model_samples), *regularization)
            tree = fitter.fit(partitioner).build()

            leaf_sums = partitioner.get_partition_values(
                return_leaf_sums=True)
            if line_search:
                step_size, dL = self.line_search(leaf_sums, tree.leaf_values)
                # print(f'{i}: Optimal step size: {step_size}')

                tree.scale(shrinkage*step_size)
                dLs.append(dL)
            else:
                tree.scale(shrinkage)
                dLs.append(delta_likelihood(leaf_sums, tree.leaf_values))

            self.add(tree)

        return dLs

    def sample_points(self, num_samples_per_chain, num_chains=1, stop_round=None, temperature=1.0, initial_samples='initial', burn_in=0, data=None, seed=None, num_threads=0):
        prng = np.random.default_rng(seed)

        if isinstance(initial_samples, str):
            if initial_samples == 'data':
                initial_samples = data.sample_points(num_chains, seed=prng)
            elif initial_samples == 'uniform':
                initial_samples = UniformDistribution(
                    self.domain).sample_points(num_chains, seed=prng).astype(np.uint8)
            elif initial_samples == 'initial':
                initial_samples = self.initial_model.sample_points(
                    num_chains, seed=prng).astype(np.uint8)
            else:
                raise ValueError(
                    f"Initial sample mthod must be 'data' or 'uniform' but was '{initial_samples}'.")
            if num_chains == 1:
                initial_samples == initial_samples[0]
        else:
            assert isinstance(initial_samples, np.ndarray)

        # TODO: shouldn't need to "recompute" this at every round
        initial_logits = (np.log(m)
                          for m in self.initial_model.encoded_marginals())
        return super().sample_points(num_samples_per_chain, num_chains=num_chains, stop_round=stop_round, temperature=temperature,
                                     initial_logits=initial_logits, initial_samples=initial_samples,
                                     burn_in=burn_in, seed=seed, num_threads=num_threads)

    def sample(self, num_samples_per_chain, num_chains=1, temperature=1.0, initial_samples='uniform', burn_in=0, data=None, seed=None, num_threads=0):
        if self.num_trees == 0:  # shortcut exact sampling
            # samples = self.initial_model.sample(num_samples, seed=seed)
            return self.initial_model.reweight(num_samples_per_chain * num_chains)

        samples = self.sample_points(
            num_samples_per_chain, num_chains=num_chains, temperature=temperature,
            initial_samples=initial_samples, burn_in=burn_in,
            data=data, seed=seed, num_threads=num_threads)
        return EmpiricalMultivariateDistribution(self.domain, samples.T)

    @classmethod
    def from_distribution_marginals(
            cls,
            distribution: MultivariateDistribution,
            uniform_mixture: float = 0.0,
    ):
        if uniform_mixture == 1.0:
            initial_model = UniformDistribution(distribution.domain)
        else:
            initial_model = ProductOfMarginalsDistribution.from_distribution(
                distribution, uniform_mixture=uniform_mixture)
        return cls([], initial_model=initial_model)

    # TODO: start model from a decision tree
    @classmethod
    def from_density_tree(
            cls,
            tree: EvalTree
    ):
        raise NotImplementedError

    def serialize(self):
        return {
            'trees': super().serialize(),
            'initial_model': self.initial_model
        }

    @classmethod
    def deserialize(cls, serialized):
        instance = super(cls, cls).deserialize(serialized['trees'])
        instance.initial_model = serialized['initial_model']
        return instance
