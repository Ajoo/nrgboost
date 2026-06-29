from dataclasses import dataclass
from typing import Optional

import numpy as np
import joblib

from nrgboost.tree.ensemble import GenerativeBoostingTreeEstimator
from nrgboost.tree.generative import constraints as c
from nrgboost.dataset import Dataset, DatasetTransforms

import warnings


def train(dataset: Dataset, params: dict, seed: Optional[int] = None):
    """Trains a NRGBoost model.

    Args:
        dataset (Dataset): Trainind dataset.
        params (dict): Training parameters. The main options are as follows:
        
            Fitting parameters
            - num_trees (int): Total number of trees to fit. Defaults to 200.
            - shrinkage (float): Constant scaling factor for each step. Defaults to 0.15.
            - line_search (bool): Use a line search to find optimal step size. 
                Shrinkage factor is applied on top. Defaults to True.
            - max_leaves (int): Maximum number of leaves for each tree. Defaults to 256.
            - max_ratio_in_leaf (float): Maximum ratio of data / model data per leaf. Defaults to 2.
            - min_data_in_leaf (float): Minimum number of data points per leaf. Defaults to 0.
            - initial_uniform_mixture (float): Mixture coeficient for the starting point of boosting:
                - 0 means starting from the product of training marginals.
                - 1 means starting from a uniform distribution.
                Defaults to 0.1.
            - categorical_split_one_vs_all (bool): If True, categorical splits are always one vs all.
                Otherwise they are many vs many. Defaults to False.
            - feature_frac (float): Fraction of features to randomly consider for splitting each node. Defaults to 1.
            - splitter (str): Determines how trees are grown. "best" is best first 
                and "depth" is breadth first. Defaults to "best".
            
            Sampling parameters
            - num_model_samples (int): Defaults to 80_000.
            - p_refresh (float): Fraction of samples to independently refresh at each round. 
              Defaults to 0.1.
            - num_chains (int): Number of chains used to draw samples. Defaults to 16.
            - burn_in (int): Number of samples to burn at start of each chain. Defaults to 100.
            - num_threads (int): Defaults to 0.
            
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        NRGBooster: Trained model.
    """
    # setup constraints
    max_ratio_in_leaf = params.pop('max_ratio_in_leaf', 2)
    constraints = [c.max_ratio_in_leaf(max_ratio_in_leaf)]
    
    min_data_in_leaf = params.pop('min_data_in_leaf', 0)
    if min_data_in_leaf > 0:
        constraints.append(c.min_data_in_leaf(min_data_in_leaf))
    
    # set up initial model
    mixing_coef = params.pop('initial_uniform_mixture', 0.1)
    bst = GenerativeBoostingTreeEstimator.from_distribution_marginals(
                    dataset.distribution, mixing_coef)
    
    bst.fit(
        dataset.distribution, 
        **params, 
        constraints=constraints, 
        seed=seed
        )
    
    return NRGBooster(bst, dataset.transform)


@dataclass
class NRGBooster:
    """Implements NRGBoost Model.

    Static Methods:
        fit: Fits and returns a new NRGBoost model.
        load: Loads an existing model from disk.

    Methods:
        predict: Computes pointwise predictions for a trained model.
        sample: Computes samples from a trained model. 
        save: Saves a trained model to disk.
    """
    booster: GenerativeBoostingTreeEstimator
    transform: DatasetTransforms
    
    def __repr__(self):
        return repr(self.booster)
    
    def __str__(self):
        return str(self.booster)
    
    def _expectation(self, col):
        discretization_types = self.transform.discretization_types
        transforms = self.transform.transforms
        
        is_discrete = (col in discretization_types['discrete_numerical']) or \
            (col in discretization_types['discrete_quantized'])
            
        if is_discrete:
            bin_centers = (transforms['num'][col][:-1] + transforms['num'][col][1:] - 1)/2
        else:
            bin_centers = (transforms['num'][col][:-1] + transforms['num'][col][1:])/2
            
        if col in self.transform.fixed_point_digits:
            bin_centers = bin_centers / 10**self.transform.fixed_point_digits[col]
        
        def expectation(p):
            return np.sum(bin_centers*np.exp(p), -1)
        
        return expectation
    
    def predict(self, 
        df, 
        col: str = None, 
        num_rounds: Optional[int] = None, 
        normalize: bool = True, 
        cumulative: bool = False
        ):
        """Compute predictions for a given set of datapoints.

        Args:
            df (Dataframe): Datapoints to compute predictions.
            col (str, optional): Compute predictions along a given dimension (specified by column name).
                Will ignore values of col in df. Defaults to None.
            num_rounds (int, optional): Includes only the first num_rounds trees in prediction. 
                Defaults to None which uses all trees.
            normalize (bool, optional): Whether to normalize 
            cumulative (bool, optional): If True returns an iterator over predictions at different
                num_rounds. Defaults to False.

        Returns:
            np.ndarray: Predictions for each datapoint. Output depends on type of col:
                - None: outputs energies at each datapoint (unnormalized log probabilities).
                - Categorical column: outputs logits for each possible class. 
                  The output shape will be (df.shape[0], K) where K is the cardinality of col.
                - Numerical column: outputs expected value for that column.
        """
        data = self.transform(df)
        
        slice_dims = [] if col is None else [self.transform.col_number(col)]
        is_numerical = col is not None and self.booster.domain.ordered[slice_dims[0]]
        if is_numerical and not normalize:
            warnings.warn(
                "Called predict with normalize=False for a numerical column. "
                "This setting will be ignored."
            )
            normalize = True
            
        preds = self.booster.predict(
            data, 
            slice_dims=slice_dims, 
            stop=num_rounds, 
            cumulative=cumulative, 
            normalize=normalize
            )
        
        # if no slice dim or is categorical return preds
        if not is_numerical: 
            return preds            
            
        # for numericals, compute the expected value for predicted distribution
        # TODO: support other types of numerical prediction (e.g., median)
        compute_expectation = self._expectation(col)
        if cumulative:
            return map(compute_expectation, preds)
        return compute_expectation(preds)
        
    def sample(self, 
        num_samples: int, 
        num_steps: int = 100, 
        num_rounds: Optional[int] = None, 
        temperature: float = 1.0, 
        num_threads: int = 0, 
        output_full_chain: bool = False, 
        seed: Optional[int] = None
        ):
        """Draws approximate samples from the model.

        Args:
            num_samples (int): Number of samples to draw.
            num_steps (int, optional): Number of Gibbs Sampling steps to run per sample. Defaults to 100.
            num_rounds (Optional[int], optional): Includes only the first num_rounds trees when sampling. 
                Defaults to None.
            temperature (float, optional): Temperature scales the model. Defaults to 1.0.
            num_threads (int, optional): Number of threads to use. Defaults to 0 which uses the default for openmp.
            output_full_chain (bool, optional): If True, outputs the entire chain used to draw each sample 
                instead of only the output of the last step. Defaults to False.
            seed (Optional[int], optional): Seed to use when sampling. Defaults to None.

        Returns:
            Dataframe: Samples from the model. If output_full_chain is True this will have include 
            num_samples * nums_steps samples in total with all samples from the same chain being consecutive.
        """
        prng = np.random.default_rng(seed)
        data = self.booster.sample_points(
            num_steps, 
            num_samples, 
            stop_round=num_rounds, 
            initial_samples='initial',
            temperature=temperature,
            seed=prng, 
            num_threads=num_threads
            )
        if not output_full_chain:
            data = data.reshape((num_samples, num_steps, -1))[:, -1]
        return self.transform.inverse_transform(data, seed=prng)
    
    def save(self, filename: str):
        """Save model to disk.

        Args:
            filename (str): Filename to save model in.
        """
        serialized = {
            'version': '0.0',
            'booster': self.booster.serialize(),
            'transform': self.transform
        }
        joblib.dump(serialized, filename)
    
    @staticmethod
    def load(filename: str):
        """Load model from disk.

        Args:
            filename (str): Filename to load model from.

        Returns:
            NRGBooster: Model.
        """
        serialized = joblib.load(filename)
        assert serialized['version'] == '0.0', (
            "Trying to load a model saved with a later version of NRGBoost."
        )
        return NRGBooster(
            GenerativeBoostingTreeEstimator.deserialize(serialized['booster']),
            serialized['transform']
        )
        
    fit = staticmethod(train)


