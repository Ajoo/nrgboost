from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from nrgboost.distribution import Distribution
from nrgboost.preprocessing import (
    convert_fixed_point, 
    fit_discretize_dataset, 
    transform_dataset, 
    infer_discretization_types, 
    map_discretization_types,
    pullback_samples
)


@dataclass
class DatasetTransforms:
    columns: list[str]
    fixed_point_digits: dict
    discretization_types: dict
    transforms: dict
    
    @staticmethod
    def fit_transform(
        df,
        num_bins=255,
        infer_fixed_point=True,
        discretization_types=None, 
        infer_ordered_categoricals=False,
        infer_continuous_ordered_categoricals=False,
    ) -> tuple[DatasetTransforms, Distribution]:
        if infer_fixed_point:
            df, fixed_point_digits = convert_fixed_point(df)
        else:
            fixed_point_digits = dict()
            
        if discretization_types is None:
            discretization_types = infer_discretization_types(
                df, 
                max_value=num_bins, 
                infer_ordered_categoricals=infer_ordered_categoricals,
                infer_continuous_ordered_categoricals=infer_continuous_ordered_categoricals
            )
        
        distribution, uniform, transforms = fit_discretize_dataset(
            df, 
            num_bins=num_bins, 
            return_uniform=True, 
            discretization_types=discretization_types
            )
        
        return DatasetTransforms(df.columns.to_list(), fixed_point_digits, discretization_types, transforms), distribution, uniform
    
    def col_number(self, col):
        return self.columns.index(col)
    
    def transform(self, df):
        df, _ = convert_fixed_point(df, digits=self.fixed_point_digits)
        return transform_dataset(df, self.transforms)
    
    __call__ = transform
    
    def inverse_transform(self, data, seed=None):
        col_types = map_discretization_types(self.discretization_types, self.columns)
        df = pullback_samples(data, self.transforms, col_types=col_types, seed=seed)
        for col, d in self.fixed_point_digits.items():
            df[col] = df[col] / 10**d
        return df
        

class Dataset:
    def __init__(self, 
        df, 
        num_bins: int = 255,
        infer_fixed_point: bool = True, 
        discretization_types: Optional[dict] = None, 
        infer_ordered_categoricals: bool = False,
        infer_continuous_ordered_categoricals: bool = False,
        ) -> Dataset:
        """Create a Dataset for training models.
        
        Automatically infers how to discretize data and creates a discrete dataset.
        Internally stores metadata to transform to and from discretized space.

        Args:
            df (Dataframe): Dataset to transform.
            num_bins (int): Maximum cardinality of each dimension. Defaults to 255.
            infer_fixed_point (bool): Whether to automatically infer fixed point precision columns. 
                Defaults to True.
            discretization_types (dict, optional): User specified discretization types. 
                Only for advanced use. Defaults to None.
            infer_ordered_categoricals (bool): Whether to infer integer columns that only
                take a small number of possible discrete values. Defaults to False.
            infer_continuous_ordered_categoricals (bool): Whether to infer floating point columns  
                that only take a small number of possible discrete values. Defaults to False.
        Returns:
            Dataset: Transformed dataset.
        """
        
        self.transform, self.distribution, self.uniform = DatasetTransforms.fit_transform(
            df,
            num_bins=num_bins,
            infer_fixed_point=infer_fixed_point,
            discretization_types=discretization_types,
            infer_ordered_categoricals=infer_ordered_categoricals,
            infer_continuous_ordered_categoricals=infer_continuous_ordered_categoricals
        )
