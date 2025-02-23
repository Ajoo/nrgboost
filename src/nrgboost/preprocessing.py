from collections import defaultdict
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.signal import lfilter

from nrgboost.distribution import EmpiricalMultivariateDistribution, ProductOfMarginalsDistribution
from nrgboost.domain import FiniteIntSet, FiniteIntRange, ProductDomain


def convert_fixed_point(df, rtol=1e-3, digits=None):
    if digits is None:
        digits = {}
        for col in df.columns:
            if df[col].dtype == 'category' or not np.issubdtype(df[col].dtype, np.floating):
                continue
            d = check_discrete(df[col].values, rtol=rtol, return_digits=True)
            if d >= 0:
                digits[col] = d
        
    df = df.copy()
    for col, d in digits.items():
        df[col] = (df[col] * 10**d).round().astype(int)
            
    return df, digits


def check_discrete(a, start=0, stop=5, rtol=0.1, return_digits=False):
    powers = np.arange(start, stop+1)
    d = 10.**-powers[:, None]
    r = a % d
    r = np.minimum(r, d-r)

    m = np.max(r, axis=-1) < rtol*d[:, 0]
    dlevel = np.argmax(m)
    if dlevel == 0 and not m[0]:
        return -1

    return powers[dlevel] if return_digits else d[dlevel, 0]


def process_categoricals(df, cols):
    categorical_values = {}
    for col in cols:
        categorical_values[col] = df[col].cat.categories.values
        df[col] = df[col].cat.codes

    return categorical_values

# bins are only used


def process_ordered_categoricals(df, cols):
    bins = {}
    for col in cols:
        values = np.unique(df[col].values)
        # bins[col] = np.append(values, values[-1] + 1)
        b = (values[:-1] + values[1:]) / 2
        bins[col] = np.insert(np.append(b, values[-1] + 1), 0, values[0])
        df[col] = np.digitize(df[col], bins[col][1:-1])

    return bins


def process_continuous_ordered_categoricals(df, cols):
    bins = {}
    for col in cols:
        values = np.unique(df[col].values)
        # bins[col] = np.append(values, values[-1] + 1)
        b = (values[:-1] + values[1:]) / 2
        bins[col] = np.insert(np.append(b, values[-1]), 0, values[0])
        df[col] = np.digitize(df[col], bins[col][1:-1])

    return bins


# TODO: handle non integer precision
def process_discrete_numerical(df, cols, shift=0):
    bins = {}
    for col in cols:
        a, b = df[col].min(), df[col].max()
        bins[col] = np.arange(a-shift, b+2-shift)
        df[col] = np.digitize(df[col], bins[col][1:-1])

    return bins


def fit_quantize_numericals(df, num_bins, cols=None, discrete=False):
    numerical_bins = {}
    for col in cols:
        if discrete:
            numerical_bins[col], df[col] = quantile_discretize_discrete(
                df[col], num_bins)
        else:
            numerical_bins[col], df[col] = quantile_discretize(
                df[col], num_bins)
    return numerical_bins


def quantile_discretize_discrete(col, num_bins):
    # method = 'closest_observation' if discrete else 'linear'
    method = 'linear'
    bins = np.quantile(col, np.linspace(0, 1, num_bins + 1), method=method)
    bins = np.ceil(bins).astype(np.int_)

    bins[np.append(np.insert(np.diff(bins[:-1]) == 0, 0, False), True)] += 1
    bin_mask = np.insert(np.diff(bins) > 0, 0, True)
    bins = bins[bin_mask]
    return bins, np.digitize(col, bins[1:-1])


def quantile_discretize(col, num_bins, discrete=False):
    # method = 'closest_observation' if discrete else 'linear'
    method = 'linear'
    bins = np.quantile(col, np.linspace(0, 1, num_bins + 1), method=method)
    if discrete:
        bins = np.ceil(bins).astype(np.int_)

    bin_mask = np.append(np.insert(np.diff(bins[:-1]) > 0, 0, True), True)
    bins = bins[bin_mask]
    if discrete:  # include max
        bins[-1] += 1
    return bins, np.digitize(col, bins[1:-1])


def infer_discretization_types(df, max_value=255, infer_ordered_categoricals=True, infer_continuous_ordered_categoricals=False):
    columns = defaultdict(list)

    for col in df:
        column = df[col]
        if column.dtype == 'category':
            columns['categorical'].append(col)
        elif np.issubdtype(column.dtype, np.integer) or (column.round() == column).all():
            if column.max() - column.min() <= max_value:
                columns['discrete_numerical'].append(col)
            elif infer_ordered_categoricals and (column.nunique() <= max_value):
                columns['ordered_categorical'].append(col)
            else:
                columns['discrete_quantized'].append(col)
        else:
            assert np.issubdtype(column.dtype, np.floating)
            if infer_continuous_ordered_categoricals and (column.nunique() <= max_value):
                columns['continuous_ordered_categorical'].append(col)
            else:
                columns['continuous_quantized'].append(col)
    return columns


def map_discretization_types(discretization_types, columns):
    col_types = {}

    for col in columns:
        for tp, cols in discretization_types.items():
            if col in cols:
                col_types[col] = tp
    return col_types


def pullback_samples(samples, transforms, col_types, seed=None):
    prng = np.random.default_rng(seed)

    samples_df = {}
    for values, (col_name, col_type) in zip(samples.T, col_types.items()):
        if col_type == 'categorical':
            categories = transforms['cat'][col_name]
            samples_df[col_name] = pd.Categorical(
                categories[values], categories=categories)
            continue
        bins = transforms['num'][col_name]
        if col_type.endswith('ordered_categorical'):
            bins = np.insert(
                lfilter([2], [1, 1], bins[1:-1], zi=[-bins[0]])[0], 0, bins[0])
            if col_type == 'ordered_categorical': # discrete
                bins = np.round(bins).astype(np.int_)
            samples_df[col_name] = bins[values]
        elif col_type == 'discrete_numerical':
            samples_df[col_name] = bins[values]
        elif col_type == 'discrete_quantized':
            samples_df[col_name] = prng.integers(bins[values], bins[values+1])
        elif col_type == 'continuous_quantized':
            samples_df[col_name] = prng.uniform(bins[values], bins[values+1])

    return pd.DataFrame(samples_df)


def auto_discretize_dataset(
    df,
    num_bins=255,
    as_dataframe=False,
    return_uniform=False,
):
    discretization_types = infer_discretization_types(df, num_bins)
    outputs = fit_discretize_dataset(
        df, num_bins, discretization_types, as_dataframe, return_uniform)
    return *outputs, map_discretization_types(discretization_types, df.columns)


def fit_discretize_dataset(
        df,
        num_bins=255,
        discretization_types=None,
        as_dataframe=False,
        return_uniform=False,
):
    if discretization_types is None:
        discretization_types = infer_discretization_types(df, num_bins)

    df = df.copy()

    transforms = {}
    transforms['num'] = process_discrete_numerical(
        df, discretization_types['discrete_numerical'],)
    transforms['num'].update(fit_quantize_numericals(
        df, num_bins, cols=discretization_types['discrete_quantized'], discrete=True))
    transforms['num'].update(fit_quantize_numericals(
        df, num_bins, cols=discretization_types['continuous_quantized']))
    transforms['num'].update(process_ordered_categoricals(
        df, cols=discretization_types['ordered_categorical']))
    transforms['num'].update(process_continuous_ordered_categoricals(
        df, cols=discretization_types['continuous_ordered_categorical']))

    # print(transforms['num'].keys())

    if as_dataframe:
        return df, transforms

    transforms['cat'] = process_categoricals(
        df, cols=discretization_types['categorical'])

    output = df.astype(np.uint8)

    domains = []
    marginals = []
    for col in df:
        if col in transforms['num']:
            bins = transforms['num'][col]
            cardinality = bins.size - 1
            domain = FiniteIntRange(cardinality)
            domains.append(domain)

            marginals.append(np.diff(bins)/(bins[-1] - bins[0]))
        elif col in transforms['cat']:
            domain = FiniteIntSet(transforms['cat'][col].size)
            domains.append(domain)
            marginals.append(np.full(domain.cardinality, 1/domain.cardinality))
        else:
            print(col)
            print(transforms)
            raise TypeError

    domain = ProductDomain(domains)
    data = np.ascontiguousarray(output.values.T)
    empirical = EmpiricalMultivariateDistribution(domain, data)
    if return_uniform:
        uniform = ProductOfMarginalsDistribution(
            domain, marginals, weight=empirical.num_samples)
        return empirical, uniform, transforms

    return empirical, transforms


def transform_dataset(
    df,
    transforms,
    as_dataframe=False
):
    df = df.copy()

    for col, bins in transforms['num'].items():
        df[col] = np.digitize(df[col], bins[1:-1]).astype(np.uint8)
    for col, values in transforms['cat'].items():
        assert np.array_equal(df[col].cat.categories.values, values)
        df[col] = df[col].cat.codes.astype(np.uint8)

    return df if as_dataframe else df.values
