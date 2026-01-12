
from itertools import product

from numpy.random import default_rng
from pandas import DataFrame, Series, cut

from lib_sbi_btokstll.util import to_torch_tensor, Interval
from lib_sbi_btokstll.constants import trial_ranges, names_of_features


def sample_from_uniform_wilson_coefficient_prior(
    interval_dc7:Interval,
    interval_dc9:Interval,
    interval_dc10:Interval,
    n_samples:int,
    rng_seed=None,
):
  
    rng = default_rng(rng_seed)
    intervals = {
        "dc7": interval_dc7,
        "dc9": interval_dc9,
        "dc10": interval_dc10
    }
    samples = {
        dci : rng.uniform(interval.lb, interval.ub, n_samples)
        for dci, interval in intervals.items() 
    }
    df_samples = DataFrame(samples)
    return df_samples


def get_split(trial):

    split = []
    for split_, range_ in trial_ranges.items():
        if trial in range_:
            split.append(split_)
    if len(split) != 1: raise ValueError(f"Trial not in known split. Trial: {trial}")
    return split[0]


def bin(data:Series, bins):

    binned_indices = cut(
        data,
        bins,
        labels=False,
        include_lowest=True
    )
    binned_intervals = cut(
        data, 
        bins, 
        labels=None, 
        include_lowest=True
    )
    binned_mids = binned_intervals.apply(lambda interval : interval.mid)
    binned = DataFrame(
        {
            "original": data,
            "bin_index": binned_indices, 
            "bin_interval": binned_intervals, 
            "bin_mid": binned_mids
        }
    )
    return binned


def calculate_discrete_label_weights_uniform_prior(labels:Series):

    normalized_label_counts = to_torch_tensor(labels.value_counts(normalize=True).sort_index())
    weights = 1 / normalized_label_counts
    return weights


def normalize_using_reference_data(data:DataFrame|Series, reference:DataFrame|Series):

    means = reference.mean()
    stds = reference.std()
    normalized = (data - means) / stds
    return normalized


def open_output_root_file(path, tree_names=["gen", "det"]):
    
    """
    Open an output root file as a pandas dataframe.

    Each tree will be labeled by a 
    pandas multi-index.
    """

    with uproot.open(path) as file:
        list_of_dataframes = [
            file[name].arrays(library="pd") 
            for name in tree_names
        ]

    final_dataframe = pandas.concat(list_of_dataframes, keys=tree_names, names=["sim_type",])
    return final_dataframe



