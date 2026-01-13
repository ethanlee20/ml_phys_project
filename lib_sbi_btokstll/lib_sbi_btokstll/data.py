

import numpy 
import pandas
from pandas import DataFrame, Series, cut
import torch
import uproot

from lib_sbi_btokstll.constants import trial_ranges


def torch_tensor_from_pandas(dataframe):

    """
    Convert a pandas dataframe to a torch tensor.
    """

    tensor = torch.from_numpy(dataframe.to_numpy())
    return tensor


def to_torch_tensor(x):

    if isinstance(x, DataFrame|Series):
        return torch_tensor_from_pandas(x)
    elif isinstance(x, numpy.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else: raise ValueError(f"Unsupported type: {type(x)}")



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



