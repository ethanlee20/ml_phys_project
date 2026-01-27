

import numpy 
import pandas
import torch
import uproot


def to_torch_tensor(x):

    """Convert to torch tensor."""
    
    def torch_tensor_from_pandas(dataframe):
        """
        Convert a pandas dataframe to a torch tensor.
        """
        tensor = torch.from_numpy(dataframe.to_numpy())
        return tensor

    if isinstance(x, pandas.DataFrame | pandas.Series):
        return torch_tensor_from_pandas(x)
    elif isinstance(x, numpy.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else: raise ValueError(f"Unsupported type: {type(x)}")


def bin(data:pandas.Series, bins):

    """Bin data using given bins."""

    binned_indices = pandas.cut(
        data,
        bins,
        labels=False,
        include_lowest=True
    )
    binned_intervals = pandas.cut(
        data, 
        bins, 
        labels=None, 
        include_lowest=True
    )
    binned_mids = binned_intervals.apply(lambda interval : interval.mid)
    binned = pandas.DataFrame(
        {
            "original": data,
            "bin_index": binned_indices, 
            "bin_interval": binned_intervals, 
            "bin_mid": binned_mids
        }
    )
    return binned


def calculate_discrete_label_weights_for_uniform_prior(labels:pandas.Series):

    """Calculate label weights for reweighting classes to uniform distribution."""

    normalized_label_counts = to_torch_tensor(
        labels.value_counts(normalize=True).sort_index()
    )
    weights = 1 / normalized_label_counts
    return weights


def normalize_using_reference_data(data:pandas.DataFrame|pandas.Series, reference:pandas.DataFrame|pandas.Series):

    """Standard scale a dataset using the mean and standard deviation of a reference dataset."""

    means = reference.mean()
    stds = reference.std()
    normalized = (data - means) / stds
    return normalized


def open_simulated_data_root_file(path, unwanted_keys=["persistent;1", "persistent;2"]):
    
    """
    Open a simulated data root file as a pandas dataframe.

    Each tree will be labeled by a pandas multi-index.
    """

    with uproot.open(path) as file:

        keys = [
            key.split(";")[0] for key in file.keys() 
            if key not in unwanted_keys
        ]
        tree_dataframes = [
            file[key].arrays(library="pd") 
            for key in keys
        ]

    dataframe = pandas.concat(tree_dataframes, keys=keys, names=["sim_type",])
    return dataframe



