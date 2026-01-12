
from pathlib import Path

import pandas
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import torch
from torch.nn import Module, Sequential, Linear, ReLU

from lib_sbi_btokstll.training import Trainer, Dataset, select_device, open_model_state_dict
from lib_sbi_btokstll.predictor import Predictor
from lib_sbi_btokstll.util import to_torch_tensor
from lib_sbi_btokstll.data import calculate_discrete_label_weights_uniform_prior, normalize_using_reference_data, bin
from lib_sbi_btokstll.constants import names_of_features, delta_wilson_coefficient_intervals
from lib_sbi_btokstll.models import MLP


if __name__ == "__main__":

    train = False
    name = "test"
    sim_type = "gen"
    label_name = "dc9"
    num_bins = 5
    device = select_device()
    optimizer = "adam"
    loss_fn = "cross_entropy"
    learn_rate = 5e-4
    learn_rate_sched = "reduce_lr_on_plateau"
    learn_rate_sched_reduce_factor = 0.95
    learn_rate_sched_patience = 0
    learn_rate_sched_threshold = 0
    learn_rate_sched_eps = 0
    epochs = 300
    epochs_checkpoint = 10
    batch_size_train = 10_000
    batch_size_eval = 10_000
    eval_sets_split = "val"
    path_to_parent_dir = "data/models"
    path_to_data = "data/combined_processed.parquet"
    
    data = pandas.read_parquet(path_to_data).xs(sim_type, level="sim_type")

    wc_interval = delta_wilson_coefficient_intervals[label_name]
    bins = numpy.linspace(wc_interval.lb, wc_interval.ub, num_bins+1)
    binned_labels = bin(data[label_name], bins)
    
    normalized_features = normalize_using_reference_data(
        data[names_of_features], 
        data[names_of_features] #.xs("train", level="split")
    )

    dataset_train = Dataset(
        features=normalized_features.astype("float32"), 
        labels=binned_labels["bin_index"]
    )
    dataset_val = Dataset(
        features=normalized_features.xs("val", level="split").astype("float32"), 
        labels=binned_labels.xs("val", level="split")["bin_index"]
    )
        
    loss_label_weights = calculate_discrete_label_weights_uniform_prior(
        binned_labels["bin_index"] # binned_labels.xs("train", level="split")["bin_index"]
    ).to(torch.float32).to(device)

    params = {
        "name": name,
        "parent_dir": path_to_parent_dir,
        "optimizer": optimizer,
        "optimizer_params": {"lr": learn_rate},
        "loss_fn": loss_fn,
        "loss_fn_params": {"weight": loss_label_weights},
        "batch_sizes": {"train": batch_size_train, "eval": batch_size_eval},
        "epochs": epochs,
        "checkpoint_epochs": epochs_checkpoint,
        "lr_scheduler": learn_rate_sched,
        "lr_scheduler_params": {
            "factor":learn_rate_sched_reduce_factor, 
            "patience":learn_rate_sched_patience, 
            "threshold":learn_rate_sched_threshold, 
            "eps":learn_rate_sched_eps
        },
    }

    model = MLP()

    if train:
        trainer = Trainer(dataset_train, dataset_val, model, params)
        trainer.train(device)
    else:
        path_to_final_model_state_dict = Path(path_to_parent_dir).joinpath("test/final.pt")
        model.load_state_dict(open_model_state_dict(path_to_final_model_state_dict))

    eval_sets_features = numpy.concatenate(
        [
            numpy.expand_dims(trial_data.to_numpy(), 0)
            for _, trial_data in normalized_features.xs(eval_sets_split, level="split").astype("float32")
            .groupby("trial")
        ]
    )
    eval_sets_labels = binned_labels.xs(eval_sets_split, level="split").groupby("trial").first()
    eval_sets_dataset = Dataset(eval_sets_features, eval_sets_labels["bin_index"])

    predictor = Predictor(model, eval_sets_dataset.features, device)
    log_probs = predictor.calc_log_probs()

    print(log_probs)
    print(eval_sets_labels)

    offset = 0
    for l, label, color in zip(log_probs, eval_sets_labels["bin_index"], TABLEAU_COLORS):
        plt.plot(range(num_bins), l.cpu(), color=color)
        plt.axvline(label+offset, color=color)
        offset += 0.02
    plt.show()
