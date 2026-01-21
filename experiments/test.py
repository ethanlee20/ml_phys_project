
from pathlib import Path

import pandas
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Colormap, Normalize
from matplotlib.cm import ScalarMappable
import torch

from lib_sbi_btokstll.training import Trainer, Dataset, select_device, open_model_state_dict
from lib_sbi_btokstll.predictor import Predictor
from lib_sbi_btokstll.data import to_torch_tensor, calculate_discrete_label_weights_uniform_prior, normalize_using_reference_data, bin
from lib_sbi_btokstll.models import MLP


if __name__ == "__main__":

    train = False
    name = "test_dc9_only_5_bins"
    sim_type = "gen"
    label_name = "dc9"
    feature_names = ["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]
    dc9_interval = (-2, 1)
    num_bins = 5
    device = select_device()
    optimizer = "adam"
    loss_fn = "cross_entropy"
    learn_rate = 3e-4
    learn_rate_sched = "reduce_lr_on_plateau"
    learn_rate_sched_reduce_factor = 0.98
    learn_rate_sched_patience = 0
    learn_rate_sched_threshold = 0
    learn_rate_sched_eps = 0
    epochs = 300
    epochs_checkpoint = 100
    batch_size_train = 10_000
    batch_size_eval = 10_000
    path_to_parent_dir = "data/models"
    path_to_data = "data/preprocessed.parquet"
    
    data = pandas.read_parquet(path_to_data).xs(sim_type, level="sim_type")
    data = data[
        (data["interval_dc7_lb"]==0) 
        & (data["interval_dc7_ub"]==0)
        & (data["interval_dc10_lb"]==0)
        & (data["interval_dc10_ub"]==0)
    ]

    bins = numpy.linspace(*dc9_interval, num_bins+1)
    binned_labels = bin(data[label_name], bins)
    
    normalized_features = normalize_using_reference_data(
        data[feature_names], 
        data[feature_names].xs("train", level="split")
    )

    dataset_train = Dataset(
        features=normalized_features.xs("train", level="split").astype("float32"), 
        labels=binned_labels.xs("train", level="split")["bin_index"]
    )
    dataset_val = Dataset(
        features=normalized_features.xs("val", level="split").astype("float32"), 
        labels=binned_labels.xs("val", level="split")["bin_index"]
    )
        
    loss_label_weights = calculate_discrete_label_weights_uniform_prior(
        binned_labels.xs("train", level="split")["bin_index"]
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
        path_to_final_model_state_dict = Path(path_to_parent_dir).joinpath(f"{name}/final.pt")
        model.load_state_dict(open_model_state_dict(path_to_final_model_state_dict))

    for eval_sets_split in ["train", "val"]:

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

        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.dpi": 400, 
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern",
        })

        def bars(ax, x, y, color):
            ax.hlines(y, xmin=x-0.5, xmax=x+0.5, color=color)

        fig, ax = plt.subplots()
        norm = Normalize(vmin=0, vmax=num_bins-1)
        cmap = mpl.colormaps["viridis"]
        for l, label in zip(log_probs, eval_sets_labels["bin_index"].to_list()):
            color = cmap(norm(label))
            # bars(ax, numpy.array(range(num_bins)), l.cpu(), color=color)
            ax.plot(range(num_bins), l.cpu(), color=color)
        ax.set_xticks(range(num_bins))
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_ticks(range(num_bins))
        cbar.set_label(r"Actual $\delta C_9$ Bin", fontsize=15)
        ax.set_xlabel(r"$\delta C_9$ Bin", fontsize=15)
        ax.set_ylabel(r"$\log P(\delta C_9 \, | \, \textrm{dataset})$", fontsize=15)
        plt.savefig(Path(path_to_parent_dir).joinpath(f"{name}/predictions_sets_{eval_sets_split}.png"), bbox_inches="tight")
        plt.close()