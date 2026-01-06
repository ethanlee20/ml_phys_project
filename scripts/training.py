
import pandas
from torch import float32

from lib_sbi_btokstll.training import Trainer, Dataset, select_device
from lib_sbi_btokstll.models import MLP


if __name__ == "__main__":


    data = pandas.read_parquet("data/combined_processed.parquet")

    trial_intervals = {"train": (0, 309), "val": (100_000, 100_009)} # inclusive 
    level = "gen"

    data_splits = {
        split : data.loc[trial_intervals[split][0]:trial_intervals[split][1],:,level] 
        for split in ["train", "val"]
    }
    datasets = {
        split : Dataset(
            features=data_split[["q_squared", "cos_theta_mu", "cos_theta_k", "chi"]], 
            labels=data_split[["dc7", "dc9", "dc10"]],
            dtype=float32
        )
        for split, data_split in data_splits.items()
    }
    # breakpoint()

    model = MLP()

    params = {
        "name": "test",
        "parent_dir": "data/models",
        "optimizer": "adam",
        "optimizer_params": {"lr": 3e-4},
        "loss_fn": "mse",
        "batch_sizes": {"train": 10, "eval": 10},
        "epochs": 10,
        "checkpoint_epochs": 2,
        "lr_scheduler": "reduce_lr_on_plateau",
        "lr_scheduler_params": {"factor":0.95, "patience":0, "threshold":0, "eps":0},
    }

    trainer = Trainer(datasets["train"], datasets["val"], model, params)

    device = select_device()

    trainer.train(device)