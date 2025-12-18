
from pathlib import Path

from lib_sbi_btokstll import Interval, sample_from_uniform_wilson_coefficient_prior


### Parameters ###

wilson_coefficient_intervals = {
    "dc7": Interval(-0.5, 0.5),
    "dc9": Interval(-2, 1),
    "dc10": Interval(-1, 1)
}

n_samples = 10_000_000
rng_seed = 42


### Sampling ###

sampled_wilson_coefficients_dataframe = sample_from_uniform_wilson_coefficient_prior(
    interval_dc7=wilson_coefficient_intervals["dc7"],
    interval_dc9=wilson_coefficient_intervals["dc9"],
    interval_dc10=wilson_coefficient_intervals["dc10"],
    n_samples=n_samples,
    rng_seed=rng_seed
)


### Saving to disk ###

path_to_data_dir = Path("/Users/elee/Desktop/ml_phys_project/data/")

file_name = "sampled_wilson_coefficients.parquet"

file_path = path_to_data_dir.joinpath(file_name)

sampled_wilson_coefficients_dataframe.to_parquet(file_path)



