
from lib_sbi_btokstll.util import Interval


trial_ranges = {
    "train" : range(0, 10_000),  # vary all wc 0-10_000, vary c9 only 10_000-20_000, vary c7 only 20_000-30_000, vary c10 only 30_000-40_000
    "val" : list(range(100_000, 110_000))+[-1], 
    "test": range(200_000, 210_000),
}

names_of_features = [
    "q_squared", 
    "cos_theta_mu", 
    "cos_theta_k", 
    "chi"
]

delta_wilson_coefficient_intervals = {
    "dc7": Interval(-0.5, 0.5),
    "dc9": Interval(-2, 1),
    "dc10": Interval(-1, 1),
}