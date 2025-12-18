
from unittest import TestCase, main

import numpy
import pandas


class Interval:

    def __init__(self, lb, ub):

        assert lb <= ub

        self.lb = lb
        self.ub = ub
        self.tuple = (lb, ub)


class TestInterval(TestCase):

    def test_basic_instaniation(self):
        interval = Interval(0, 10)
        self.assertEqual(interval.lb, 0)
        self.assertEqual(interval.ub, 10)

    def test_wrong_instantiation_lb_greater_than_ub(self):
        with self.assertRaises(AssertionError):
            Interval(0, -10)


def sample_from_uniform_wilson_coefficient_prior(
    interval_dc7:Interval,
    interval_dc9:Interval,
    interval_dc10:Interval,
    n_samples:int,
    rng_seed:int=42,
):
  
    rng = numpy.random.default_rng(rng_seed)

    intervals = {
        "dc7": interval_dc7,
        "dc9": interval_dc9,
        "dc10": interval_dc10
    }

    samples = {
        dci : rng.uniform(interval.lb, interval.ub, n_samples)
        for dci, interval in intervals.items() 
    }

    df_samples = pandas.DataFrame(samples)
    return df_samples


class TestSampleFromUniformWilsonCoefficientPrior(TestCase):

    def setUp(self):
        self.n_samples = 100
        self.rng_seed = 42
        self.interval_dc7 = Interval(-1, 0)
        self.interval_dc9 = Interval(1, 2)
        self.interval_dc10 = Interval(3, 4)
        self.samples = sample_from_uniform_wilson_coefficient_prior(
            self.interval_dc7, 
            self.interval_dc9, 
            self.interval_dc10,
            self.n_samples,
            rng_seed=self.rng_seed
        )

    def test_correct_number_of_samples(self):
        self.assertEqual(len(self.samples), self.n_samples)

    def test_samples_in_correct_intervals(self):
        for dci, interval in zip(["dc7", "dc9", "dc10"], [self.interval_dc7, self.interval_dc9, self.interval_dc10]):
            self.assertTrue(
                (self.samples[dci].min() >= interval.lb) 
                and (self.samples[dci].max() <= interval.ub)
            )


def apply_offline_cuts(
    dataframe:pandas.DataFrame, 
    mbc_lower_bound:float, 
    deltaE_keep_interval:Interval,
    muon_id_lower_bound:float,
    pion_id_lower_bound:float,
    inv_mass_pi_lepton_remove_interval:Interval,
):

    """
    Signal: Mbc > 5.27, -0.05 for mu or -0.075 for e < delta E < 0.05
    Mbc sideband: 4.5 or 5.2 < Mbc < 5.26
    """

    dataframe = dataframe[dataframe["Mbc"] > mbc_lower_bound]
    dataframe = dataframe[(dataframe["deltaE"] > deltaE_keep_interval.lb) & (dataframe["deltaE"] < deltaE_keep_interval.ub)]
    return dataframe.copy()


if __name__ == "__main__":

    main()

