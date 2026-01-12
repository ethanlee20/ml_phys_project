
from unittest import TestCase, main

import numpy
import pandas
import torch


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


def safer_convert_to_int(x):

    assert x.is_integer()
    return int(x)


class TestSaferConvertToInt(TestCase):

    def test_basic(self):
        self.assertIs(safer_convert_to_int(3.0), 3)

    def test_non_integer(self):
        with self.assertRaises(AssertionError):
            safer_convert_to_int(3.2)


    
def are_instance(objects:list, classinfo):

    assert isinstance(objects, list)

    for obj in objects:
        if not isinstance(obj, classinfo):
            return False
    return True


class TestAreInstance(TestCase):

    def test_basic_are_instance(self):

        self.assertTrue(are_instance([1,2,3], int))
        self.assertFalse(are_instance([1,2,3], float))


def torch_tensor_from_pandas(dataframe):

    """
    Convert a pandas dataframe to a torch tensor.
    """

    tensor = torch.from_numpy(dataframe.to_numpy())
    return tensor


def to_torch_tensor(x):

    if isinstance(x, pandas.DataFrame|pandas.Series):
        return torch_tensor_from_pandas(x)
    elif isinstance(x, numpy.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else: raise ValueError(f"Unsupported type: {type(x)}")



if __name__ == "__main__":

    main()

