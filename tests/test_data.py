import numpy as np
from amr.data import sample_inputs
from amr.functions import get_domain


def test_samples_within_domain():
    x = sample_inputs("log10", n=500, seed=1)
    lo, hi = get_domain("log10")
    assert x.shape == (500,)
    assert x.min() >= lo
    assert x.max() <= hi


def test_sampling_is_reproducible():
    a = sample_inputs("sin", n=100, seed=7)
    b = sample_inputs("sin", n=100, seed=7)
    assert np.array_equal(a, b)  # same seed should give same result


def test_different_seeds_differ():
    a = sample_inputs("sin", n=100, seed=1)
    b = sample_inputs("sin", n=100, seed=2)
    assert not np.array_equal(a, b)
