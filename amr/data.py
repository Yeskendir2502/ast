import numpy as np
from .functions import get_domain


def sample_inputs(name, n, seed=42):
    lo, hi = get_domain(name)
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=n)  # uniform draw from valid domain
