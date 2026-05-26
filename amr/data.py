import numpy as np
from .functions import get_domain

def sample_inputs(name: str, n: int, seed: int = 42) -> np.ndarray:
    lo, hi = get_domain(name)
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=n)
