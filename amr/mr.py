import numpy as np
from .interfaces import RunConfig, MRCandidate
from .functions import get_function
from .data import sample_inputs

EPS = 0.0                      # template noise term; fixed at 0 by default
MIN_COEFF_MASS = 0.5           # |c1|+|c2| must exceed this to be non-trivial
PENALTY_WEIGHT = 10.0


def make_fitness(name: str, config: RunConfig):
    P = get_function(name)
    x1 = sample_inputs(name, config.n_inputs, config.seed)

    def fitness(params: np.ndarray) -> float:
        c1, c2, a, b, d = params
        residual = c1 * P(x1) + c2 * P(a * x1 + b + EPS) + d
        mse = float(np.mean(residual ** 2))
        mass = abs(c1) + abs(c2)
        penalty = PENALTY_WEIGHT * max(0.0, MIN_COEFF_MASS - mass)
        return mse + penalty

    return fitness


def validate(name: str, params, config: RunConfig) -> MRCandidate:
    P = get_function(name)
    x = sample_inputs(name, config.n_inputs, config.seed + 1)   # fresh inputs
    c1, c2, a, b, d = params
    residual = c1 * P(x) + c2 * P(a * x + b + EPS) + d
    max_res = float(np.max(np.abs(residual)))
    non_trivial = (abs(c1) + abs(c2)) >= MIN_COEFF_MASS
    coeffs = {"c1": float(c1), "c2": float(c2), "a": float(a), "b": float(b), "d": float(d)}
    return MRCandidate(coefficients=coeffs,
                       residual=max_res,
                       valid=(max_res < config.tolerance and non_trivial))


def normalize_coefficients(coeffs: dict, decimals: int = 2) -> tuple:
    # Amplitude coefficients (c1, c2, d) are scale-equivalent; transform
    # parameters (a, b) are not.  Normalize only the amplitudes so that
    # [1,1,-1,0,0] and [2,2,-1,0,0] collapse to the same canonical key.
    c1 = float(coeffs["c1"])
    c2 = float(coeffs["c2"])
    a  = float(coeffs["a"])
    b  = float(coeffs["b"])
    d  = float(coeffs["d"])

    # Find scale from amplitudes (c1 first, then c2, then d)
    scale = None
    for v in (c1, c2, d):
        if abs(v) > 0:
            scale = v
            break
    if scale is None or scale == 0:
        scale = 1.0

    nc1 = round(c1 / scale, decimals)
    nc2 = round(c2 / scale, decimals)
    nd  = round(d  / scale, decimals)
    na  = round(a, decimals)
    nb  = round(b, decimals)
    return (nc1, nc2, na, nb, nd)


def deduplicate(coeff_list: list) -> list:
    seen = {}
    for c in coeff_list:
        key = normalize_coefficients(c)
        if key not in seen:
            seen[key] = c
    return list(seen.values())
