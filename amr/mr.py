import numpy as np
from .interfaces import RunConfig, MRCandidate
from .functions import get_function, get_domain
from .data import sample_inputs

EPS = 0.0             # template noise term, fixed at 0 for now
MIN_COEFF_MASS = 0.5  # |c1|+|c2| must be above this to be non-trivial
PENALTY_WEIGHT = 10.0


def make_fitness(name, config):
    P = get_function(name)
    lo, hi = get_domain(name)  # need domain to clamp x2, else log funcs get NaN
    x1 = sample_inputs(name, config.n_inputs, config.seed)

    def fitness(params):
        c1, c2, a, b, d = params
        x2 = np.clip(a * x1 + b + EPS, lo, hi)  # clamp to valid domain
        residual = c1 * P(x1) + c2 * P(x2) + d
        mse = float(np.mean(residual ** 2))
        mass = abs(c1) + abs(c2)
        penalty = PENALTY_WEIGHT * max(0.0, MIN_COEFF_MASS - mass)  # penalize trivial zeros
        return mse + penalty

    return fitness


def validate(name, params, config):
    P = get_function(name)
    lo, hi = get_domain(name)
    x = sample_inputs(name, config.n_inputs, config.seed + 1)  # fresh inputs for validation
    c1, c2, a, b, d = params
    x2 = np.clip(a * x + b + EPS, lo, hi)  # same clamping as in fitness
    residual = c1 * P(x) + c2 * P(x2) + d
    max_res = float(np.max(np.abs(residual)))
    non_trivial = (abs(c1) + abs(c2)) >= MIN_COEFF_MASS
    coeffs = {"c1": float(c1), "c2": float(c2), "a": float(a), "b": float(b), "d": float(d)}
    return MRCandidate(coefficients=coeffs,
                       residual=max_res,
                       valid=(max_res < config.tolerance and non_trivial))


def normalize_coefficients(coeffs, decimals=2):
    # normalize amplitude params (c1, c2, d) so scaled duplicates collapse to same key
    # we dont normalize a and b - they are transform params, not amplitudes
    c1 = float(coeffs["c1"])
    c2 = float(coeffs["c2"])
    a  = float(coeffs["a"])
    b  = float(coeffs["b"])
    d  = float(coeffs["d"])

    # find scale from c1 first, then c2, then d
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


def deduplicate(coeff_list):
    seen = {}
    for c in coeff_list:
        key = normalize_coefficients(c)
        if key not in seen:
            seen[key] = c
    return list(seen.values())
