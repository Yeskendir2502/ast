import numpy as np
from .interfaces import RunConfig, MRCandidate
from .functions import get_function, get_domain, needs_clamp
from .data import sample_inputs

EPS = 0.0             # template noise term, fixed at 0 for now
MIN_COEFF_MASS = 0.5  # |c1|+|c2| must be above this to be non-trivial
PENALTY_WEIGHT = 10.0
MIN_SPREAD = 0.5      # x1 and x2 must be this far apart on average
MIN_A = 0.1           # |a| must be away from 0, else P(ax+b) is constant - a point oracle not a real MR


def make_fitness(name, config):
    P = get_function(name)
    lo, hi = get_domain(name)
    clamp = needs_clamp(name)  # only log funcs need clamping, sin/cos are defined everywhere
    x1 = sample_inputs(name, config.n_inputs, config.seed)

    def fitness(params):
        c1, c2, a, b, d = params
        x2_raw = a * x1 + b + EPS
        x2 = np.clip(x2_raw, lo, hi) if clamp else x2_raw  # dont clamp sin/cos - breaks shift MRs
        residual = c1 * P(x1) + c2 * P(x2) + d
        mse = float(np.mean(residual ** 2))
        mass = abs(c1) + abs(c2)
        penalty = PENALTY_WEIGHT * max(0.0, MIN_COEFF_MASS - mass)  # penalize trivial zeros
        # penalize degenerate case where x1 and x2 are basically the same point
        # without this, PSO finds P(x) - P(x) = 0 which kills nothing
        spread = float(np.mean(np.abs(x2 - x1)))
        spread_penalty = PENALTY_WEIGHT * max(0.0, MIN_SPREAD - spread)
        # penalize a~0: that turns P(ax+b) into a constant, so the relation is just
        # an oracle P(b)=const, not a real metamorphic relation that transforms the input
        a_penalty = PENALTY_WEIGHT * max(0.0, MIN_A - abs(a))
        return mse + penalty + spread_penalty + a_penalty

    return fitness


def validate(name, params, config):
    P = get_function(name)
    lo, hi = get_domain(name)
    clamp = needs_clamp(name)
    x = sample_inputs(name, config.n_inputs, config.seed + 1)  # fresh inputs for validation
    c1, c2, a, b, d = params
    x2_raw = a * x + b + EPS
    x2 = np.clip(x2_raw, lo, hi) if clamp else x2_raw
    spread = float(np.mean(np.abs(x2 - x)))
    residual = c1 * P(x) + c2 * P(x2) + d
    max_res = float(np.max(np.abs(residual)))
    non_trivial = (abs(c1) + abs(c2)) >= MIN_COEFF_MASS
    diverse = spread > 0.1            # reject MRs where both evals are at the same point
    genuine = abs(a) >= MIN_A         # reject a~0 oracles, we only want true transforming MRs
    coeffs = {"c1": float(c1), "c2": float(c2), "a": float(a), "b": float(b), "d": float(d)}
    return MRCandidate(coefficients=coeffs,
                       residual=max_res,
                       valid=(max_res < config.tolerance and non_trivial and diverse and genuine))


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
