import numpy as np

EPS = 0.0


def mutant_killed(mr_params, mutant_fn, inputs, tolerance):
    c1, c2, a, b, d = mr_params
    residual = c1 * mutant_fn(inputs) + c2 * mutant_fn(a * inputs + b + EPS) + d
    return bool(np.max(np.abs(residual)) > tolerance)  # true if MR is violated


def kill_rate(mr_param_list, mutant_fns, inputs, tolerance):
    if not mutant_fns:
        return 0.0
    killed = 0
    for m in mutant_fns:
        # we count a mutant as killed if any of our MRs detect it
        if any(mutant_killed(p, m, inputs, tolerance) for p in mr_param_list):
            killed += 1
    return killed / len(mutant_fns)


def mr_precision(valid, total):
    if total == 0:
        return 0.0
    return valid / total * 100.0
