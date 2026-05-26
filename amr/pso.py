import numpy as np
from .interfaces import SearchBounds, RunConfig, PSOResult

def optimize(fitness_fn, bounds: SearchBounds, config: RunConfig) -> PSOResult:
    rng = np.random.default_rng(config.seed)
    lows, highs = bounds.as_arrays()
    dim = len(lows)
    n = config.n_particles
    span = np.where(highs > lows, highs - lows, 1.0)

    pos = rng.uniform(lows, highs, size=(n, dim))
    vel = rng.uniform(-span, span, size=(n, dim)) * 0.1

    fit = np.array([fitness_fn(p) for p in pos])
    pbest = pos.copy()
    pbest_fit = fit.copy()
    g = int(np.argmin(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = float(pbest_fit[g])

    history = [gbest_fit]
    no_improve = 0
    converged = False
    iterations = 0

    for it in range(config.max_iterations):
        iterations = it + 1
        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))
        vel = (config.inertia * vel
               + config.cognitive * r1 * (pbest - pos)
               + config.social * r2 * (gbest - pos))
        pos = np.clip(pos + vel, lows, highs)

        fit = np.array([fitness_fn(p) for p in pos])
        improved = fit < pbest_fit
        pbest[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]

        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gbest_fit - config.convergence_eps:
            gbest = pbest[g].copy()
            gbest_fit = float(pbest_fit[g])
            no_improve = 0
        else:
            no_improve += 1

        history.append(gbest_fit)
        if no_improve >= config.convergence_window:
            converged = True
            break

    return PSOResult(
        best_params=gbest.tolist(),
        best_fitness=gbest_fit,
        iterations=iterations,
        converged=converged,
        history=history,
    )
