import numpy as np
from amr.pso import optimize
from amr.interfaces import SearchBounds, ParamBound, RunConfig


def _wide_bounds():
    return SearchBounds(
        c1=ParamBound(-5, 5), c2=ParamBound(-5, 5),
        a=ParamBound(-5, 5), b=ParamBound(-5, 5), d=ParamBound(-5, 5),
    )


def test_pso_minimizes_sphere_near_zero():
    # sphere function has its minimum at the origin
    cfg = RunConfig(n_particles=40, max_iterations=200, seed=1)
    res = optimize(lambda p: float(np.sum(p ** 2)), _wide_bounds(), cfg)
    assert res.best_fitness < 1e-2
    assert all(abs(v) < 0.2 for v in res.best_params)


def test_pso_respects_fixed_param():
    # fix a=3 and check that pso keeps it there
    bounds = _wide_bounds()
    bounds.a = ParamBound(3.0, 3.0, fixed=3.0)
    cfg = RunConfig(n_particles=30, max_iterations=120, seed=2)
    res = optimize(lambda p: float(np.sum(p ** 2)), bounds, cfg)
    assert res.best_params[2] == 3.0


def test_pso_records_history_and_iteration_count():
    cfg = RunConfig(n_particles=20, max_iterations=50, seed=3)
    res = optimize(lambda p: float(np.sum(p ** 2)), _wide_bounds(), cfg)
    assert len(res.history) >= 1
    assert res.iterations >= 1
    assert res.history[-1] <= res.history[0]  # fitness shouldnt get worse
