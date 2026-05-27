import numpy as np
from amr.interfaces import RunConfig
from amr.mr import make_fitness, validate, normalize_coefficients, deduplicate


def test_fitness_low_for_known_odd_relation_sin():
    # sin(x) + sin(-x) = 0 so params [1,1,-1,0,0] should give near zero fitness
    cfg = RunConfig(n_inputs=300, seed=5)
    fitness = make_fitness("sin", cfg)
    good = fitness(np.array([1.0, 1.0, -1.0, 0.0, 0.0]))
    assert good < 1e-6


def test_fitness_penalizes_trivial_zero_solution():
    cfg = RunConfig(n_inputs=100, seed=5)
    fitness = make_fitness("sin", cfg)
    trivial = fitness(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert trivial > 1.0  # penalty should dominate here


def test_validate_accepts_true_relation():
    cfg = RunConfig(n_inputs=300, tolerance=1e-3, seed=9)
    cand = validate("sin", [1.0, 1.0, -1.0, 0.0, 0.0], cfg)
    assert cand.valid is True
    assert cand.residual < 1e-3


def test_validate_rejects_false_relation():
    cfg = RunConfig(n_inputs=300, tolerance=1e-3, seed=9)
    cand = validate("sin", [1.0, 1.0, 1.0, 0.5, 0.0], cfg)
    assert cand.valid is False


def test_validate_rejects_constant_oracle():
    # a~0 makes P(ax+b) constant, i.e. a point oracle abs(b)=const, not a true MR
    cfg = RunConfig(n_inputs=300, tolerance=1e-3, seed=9)
    cand = validate("abs", [0.0, -0.572, 0.0, -3.33, 1.904], cfg)
    assert cand.valid is False


def test_normalize_and_deduplicate_collapse_scaled_duplicates():
    a = {"c1": 1.0, "c2": 1.0, "a": -1.0, "b": 0.0, "d": 0.0}
    b = {"c1": 2.0, "c2": 2.0, "a": -1.0, "b": 0.0, "d": 0.0}  # scaled copy of a
    na, nb = normalize_coefficients(a), normalize_coefficients(b)
    assert na == nb
    unique = deduplicate([a, b])
    assert len(unique) == 1
