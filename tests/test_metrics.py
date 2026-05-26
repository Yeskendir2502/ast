import numpy as np
from amr.metrics import mutant_killed, kill_rate, mr_precision

def test_mutant_killed_detects_offset_fault_on_sin():
    # MR: sin(x) + sin(-x) = 0. Mutant adds 0.01 -> relation broken -> killed.
    base = np.sin
    mutant = lambda x: np.sin(x) + 0.01
    x = np.linspace(-3, 3, 50)
    params = [1.0, 1.0, -1.0, 0.0, 0.0]
    assert mutant_killed(params, mutant, x, tolerance=1e-3) is True

def test_mutant_not_killed_when_relation_still_holds():
    # negate mutant: -sin(x). For sin(x)+sin(-x)=0, -sin still satisfies it.
    mutant = lambda x: -np.sin(x)
    x = np.linspace(-3, 3, 50)
    params = [1.0, 1.0, -1.0, 0.0, 0.0]
    assert mutant_killed(params, mutant, x, tolerance=1e-3) is False

def test_kill_rate_counts_unique_killed_mutants():
    x = np.linspace(-3, 3, 50)
    mr_params = [[1.0, 1.0, -1.0, 0.0, 0.0]]
    mutants = [lambda x: np.sin(x) + 0.01,   # killed
               lambda x: -np.sin(x)]         # not killed
    rate = kill_rate(mr_params, mutants, x, tolerance=1e-3)
    assert rate == 0.5

def test_mr_precision():
    assert mr_precision(valid=3, total=4) == 75.0
    assert mr_precision(valid=0, total=0) == 0.0
