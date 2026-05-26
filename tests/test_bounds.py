from amr.interfaces import FunctionProfile
from amr.bounds import default_bounds, profile_to_bounds

def test_default_bounds_are_wide_and_free():
    b = default_bounds()
    lows, highs = b.as_arrays()
    assert lows[2] < highs[2]            # a is free
    assert b.a.fixed is None

def test_even_function_fixes_reflection():
    # even: P(x) == P(-x) -> search reflection a=-1, b=0
    prof = FunctionProfile(name="cos", symmetry="even")
    b = profile_to_bounds(prof)
    assert b.a.effective_bounds() == (-1.0, -1.0)
    assert b.b.effective_bounds() == (0.0, 0.0)

def test_periodic_function_bounds_shift_to_period():
    import numpy as np
    prof = FunctionProfile(name="sin", symmetry="odd", periodic=True,
                           period=2 * np.pi)
    b = profile_to_bounds(prof)
    # scale fixed to 1, shift bounded by the period
    assert b.a.effective_bounds() == (1.0, 1.0)
    lo, hi = b.b.effective_bounds()
    assert lo == -2 * np.pi and hi == 2 * np.pi
