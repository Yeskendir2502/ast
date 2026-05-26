import numpy as np
import pytest
from amr.functions import FUNCTION_NAMES, get_function, get_domain


def test_eight_functions_registered():
    assert sorted(FUNCTION_NAMES) == sorted(
        ["abs", "asinh", "atan", "cos", "log1p", "log10", "sin", "tan"]
    )


def test_get_function_computes_known_values():
    assert get_function("sin")(0.0) == pytest.approx(0.0)
    assert get_function("cos")(0.0) == pytest.approx(1.0)
    assert get_function("abs")(-3.0) == pytest.approx(3.0)
    assert get_function("log10")(100.0) == pytest.approx(2.0)


def test_domain_within_valid_range_for_log10():
    lo, hi = get_domain("log10")
    assert lo > 0  # log10 is undefined at 0 and below


def test_unknown_function_raises():
    with pytest.raises(KeyError):
        get_function("does_not_exist")
