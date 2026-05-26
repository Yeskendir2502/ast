import numpy as np
from amr.interfaces import (
    FunctionProfile, ParamBound, SearchBounds, MRCandidate, PSOResult, RunConfig,
)

def test_searchbounds_as_arrays_returns_lows_and_highs_in_param_order():
    b = SearchBounds(
        c1=ParamBound(-2, 2), c2=ParamBound(-2, 2),
        a=ParamBound(-1, 1), b=ParamBound(-10, 10), d=ParamBound(-2, 2),
    )
    lows, highs = b.as_arrays()
    assert list(lows) == [-2, -2, -1, -10, -2]
    assert list(highs) == [2, 2, 1, 10, 2]

def test_parambound_fixed_collapses_bounds():
    pb = ParamBound(-5, 5, fixed=1.0)
    assert pb.effective_bounds() == (1.0, 1.0)

def test_runconfig_defaults():
    c = RunConfig()
    assert c.model == "qwen2.5:7b-instruct"
    assert c.use_llm is True
    assert c.n_inputs == 200
