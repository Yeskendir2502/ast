from .interfaces import FunctionProfile, SearchBounds, ParamBound


def default_bounds():
    # start with wide ranges, llm narrows them if use_llm is on
    return SearchBounds(
        c1=ParamBound(-2.0, 2.0),
        c2=ParamBound(-2.0, 2.0),
        a=ParamBound(-2.0, 2.0),
        b=ParamBound(-10.0, 10.0),
        d=ParamBound(-2.0, 2.0),
    )


def profile_to_bounds(profile):
    b = default_bounds()

    # odd functions: P(x) + P(-x) = 0 is a real non-trivial relation
    # even functions: P(x) - P(-x) = 0 is trivially true for ALL mutants, useless
    # so we only fix a=-1 for odd, not even
    if profile.symmetry == "odd":
        b.a = ParamBound(-1.0, -1.0, fixed=-1.0)
        b.b = ParamBound(0.0, 0.0, fixed=0.0)

    # periodic functions have shift relations within one period
    if profile.periodic and profile.period:
        p = float(profile.period)
        b.a = ParamBound(1.0, 1.0, fixed=1.0)
        b.b = ParamBound(-p, p)

    return b
