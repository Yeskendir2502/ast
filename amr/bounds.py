from .interfaces import FunctionProfile, SearchBounds, ParamBound

def default_bounds() -> SearchBounds:
    return SearchBounds(
        c1=ParamBound(-2.0, 2.0),
        c2=ParamBound(-2.0, 2.0),
        a=ParamBound(-2.0, 2.0),
        b=ParamBound(-10.0, 10.0),
        d=ParamBound(-2.0, 2.0),
    )

def profile_to_bounds(profile: FunctionProfile) -> SearchBounds:
    b = default_bounds()

    # Symmetry implies a reflection relation: compare P(x) against P(-x).
    if profile.symmetry in ("even", "odd"):
        b.a = ParamBound(-1.0, -1.0, fixed=-1.0)
        b.b = ParamBound(0.0, 0.0, fixed=0.0)

    # Periodicity implies a shift relation: same scale, shift within the period.
    if profile.periodic and profile.period:
        p = float(profile.period)
        b.a = ParamBound(1.0, 1.0, fixed=1.0)
        b.b = ParamBound(-p, p)

    return b
