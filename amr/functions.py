import numpy as np

# all 8 functions from the apache commons math benchmark
_FUNCTIONS = {
    "abs": np.abs,
    "asinh": np.arcsinh,
    "atan": np.arctan,
    "cos": np.cos,
    "log1p": np.log1p,
    "log10": np.log10,
    "sin": np.sin,
    "tan": np.tan,
}

# we picked these domains to stay away from undefined/asymptotic regions
_DOMAINS = {
    "abs": (-100.0, 100.0),
    "asinh": (-100.0, 100.0),
    "atan": (-100.0, 100.0),
    "cos": (-2.0 * np.pi, 2.0 * np.pi),
    "log1p": (-0.999, 100.0),    # x > -1 for log1p to be defined
    "log10": (1e-6, 100.0),      # cant use 0 or negatives here
    "sin": (-2.0 * np.pi, 2.0 * np.pi),
    "tan": (-1.5, 1.5),          # avoids blowup near +-pi/2
}

FUNCTION_NAMES = list(_FUNCTIONS.keys())

# only log functions blow up outside their domain - all others are safe everywhere
_NEEDS_CLAMP = {"log10", "log1p"}


def get_function(name):
    return _FUNCTIONS[name]


def get_domain(name):
    return _DOMAINS[name]


def needs_clamp(name):
    return name in _NEEDS_CLAMP
