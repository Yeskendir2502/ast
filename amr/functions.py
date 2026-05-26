from typing import Callable
import numpy as np

_FUNCTIONS: dict[str, Callable] = {
    "abs": np.abs,
    "asinh": np.arcsinh,
    "atan": np.arctan,
    "cos": np.cos,
    "log1p": np.log1p,
    "log10": np.log10,
    "sin": np.sin,
    "tan": np.tan,
}

# Domains chosen to stay numerically well-behaved (avoid asymptotes / undefined regions).
_DOMAINS = {
    "abs": (-100.0, 100.0),
    "asinh": (-100.0, 100.0),
    "atan": (-100.0, 100.0),
    "cos": (-2.0 * np.pi, 2.0 * np.pi),
    "log1p": (-0.999, 100.0),
    "log10": (1e-6, 100.0),
    "sin": (-2.0 * np.pi, 2.0 * np.pi),
    "tan": (-1.5, 1.5),
}

FUNCTION_NAMES: list[str] = list(_FUNCTIONS.keys())


def get_function(name: str) -> Callable:
    return _FUNCTIONS[name]


def get_domain(name: str) -> tuple[float, float]:
    return _DOMAINS[name]
