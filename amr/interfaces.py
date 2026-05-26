from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class FunctionProfile:
    name: str
    symmetry: str = "none"          # "even" | "odd" | "none"
    periodic: bool = False
    period: Optional[float] = None
    monotonic: str = "none"         # "increasing" | "decreasing" | "none"
    domain: tuple = (-100.0, 100.0)
    range: tuple = (-100.0, 100.0)

@dataclass
class ParamBound:
    lower: float
    upper: float
    fixed: Optional[float] = None

    def effective_bounds(self):
        if self.fixed is not None:
            return (self.fixed, self.fixed)
        return (self.lower, self.upper)

@dataclass
class SearchBounds:
    c1: ParamBound
    c2: ParamBound
    a: ParamBound
    b: ParamBound
    d: ParamBound

    def as_arrays(self):
        lows, highs = [], []
        for pb in (self.c1, self.c2, self.a, self.b, self.d):
            lo, hi = pb.effective_bounds()
            lows.append(lo)
            highs.append(hi)
        return np.array(lows, dtype=float), np.array(highs, dtype=float)

@dataclass
class MRCandidate:
    coefficients: dict       # {"c1":..,"c2":..,"a":..,"b":..,"d":..}
    residual: float
    valid: bool = False

@dataclass
class PSOResult:
    best_params: list
    best_fitness: float
    iterations: int
    converged: bool
    history: list

@dataclass
class RunConfig:
    model: str = "qwen2.5:7b-instruct"
    n_inputs: int = 200
    tolerance: float = 1e-3
    n_particles: int = 50
    max_iterations: int = 350
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    seed: int = 42
    use_llm: bool = True
    out_dir: str = "results"
    convergence_window: int = 30
    convergence_eps: float = 1e-8
