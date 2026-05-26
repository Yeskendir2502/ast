from dataclasses import dataclass
import numpy as np

# shared data structures - all modules import from here


@dataclass
class FunctionProfile:
    # what the llm tells us about a function's math properties
    name: str
    symmetry: str = "none"       # "even" | "odd" | "none"
    periodic: bool = False
    period: float = None         # only set when periodic is true
    monotonic: str = "none"
    domain: tuple = (-100.0, 100.0)
    output_range: tuple = (-100.0, 100.0)


@dataclass
class ParamBound:
    lower: float
    upper: float
    fixed: float = None  # if set, pso wont move this param at all

    def effective_bounds(self):
        if self.fixed is not None:
            return (self.fixed, self.fixed)  # collapse to a single point
        return (self.lower, self.upper)


@dataclass
class SearchBounds:
    c1: ParamBound
    c2: ParamBound
    a: ParamBound
    b: ParamBound
    d: ParamBound

    def as_arrays(self):
        # we need lows and highs as numpy arrays for the pso
        lows, highs = [], []
        for pb in (self.c1, self.c2, self.a, self.b, self.d):
            lo, hi = pb.effective_bounds()
            lows.append(lo)
            highs.append(hi)
        return np.array(lows, dtype=float), np.array(highs, dtype=float)


@dataclass
class MRCandidate:
    coefficients: dict   # {"c1":..,"c2":..,"a":..,"b":..,"d":..}
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
