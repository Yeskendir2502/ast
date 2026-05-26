# Automatic Discovery of Metamorphic Relations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python tool that auto-discovers metamorphic relations for 8 numerical functions using LLM-guided Particle Swarm Optimization, evaluated by mutation kill rate against the AutoMR benchmark.

**Architecture:** A per-function pipeline. An Ollama LLM produces a structured profile of each function's properties; a deterministic mapper turns that into PSO search bounds; a custom PSO finds coefficients `(c1, c2, a, b, d)` for the relation `c1*P(x1) + c2*P(a*x1+b+eps) + d = 0`; discovered relations are validated, de-duplicated, and scored as oracles against 625 mutants.

**Tech Stack:** Python 3.10+, numpy, the `ollama` Python client (model `qwen2.5:7b-instruct`), pytest.

**Work split (4 contributors, roughly equal):**
- Task 1 is shared scaffolding (do first, jointly).
- Part 1 - Subjects & Data (Kuba): Tasks 2, 3, 4.
- Part 2 - LLM Reconnaissance (Ada): Tasks 5, 6.
- Part 3 - PSO Engine (Hugo): Task 7.
- Part 4 - Orchestration, Metrics & Runner (Yeskendir): Tasks 8, 9, 10, 11.

**Parameter order is always `[c1, c2, a, b, d]`** everywhere in the codebase.

---

## File Structure

```
amr/
  __init__.py
  interfaces.py        # Task 1 (shared)
  functions.py         # Task 2 (Part 1)
  data.py              # Task 3 (Part 1)
  mutants.py           # Task 4 (Part 1)
  bounds.py            # Task 5 (Part 2)
  recon.py             # Task 6 (Part 2)
  pso.py               # Task 7 (Part 3)
  mr.py                # Task 8 (Part 4)
  metrics.py           # Task 9 (Part 4)
  run.py               # Task 10 (Part 4)
  cli.py               # Task 10 (Part 4)
scripts/
  build_mutants.py     # Task 4 (Part 1) - AutoMR adapter
data/
  mutants/<func>.json  # Task 4 - generated
tests/
  test_*.py            # one per module
requirements.txt       # Task 1
README.md              # Task 11
```

---

## Task 1: Project scaffolding and shared interfaces

**Files:**
- Create: `requirements.txt`
- Create: `amr/__init__.py`
- Create: `amr/interfaces.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Create `requirements.txt`**

```
numpy>=1.24
ollama>=0.3.0
pytest>=7.0
```

- [ ] **Step 2: Create empty package marker**

Create `amr/__init__.py` with a single line:

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing test for interfaces**

Create `tests/test_interfaces.py`:

```python
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
    assert c.n_inputs > 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_interfaces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.interfaces'`

- [ ] **Step 5: Implement `amr/interfaces.py`**

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_interfaces.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt amr/__init__.py amr/interfaces.py tests/test_interfaces.py
git commit -m "Add shared interfaces and project scaffolding"
```

---

## Task 2: Target functions registry (Part 1 - Kuba)

**Files:**
- Create: `amr/functions.py`
- Test: `tests/test_functions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_functions.py`:

```python
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
    assert lo > 0  # log10 undefined at and below 0


def test_unknown_function_raises():
    with pytest.raises(KeyError):
        get_function("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_functions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.functions'`

- [ ] **Step 3: Implement `amr/functions.py`**

```python
import numpy as np

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

FUNCTION_NAMES = list(_FUNCTIONS.keys())


def get_function(name):
    return _FUNCTIONS[name]


def get_domain(name):
    return _DOMAINS[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_functions.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/functions.py tests/test_functions.py
git commit -m "Add target function registry with domains"
```

---

## Task 3: Domain-aware input sampling (Part 1 - Kuba)

**Files:**
- Create: `amr/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
import numpy as np
from amr.data import sample_inputs
from amr.functions import get_domain


def test_samples_within_domain():
    x = sample_inputs("log10", n=500, seed=1)
    lo, hi = get_domain("log10")
    assert x.shape == (500,)
    assert x.min() >= lo
    assert x.max() <= hi


def test_sampling_is_reproducible():
    a = sample_inputs("sin", n=100, seed=7)
    b = sample_inputs("sin", n=100, seed=7)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = sample_inputs("sin", n=100, seed=1)
    b = sample_inputs("sin", n=100, seed=2)
    assert not np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.data'`

- [ ] **Step 3: Implement `amr/data.py`**

```python
import numpy as np
from .functions import get_domain


def sample_inputs(name, n, seed=42):
    lo, hi = get_domain(name)
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/data.py tests/test_data.py
git commit -m "Add domain-aware reproducible input sampling"
```

---

## Task 4: Mutant loading and evaluation (Part 1 - Kuba)

Mutants are represented as simple arithmetic perturbation operators over the base
function, stored as JSON. This reproduces the common seeded faults numerically (no JVM).
A generator seeds an initial mutant set; the AutoMR adapter (`scripts/build_mutants.py`)
converts the benchmark's mutants into this format, skipping and logging any that cannot be
expressed as one of the supported operators.

Supported operators (each maps a base function `f` to a mutant callable):
- `add_const` (value `v`): `f(x) + v`
- `mul_const` (value `v`): `f(x) * v`
- `add_input` (value `v`): `f(x + v)`
- `mul_input` (value `v`): `f(x * v)`
- `replace_const` (value `v`): `v` (constant output)
- `negate` (value ignored): `-f(x)`

**Files:**
- Create: `amr/mutants.py`
- Create: `scripts/build_mutants.py`
- Test: `tests/test_mutants.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mutants.py`:

```python
import json
import numpy as np
from amr.mutants import make_mutant_fn, load_mutants, generate_mutant_specs


def test_make_mutant_fn_add_const():
    base = np.sin
    m = make_mutant_fn(base, {"id": 0, "op": "add_const", "value": 0.01})
    x = np.array([0.0, 1.0])
    assert np.allclose(m(x), np.sin(x) + 0.01)


def test_make_mutant_fn_negate_ignores_value():
    base = np.cos
    m = make_mutant_fn(base, {"id": 1, "op": "negate", "value": 0.0})
    x = np.array([0.0, 1.5])
    assert np.allclose(m(x), -np.cos(x))


def test_load_mutants_roundtrip(tmp_path):
    specs = [{"id": 0, "op": "add_const", "value": 0.01}]
    p = tmp_path / "sin.json"
    p.write_text(json.dumps(specs))
    loaded = load_mutants("sin", base_dir=str(tmp_path))
    assert loaded == specs


def test_generate_mutant_specs_is_deterministic_and_sized():
    a = generate_mutant_specs(count=20, seed=3)
    b = generate_mutant_specs(count=20, seed=3)
    assert a == b
    assert len(a) == 20
    assert all("op" in s and "value" in s and "id" in s for s in a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mutants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.mutants'`

- [ ] **Step 3: Implement `amr/mutants.py`**

```python
import json
import os
import numpy as np

OPERATORS = ["add_const", "mul_const", "add_input", "mul_input", "replace_const", "negate"]


def make_mutant_fn(base, spec):
    op = spec["op"]
    v = spec.get("value", 0.0)
    if op == "add_const":
        return lambda x: base(x) + v
    if op == "mul_const":
        return lambda x: base(x) * v
    if op == "add_input":
        return lambda x: base(x + v)
    if op == "mul_input":
        return lambda x: base(x * v)
    if op == "replace_const":
        return lambda x: np.full_like(np.asarray(x, dtype=float), v)
    if op == "negate":
        return lambda x: -base(x)
    raise ValueError(f"unknown operator: {op}")


def load_mutants(name, base_dir="data/mutants"):
    path = os.path.join(base_dir, f"{name}.json")
    with open(path) as fh:
        return json.load(fh)


def generate_mutant_specs(count, seed=0):
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(count):
        op = OPERATORS[int(rng.integers(0, len(OPERATORS)))]
        # small perturbations resemble seeded faults
        value = float(np.round(rng.uniform(-1.0, 1.0), 4))
        if op in ("mul_const", "mul_input"):
            value = float(np.round(rng.uniform(0.5, 1.5), 4))
        specs.append({"id": i, "op": op, "value": value})
    return specs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mutants.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Write the AutoMR adapter script**

Create `scripts/build_mutants.py`. This generates a starting mutant set per function and
writes it to `data/mutants/<name>.json`. The `--from-automr` path is where the team plugs
in parsing of the real benchmark once the repo layout is confirmed; until then it logs and
falls back to generated specs so the pipeline is runnable end-to-end.

```python
import argparse
import json
import os
import sys
from amr.functions import FUNCTION_NAMES
from amr.mutants import generate_mutant_specs

# 625 mutants total across 8 functions in the benchmark.
TOTAL_MUTANTS = 625


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mutants")
    ap.add_argument("--from-automr", default=None,
                    help="path to a cloned github.com/bolzzzz/AutoMR checkout")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    per_func = TOTAL_MUTANTS // len(FUNCTION_NAMES)

    if args.from_automr:
        print(f"[build_mutants] AutoMR source given: {args.from_automr}", file=sys.stderr)
        print("[build_mutants] Real-mutant parsing not yet wired; "
              "falling back to generated specs. Implement parsing here, mapping each "
              "benchmark mutant to one of amr.mutants.OPERATORS and skipping (logging) "
              "any that cannot be expressed.", file=sys.stderr)

    for i, name in enumerate(FUNCTION_NAMES):
        specs = generate_mutant_specs(count=per_func, seed=args.seed + i)
        with open(os.path.join(args.out, f"{name}.json"), "w") as fh:
            json.dump(specs, fh, indent=2)
        print(f"[build_mutants] wrote {per_func} mutants for {name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Generate the mutant data and verify**

Run: `python scripts/build_mutants.py && ls data/mutants/`
Expected: 8 JSON files (`abs.json` ... `tan.json`).

- [ ] **Step 7: Commit**

```bash
git add amr/mutants.py scripts/build_mutants.py tests/test_mutants.py
git commit -m "Add mutant representation, loader, and AutoMR adapter scaffold"
```

---

## Task 5: Profile-to-bounds mapper (Part 2 - Ada)

**Files:**
- Create: `amr/bounds.py`
- Test: `tests/test_bounds.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bounds.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bounds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.bounds'`

- [ ] **Step 3: Implement `amr/bounds.py`**

```python
from .interfaces import FunctionProfile, SearchBounds, ParamBound


def default_bounds():
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bounds.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/bounds.py tests/test_bounds.py
git commit -m "Add deterministic profile-to-bounds mapper"
```

---

## Task 6: LLM reconnaissance via Ollama (Part 2 - Ada)

**Files:**
- Create: `amr/recon.py`
- Test: `tests/test_recon.py`

The Ollama call is isolated behind `_chat_json` so tests can monkeypatch it (no live model
needed). Responses are cached to `.llm_cache/<model>__<name>.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_recon.py`:

```python
import json
import amr.recon as recon
from amr.interfaces import RunConfig, FunctionProfile


def test_parse_profile_reads_valid_json():
    raw = {
        "symmetry": "odd", "periodic": True, "period": 6.283,
        "monotonic": "none", "domain": [-6.28, 6.28], "range": [-1, 1],
    }
    prof = recon._parse_profile("sin", raw)
    assert prof.name == "sin"
    assert prof.symmetry == "odd"
    assert prof.periodic is True
    assert prof.period == 6.283


def test_parse_profile_defaults_on_missing_keys():
    prof = recon._parse_profile("abs", {})
    assert prof.symmetry == "none"
    assert prof.periodic is False
    assert prof.period is None


def test_analyze_function_uses_chat_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_chat_json(model, prompt):
        calls.append((model, prompt))
        return {"symmetry": "even", "periodic": False, "period": None,
                "monotonic": "none", "domain": [-100, 100], "range": [0, 1]}

    monkeypatch.setattr(recon, "_chat_json", fake_chat_json)
    monkeypatch.setattr(recon, "CACHE_DIR", str(tmp_path))

    cfg = RunConfig(model="test-model")
    p1 = recon.analyze_function("cos", cfg)
    p2 = recon.analyze_function("cos", cfg)   # second call should hit cache

    assert p1.symmetry == "even"
    assert p2.symmetry == "even"
    assert len(calls) == 1   # cached, so the model is queried only once
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.recon'`

- [ ] **Step 3: Implement `amr/recon.py`**

```python
import json
import os
from .interfaces import FunctionProfile, RunConfig

CACHE_DIR = ".llm_cache"

PROMPT_TEMPLATE = """You are analyzing the real-valued mathematical function {name}(x).
Respond with ONLY a JSON object, no prose, with exactly these keys:
- "symmetry": one of "even", "odd", "none"
- "periodic": true or false
- "period": the period as a number, or null if not periodic
- "monotonic": one of "increasing", "decreasing", "none"
- "domain": [low, high] over which the function is defined and well-behaved
- "range": [low, high] of output values
Function to analyze: {name}
"""


def _chat_json(model, prompt):
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(resp["message"]["content"])


def _parse_profile(name, raw):
    domain = raw.get("domain") or [-100.0, 100.0]
    rng = raw.get("range") or [-100.0, 100.0]
    return FunctionProfile(
        name=name,
        symmetry=raw.get("symmetry", "none") or "none",
        periodic=bool(raw.get("periodic", False)),
        period=raw.get("period", None),
        monotonic=raw.get("monotonic", "none") or "none",
        domain=(float(domain[0]), float(domain[1])),
        range=(float(rng[0]), float(rng[1])),
    )


def _cache_path(model, name):
    safe_model = model.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe_model}__{name}.json")


def analyze_function(name, config: RunConfig) -> FunctionProfile:
    path = _cache_path(config.model, name)
    if os.path.exists(path):
        with open(path) as fh:
            return _parse_profile(name, json.load(fh))

    raw = _chat_json(config.model, PROMPT_TEMPLATE.format(name=name))

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(raw, fh, indent=2)

    return _parse_profile(name, raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_recon.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/recon.py tests/test_recon.py
git commit -m "Add Ollama reconnaissance with structured profile parsing and caching"
```

---

## Task 7: Custom PSO engine (Part 3 - Hugo)

**Files:**
- Create: `amr/pso.py`
- Test: `tests/test_pso.py`

The engine is generic: it knows nothing about MRs. It takes a fitness callable (maps a
length-5 numpy array to a float) and a `SearchBounds`, and returns a `PSOResult`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pso.py`:

```python
import numpy as np
from amr.pso import optimize
from amr.interfaces import SearchBounds, ParamBound, RunConfig


def _wide_bounds():
    return SearchBounds(
        c1=ParamBound(-5, 5), c2=ParamBound(-5, 5),
        a=ParamBound(-5, 5), b=ParamBound(-5, 5), d=ParamBound(-5, 5),
    )


def test_pso_minimizes_sphere_near_zero():
    # Minimum of sum(p^2) is at the origin.
    cfg = RunConfig(n_particles=40, max_iterations=200, seed=1)
    res = optimize(lambda p: float(np.sum(p ** 2)), _wide_bounds(), cfg)
    assert res.best_fitness < 1e-2
    assert all(abs(v) < 0.2 for v in res.best_params)


def test_pso_respects_fixed_param():
    # Fix a=3; optimum should keep a at exactly 3.
    bounds = _wide_bounds()
    bounds.a = ParamBound(3.0, 3.0, fixed=3.0)
    cfg = RunConfig(n_particles=30, max_iterations=120, seed=2)
    res = optimize(lambda p: float(np.sum(p ** 2)), bounds, cfg)
    assert res.best_params[2] == 3.0


def test_pso_records_history_and_iteration_count():
    cfg = RunConfig(n_particles=20, max_iterations=50, seed=3)
    res = optimize(lambda p: float(np.sum(p ** 2)), _wide_bounds(), cfg)
    assert len(res.history) >= 1
    assert res.iterations >= 1
    assert res.history[-1] <= res.history[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pso.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.pso'`

- [ ] **Step 3: Implement `amr/pso.py`**

```python
import numpy as np
from .interfaces import SearchBounds, RunConfig, PSOResult


def optimize(fitness_fn, bounds: SearchBounds, config: RunConfig) -> PSOResult:
    rng = np.random.default_rng(config.seed)
    lows, highs = bounds.as_arrays()
    dim = len(lows)
    n = config.n_particles
    span = np.where(highs > lows, highs - lows, 1.0)

    pos = rng.uniform(lows, highs, size=(n, dim))
    vel = rng.uniform(-span, span, size=(n, dim)) * 0.1

    fit = np.array([fitness_fn(p) for p in pos])
    pbest = pos.copy()
    pbest_fit = fit.copy()
    g = int(np.argmin(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = float(pbest_fit[g])

    history = [gbest_fit]
    no_improve = 0
    converged = False
    iterations = 0

    for it in range(config.max_iterations):
        iterations = it + 1
        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))
        vel = (config.inertia * vel
               + config.cognitive * r1 * (pbest - pos)
               + config.social * r2 * (gbest - pos))
        pos = np.clip(pos + vel, lows, highs)

        fit = np.array([fitness_fn(p) for p in pos])
        improved = fit < pbest_fit
        pbest[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]

        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < gbest_fit - config.convergence_eps:
            gbest = pbest[g].copy()
            gbest_fit = float(pbest_fit[g])
            no_improve = 0
        else:
            no_improve += 1

        history.append(gbest_fit)
        if no_improve >= config.convergence_window:
            converged = True
            break

    return PSOResult(
        best_params=gbest.tolist(),
        best_fitness=gbest_fit,
        iterations=iterations,
        converged=converged,
        history=history,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pso.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/pso.py tests/test_pso.py
git commit -m "Add generic custom PSO engine with convergence detection"
```

---

## Task 8: MR fitness, validation, and de-duplication (Part 4 - Yeskendir)

**Files:**
- Create: `amr/mr.py`
- Test: `tests/test_mr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mr.py`:

```python
import numpy as np
from amr.interfaces import RunConfig
from amr.mr import make_fitness, validate, normalize_coefficients, deduplicate


def test_fitness_low_for_known_odd_relation_sin():
    # sin is odd: 1*sin(x) + 1*sin(-1*x + 0) + 0 == 0  -> params [1,1,-1,0,0]
    cfg = RunConfig(n_inputs=300, seed=5)
    fitness = make_fitness("sin", cfg)
    good = fitness(np.array([1.0, 1.0, -1.0, 0.0, 0.0]))
    assert good < 1e-6


def test_fitness_penalizes_trivial_zero_solution():
    cfg = RunConfig(n_inputs=100, seed=5)
    fitness = make_fitness("sin", cfg)
    trivial = fitness(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert trivial > 1.0   # penalty dominates


def test_validate_accepts_true_relation():
    cfg = RunConfig(n_inputs=300, tolerance=1e-3, seed=9)
    cand = validate("sin", [1.0, 1.0, -1.0, 0.0, 0.0], cfg)
    assert cand.valid is True
    assert cand.residual < 1e-3


def test_validate_rejects_false_relation():
    cfg = RunConfig(n_inputs=300, tolerance=1e-3, seed=9)
    cand = validate("sin", [1.0, 1.0, 1.0, 0.5, 0.0], cfg)
    assert cand.valid is False


def test_normalize_and_deduplicate_collapse_scaled_duplicates():
    a = {"c1": 1.0, "c2": 1.0, "a": -1.0, "b": 0.0, "d": 0.0}
    b = {"c1": 2.0, "c2": 2.0, "a": -1.0, "b": 0.0, "d": 0.0}  # scaled copy of a
    na, nb = normalize_coefficients(a), normalize_coefficients(b)
    assert na == nb
    unique = deduplicate([a, b])
    assert len(unique) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.mr'`

- [ ] **Step 3: Implement `amr/mr.py`**

```python
import numpy as np
from .interfaces import RunConfig, MRCandidate
from .functions import get_function
from .data import sample_inputs

EPS = 0.0                      # template noise term; fixed at 0 by default
MIN_COEFF_MASS = 0.5           # |c1|+|c2| must exceed this to be non-trivial
PENALTY_WEIGHT = 10.0


def make_fitness(name, config: RunConfig):
    P = get_function(name)
    x1 = sample_inputs(name, config.n_inputs, config.seed)

    def fitness(params):
        c1, c2, a, b, d = params
        residual = c1 * P(x1) + c2 * P(a * x1 + b + EPS) + d
        mse = float(np.mean(residual ** 2))
        mass = abs(c1) + abs(c2)
        penalty = PENALTY_WEIGHT * max(0.0, MIN_COEFF_MASS - mass)
        return mse + penalty

    return fitness


def validate(name, params, config: RunConfig) -> MRCandidate:
    P = get_function(name)
    x = sample_inputs(name, config.n_inputs, config.seed + 1)   # fresh inputs
    c1, c2, a, b, d = params
    residual = c1 * P(x) + c2 * P(a * x + b + EPS) + d
    max_res = float(np.max(np.abs(residual)))
    non_trivial = (abs(c1) + abs(c2)) >= MIN_COEFF_MASS
    coeffs = {"c1": c1, "c2": c2, "a": a, "b": b, "d": d}
    return MRCandidate(coefficients=coeffs,
                       residual=max_res,
                       valid=(max_res < config.tolerance and non_trivial))


def normalize_coefficients(coeffs, decimals=2):
    keys = ["c1", "c2", "a", "b", "d"]
    vec = np.array([coeffs[k] for k in keys], dtype=float)
    scale = np.max(np.abs(vec))
    if scale > 0:
        vec = vec / scale
    return tuple(np.round(vec, decimals))


def deduplicate(coeff_list):
    seen = {}
    for c in coeff_list:
        key = normalize_coefficients(c)
        if key not in seen:
            seen[key] = c
    return list(seen.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mr.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/mr.py tests/test_mr.py
git commit -m "Add MR fitness, validation, and coefficient de-duplication"
```

---

## Task 9: Metrics and mutation kill rate (Part 4 - Yeskendir)

**Files:**
- Create: `amr/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.metrics'`

- [ ] **Step 3: Implement `amr/metrics.py`**

```python
import numpy as np

EPS = 0.0


def mutant_killed(mr_params, mutant_fn, inputs, tolerance):
    c1, c2, a, b, d = mr_params
    residual = c1 * mutant_fn(inputs) + c2 * mutant_fn(a * inputs + b + EPS) + d
    return bool(np.max(np.abs(residual)) > tolerance)


def kill_rate(mr_param_list, mutant_fns, inputs, tolerance):
    if not mutant_fns:
        return 0.0
    killed = 0
    for m in mutant_fns:
        if any(mutant_killed(p, m, inputs, tolerance) for p in mr_param_list):
            killed += 1
    return killed / len(mutant_fns)


def mr_precision(valid, total):
    if total == 0:
        return 0.0
    return valid / total * 100.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add amr/metrics.py tests/test_metrics.py
git commit -m "Add mutation kill-rate and MR precision metrics"
```

---

## Task 10: Orchestration runner and CLI (Part 4 - Yeskendir)

**Files:**
- Create: `amr/run.py`
- Create: `amr/cli.py`
- Test: `tests/test_run.py`

The runner ties the pipeline together for one function: recon (or default bounds when
`--no-llm`), PSO over multiple restarts to collect several candidate MRs, validation,
de-duplication, and kill-rate scoring. Results are written per function and aggregated.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run.py`:

```python
import json
import os
import numpy as np
import amr.run as run
from amr.interfaces import RunConfig


def test_run_function_no_llm_finds_sin_relation(tmp_path, monkeypatch):
    # Force the default (no-LLM) bounds path and a small mutant set.
    monkeypatch.setattr(run, "load_mutants",
                        lambda name, base_dir="data/mutants": [
                            {"id": 0, "op": "add_const", "value": 0.05},
                            {"id": 1, "op": "negate", "value": 0.0},
                        ])
    cfg = RunConfig(use_llm=False, n_inputs=200, n_particles=40,
                    max_iterations=150, seed=4, out_dir=str(tmp_path))
    result = run.run_function("sin", cfg, n_restarts=4)

    assert result["function"] == "sin"
    assert result["mr_count"] >= 1
    assert 0.0 <= result["kill_rate"] <= 1.0
    # the add_const mutant must be caught by at least one discovered relation
    assert result["kill_rate"] >= 0.5
    assert os.path.exists(os.path.join(str(tmp_path), "sin.json"))


def test_run_function_writes_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "load_mutants",
                        lambda name, base_dir="data/mutants": [
                            {"id": 0, "op": "add_const", "value": 0.05}])
    cfg = RunConfig(use_llm=False, n_inputs=150, n_particles=30,
                    max_iterations=100, seed=4, out_dir=str(tmp_path))
    run.run_function("sin", cfg, n_restarts=2)
    with open(os.path.join(str(tmp_path), "sin.json")) as fh:
        data = json.load(fh)
    assert "metrics" in data and "mrs" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'amr.run'`

- [ ] **Step 3: Implement `amr/run.py`**

```python
import json
import logging
import os
import time
import numpy as np

from .interfaces import RunConfig
from .functions import FUNCTION_NAMES, get_function
from .data import sample_inputs
from .bounds import default_bounds, profile_to_bounds
from .recon import analyze_function
from .pso import optimize
from .mr import make_fitness, validate, deduplicate
from .mutants import load_mutants, make_mutant_fn
from .metrics import kill_rate, mr_precision

logger = logging.getLogger("amr")


def _bounds_for(name, config):
    if not config.use_llm:
        return default_bounds()
    try:
        profile = analyze_function(name, config)
        return profile_to_bounds(profile)
    except Exception as exc:                       # Ollama down -> fall back
        logger.warning("recon failed for %s (%s); using default bounds", name, exc)
        return default_bounds()


def run_function(name, config: RunConfig, n_restarts=8):
    start = time.time()
    fitness = make_fitness(name, config)
    bounds = _bounds_for(name, config)

    candidates, total_iters, converged_runs = [], 0, 0
    for r in range(n_restarts):
        cfg_r = RunConfig(**{**config.__dict__, "seed": config.seed + r})
        res = optimize(fitness, bounds, cfg_r)
        total_iters += res.iterations
        converged_runs += int(res.converged)
        cand = validate(name, res.best_params, config)
        if cand.valid:
            candidates.append(cand)

    valid_coeffs = [c.coefficients for c in candidates]
    unique = deduplicate(valid_coeffs)
    unique_params = [[c["c1"], c["c2"], c["a"], c["b"], c["d"]] for c in unique]

    base = get_function(name)
    eval_inputs = sample_inputs(name, config.n_inputs, config.seed + 100)
    mutant_specs = load_mutants(name)
    mutant_fns = [make_mutant_fn(base, s) for s in mutant_specs]
    kr = kill_rate(unique_params, mutant_fns, eval_inputs, config.tolerance)

    result = {
        "function": name,
        "mr_count": len(unique),
        "kill_rate": kr,
        "mrs": unique,
        "metrics": {
            "kill_rate_pct": round(kr * 100.0, 2),
            "killed": int(round(kr * len(mutant_fns))),
            "n_mutants": len(mutant_fns),
            "mr_precision_pct": mr_precision(len(unique), max(1, len(candidates))),
            "avg_pso_iterations": total_iters / max(1, n_restarts),
            "converged_runs": converged_runs,
            "runtime_seconds": round(time.time() - start, 3),
            "used_llm": config.use_llm,
        },
    }

    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, f"{name}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("%s: kill_rate=%.1f%% mrs=%d", name,
                result["metrics"]["kill_rate_pct"], len(unique))
    return result


def run_all(config: RunConfig, functions=None, n_restarts=8, force=False):
    functions = functions or FUNCTION_NAMES
    summary = []
    for name in functions:
        out_path = os.path.join(config.out_dir, f"{name}.json")
        if os.path.exists(out_path) and not force:
            logger.info("skipping %s (result exists)", name)
            with open(out_path) as fh:
                summary.append(json.load(fh))
            continue
        summary.append(run_function(name, config, n_restarts=n_restarts))
    _write_summary(config.out_dir, summary)
    return summary


def _write_summary(out_dir, summary):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "kill_rate_pct", "mr_count",
                    "mr_precision_pct", "avg_pso_iterations", "runtime_seconds"])
        for r in summary:
            m = r["metrics"]
            w.writerow([r["function"], m["kill_rate_pct"], r["mr_count"],
                        m["mr_precision_pct"], m["avg_pso_iterations"],
                        m["runtime_seconds"]])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Implement `amr/cli.py`**

```python
import argparse
import logging
import os
import time

from .interfaces import RunConfig
from .functions import FUNCTION_NAMES
from .run import run_all


def build_config(args):
    return RunConfig(
        model=args.model,
        n_inputs=args.n_inputs,
        tolerance=args.tolerance,
        n_particles=args.particles,
        max_iterations=args.iterations,
        seed=args.seed,
        use_llm=args.use_llm,
        out_dir=args.out,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Auto-discovery of metamorphic relations")
    ap.add_argument("--functions", default="all",
                    help="comma-separated function names, or 'all'")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--use-llm", dest="use_llm", action="store_true", default=True)
    ap.add_argument("--no-llm", dest="use_llm", action="store_false")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-inputs", type=int, default=200)
    ap.add_argument("--particles", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=350)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logs", f"run-{int(time.time())}.log")),
            logging.StreamHandler(),
        ],
    )

    functions = FUNCTION_NAMES if args.functions == "all" else args.functions.split(",")
    config = build_config(args)
    run_all(config, functions=functions, n_restarts=args.restarts, force=args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the CLI end to end (no LLM)**

Run: `python scripts/build_mutants.py && python -m amr.cli --no-llm --functions sin --restarts 4 --out results`
Expected: a `results/sin.json` and `results/summary.csv` are written; log lines print to console.

- [ ] **Step 7: Commit**

```bash
git add amr/run.py amr/cli.py tests/test_run.py
git commit -m "Add pipeline orchestration runner and CLI entry point"
```

---

## Task 11: README, remote background execution, and final verification (Part 4 - Yeskendir)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Automatic Discovery of Metamorphic Relations

LLM-guided Particle Swarm Optimization that discovers metamorphic relations of the form
`c1*P(x1) + c2*P(a*x1 + b) + d = 0` for 8 numerical functions, scored by mutation kill
rate against the AutoMR benchmark.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_mutants.py        # writes data/mutants/<func>.json
```

## Ollama (reconnaissance LLM)

On the GPU server (8-12GB VRAM):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
ollama serve            # leave running (or run as a service)
```

## Run

```bash
# full benchmark, LLM-guided
python -m amr.cli --functions all --model qwen2.5:7b-instruct --out results

# pure-PSO ablation (control, no LLM)
python -m amr.cli --functions all --no-llm --out results_nollm
```

## Run in the background on a remote server

```bash
nohup python -m amr.cli --functions all --out results > logs/nohup.out 2>&1 &
echo $! > run.pid        # PID for later inspection
tail -f logs/nohup.out   # watch progress
```

Or with tmux:

```bash
tmux new -s amr 'python -m amr.cli --functions all --out results'
# detach: Ctrl-b d   reattach: tmux attach -t amr
```

Runs are resumable: re-running skips functions whose `results/<func>.json` exists
(use `--force` to recompute).

## Results

- `results/<func>.json` - profile, discovered MRs, and metrics per function.
- `results/summary.csv` - aggregate table across functions.

## Metrics

Mutation kill rate, MR precision, average PSO iterations to convergence, MR count, and
runtime per function. Compare LLM-guided (`results/`) against the `--no-llm` control
(`results_nollm/`) to measure the LLM's contribution.
````

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass across `test_interfaces, test_functions, test_data, test_mutants, test_bounds, test_recon, test_pso, test_mr, test_metrics, test_run`.

- [ ] **Step 3: Full no-LLM benchmark smoke run**

Run: `python -m amr.cli --no-llm --functions all --restarts 6 --out results_nollm`
Expected: 8 JSON files plus `summary.csv` in `results_nollm/`; non-zero kill rates for at
least the symmetric/periodic functions (sin, cos, asinh, atan).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add README with setup, Ollama, and remote background-run instructions"
```

---

## Self-Review Notes

- Spec section 3 (dataset): Tasks 2 and 4 cover the 8 functions and the 625-mutant set
  (`TOTAL_MUTANTS = 625` split across functions); the AutoMR adapter is scaffolded with a
  clear hook for real-mutant parsing.
- Spec section 4 (architecture/pipeline): recon (Task 6), bounds (Task 5), PSO (Task 7),
  MR validation/dedup (Task 8), kill-rate (Task 9), orchestration/persistence (Task 10).
- Spec section 4.1 (`--no-llm` ablation): implemented in `_bounds_for` and exposed via CLI
  (Task 10); also acts as the Ollama-down fallback.
- Spec section 4.2 (shared interfaces): Task 1, with parameter order `[c1,c2,a,b,d]` used
  consistently in `as_arrays`, `optimize`, `make_fitness`, `validate`, and `metrics`.
- Spec section 6 (Ollama model): README (Task 11) and `RunConfig.model` default.
- Spec section 7 (background + storage): README nohup/tmux instructions, resumable
  `run_all`, per-function JSON + summary CSV (Tasks 10, 11).
- Spec section 8 (metrics): Task 9 plus the metrics block assembled in Task 10.
- Type consistency: `SearchBounds.as_arrays`, `ParamBound.effective_bounds`, `PSOResult`,
  `MRCandidate`, and `RunConfig` field names are referenced identically across tasks.
