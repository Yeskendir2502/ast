import json
import os
import numpy as np

OPERATORS = ["add_const", "mul_const", "add_input", "mul_input", "replace_const", "negate"]


def make_mutant_fn(base, spec):
    """Returns a callable(x: np.ndarray) -> np.ndarray that applies the mutant operator."""
    op = spec["op"]
    v = spec.get("value", 0.0)
    if op == "add_const":
        return lambda x, v=v: base(x) + v
    if op == "mul_const":
        return lambda x, v=v: base(x) * v
    if op == "add_input":
        return lambda x, v=v: base(x + v)
    if op == "mul_input":
        return lambda x, v=v: base(x * v)
    if op == "replace_const":
        return lambda x, v=v: np.full_like(np.asarray(x, dtype=float), v)
    if op == "negate":
        return lambda x: -base(x)
    raise ValueError(f"unknown operator: {op}")


def load_mutants(name, base_dir="data/mutants"):
    """Load mutant specs from a JSON file in base_dir/<name>.json."""
    path = os.path.join(base_dir, f"{name}.json")
    with open(path) as fh:
        return json.load(fh)


def generate_mutant_specs(count, seed=0):
    """
    Returns a list of count mutant spec dicts, each with keys: id, op, value.
    - ops are chosen randomly from OPERATORS
    - for mul_const and mul_input: value in [0.5, 1.5] (perturbation range)
    - for others: value in [-1.0, 1.0]
    - deterministic given the same count and seed
    """
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(count):
        op = OPERATORS[rng.integers(0, len(OPERATORS))]
        if op in ("mul_const", "mul_input"):
            value = float(rng.uniform(0.5, 1.5))
        else:
            value = float(rng.uniform(-1.0, 1.0))
        specs.append({"id": i, "op": op, "value": value})
    return specs
