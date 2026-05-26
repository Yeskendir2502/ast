import json
import os
import numpy as np

OPERATORS = ["add_const", "mul_const", "add_input", "mul_input", "replace_const", "negate"]


def make_mutant_fn(base, spec):
    op = spec["op"]
    v = spec.get("value", 0.0)
    if op == "add_const":
        return lambda x, v=v: base(x) + v    # v=v captures value now, not at call time
    if op == "mul_const":
        return lambda x, v=v: base(x) * v
    if op == "add_input":
        return lambda x, v=v: base(x + v)
    if op == "mul_input":
        return lambda x, v=v: base(x * v)
    if op == "replace_const":
        return lambda x, v=v: np.full_like(np.asarray(x, dtype=float), v)  # constant output
    if op == "negate":
        return lambda x: -base(x)
    raise ValueError(f"unknown operator: {op}")


def load_mutants(name, base_dir="data/mutants"):
    path = os.path.join(base_dir, f"{name}.json")
    with open(path) as fh:
        return json.load(fh)


def generate_mutant_specs(count, seed=0):
    # deterministic so we get the same mutants on every run
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(count):
        op = OPERATORS[rng.integers(0, len(OPERATORS))]
        if op in ("mul_const", "mul_input"):
            value = float(rng.uniform(0.5, 1.5))  # mild scale, not too wild
        else:
            value = float(rng.uniform(-1.0, 1.0))
        specs.append({"id": i, "op": op, "value": value})
    return specs
