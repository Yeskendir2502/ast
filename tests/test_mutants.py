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
