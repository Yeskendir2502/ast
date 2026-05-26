import json
import os
import numpy as np
import amr.run as run
from amr.interfaces import RunConfig


def test_run_function_no_llm_finds_sin_relation(tmp_path, monkeypatch):
    # small mutant set to keep the test fast
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
    assert result["kill_rate"] >= 0.5  # add_const mutant should be caught
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
