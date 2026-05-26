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
