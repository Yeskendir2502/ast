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

def _chat_json(model: str, prompt: str) -> dict:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(resp["message"]["content"])

def _parse_profile(name: str, raw: dict) -> FunctionProfile:
    domain = raw.get("domain") or [-100.0, 100.0]
    rng = raw.get("range") or [-100.0, 100.0]
    return FunctionProfile(
        name=name,
        symmetry=raw.get("symmetry", "none") or "none",
        periodic=bool(raw.get("periodic", False)),
        period=raw.get("period", None),
        monotonic=raw.get("monotonic", "none") or "none",
        domain=(float(domain[0]), float(domain[1])),
        output_range=(float(rng[0]), float(rng[1])),
    )

def _cache_path(model: str, name: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe_model}__{name}.json")

def analyze_function(name: str, config: RunConfig) -> FunctionProfile:
    path = _cache_path(config.model, name)
    if os.path.exists(path):
        with open(path) as fh:
            return _parse_profile(name, json.load(fh))

    raw = _chat_json(config.model, PROMPT_TEMPLATE.format(name=name))

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(raw, fh, indent=2)

    return _parse_profile(name, raw)
