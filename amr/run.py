import json
import logging
import os
import time

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
    except Exception as exc:
        # if llm fails we fallback to default bounds and keep going
        logger.warning("recon failed for %s (%s); using default bounds", name, exc)
        return default_bounds()


def run_function(name, config, n_restarts=8):
    start = time.time()
    fitness = make_fitness(name, config)
    llm_bounds = _bounds_for(name, config)   # llm-guided, may pin a/b to one relation
    free_bounds = default_bounds()           # always wide, for diverse coverage

    candidates, total_iters, converged_runs = [], 0, 0
    for r in range(n_restarts):
        # different seed per restart so we explore the space more broadly
        cfg_r = RunConfig(**{**vars(config), "seed": config.seed + r})
        # alternate bounds: llm restarts find the canonical relation fast,
        # free restarts find diverse relations that cover more mutants
        bounds = llm_bounds if r % 2 == 0 else free_bounds
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


def run_all(config, functions=None, n_restarts=8, force=False):
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
