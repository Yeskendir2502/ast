# Automatic Discovery of Metamorphic Relations - Design Specification

Date: 2026-05-26
Team: Yeskendir Assankul, Ada Jacyna, Hugo Tarczynski, Kuba Skibicki

## 1. Goal

Build a tool that automatically discovers metamorphic relations (MRs) for numerical
black-box functions using LLM-guided Particle Swarm Optimization (PSO). The tool
searches for linear relations between two executions of a function `P`, expressed by
the template equation:

```
c1 * P(x1) + c2 * P(a * x1 + b + eps) + d = 0
```

The tool finds coefficients `(c1, c2, a, b, d)` such that the equation holds (within a
tolerance) across many sampled inputs. Each discovered relation becomes a metamorphic
relation usable as a test oracle. The function need not be linear itself; we only look
for linear relations between its outputs.

An LLM performs a reconnaissance stage: it analyzes the mathematical properties of `P`
(even/odd symmetry, periodicity, monotonicity, domain and range) and that analysis is
mapped into a smaller, better-shaped PSO search space. The hypothesis under test is
that this guidance reduces PSO iterations to convergence and runtime while preserving
mutation kill rate and MR precision.

## 2. Scope

In scope:

- 8 numerical target functions from the Apache Commons Math 2.2 benchmark:
  `abs, asinh, atan, cos, log1p, log10, sin, tan`.
- LLM reconnaissance via a locally hosted Ollama model.
- Custom PSO over the 5 template parameters.
- MR validation and de-duplication.
- Mutation kill-rate evaluation against the 625 AutoMR mutants.
- Metrics, result persistence, and a background-capable runner for a remote server.
- A `--no-llm` ablation path (pure PSO) used as the control.

Out of scope:

- Executing the original Java sources or mutants on a JVM. The 8 functions are
  reproduced with numpy; mutant input/output data is consumed from the AutoMR
  repository. (If a mutant cannot be reproduced numerically, it is skipped and logged,
  not executed via Java.)
- Non-linear relation templates beyond the one above.
- A graphical interface.

## 3. Dataset

- Apache Commons Math 2.2, the benchmark used by MRI and AutoMR.
- 8 functions listed above.
- 625 pre-generated mutants (the standard benchmark).
- Mutants and experimental data: `github.com/bolzzzz/AutoMR`.
- The 8 functions all have numpy equivalents (`numpy.abs`, `numpy.arcsinh`,
  `numpy.arctan`, `numpy.cos`, `numpy.log1p`, `numpy.log10`, `numpy.sin`, `numpy.tan`),
  so function outputs are computed directly in Python.

## 4. Architecture

The system is a per-function pipeline. For each target function `P`:

1. Reconnaissance (LLM): Ollama analyzes `P` and returns a structured `FunctionProfile`
   (JSON) describing symmetry, periodicity, monotonicity, domain, and range.
2. Bounds mapping: a deterministic mapper converts the profile into a `SearchBounds`
   object - per-parameter lower/upper bounds and sign priors for `(c1, c2, a, b, d)`.
3. PSO: a custom particle swarm searches the 5-D parameter space, minimizing the
   aggregate violation of the template equation over sampled inputs, penalizing the
   trivial all-zero coefficient solution.
4. MR validation: a candidate must satisfy the equation across a fresh set of inputs
   within tolerance; it is normalized and de-duplicated against previously found MRs.
5. Kill-rate evaluation: each valid MR is applied as an oracle to the 625 mutants; a
   mutant is killed when it violates the relation beyond tolerance.
6. Metrics and persistence: per-function metrics are computed and written to disk.

### 4.1 LLM-to-PSO design decision

The LLM returns a structured profile only; numeric bounds are derived by our own
deterministic mapper. This keeps the LLM output small and testable, and keeps numeric
reliability in code rather than in the model's numeric guesses. A `--no-llm` toggle
replaces the recon+mapping stages with wide default bounds, providing the pure-PSO
control for the comparison table and a fallback when Ollama is unavailable.

### 4.2 Shared interfaces

A single `interfaces.py` is written first and owned jointly. All four parts code
against it so they can be developed in parallel. It contains dataclasses:

- `FunctionProfile` - symmetry (`even`/`odd`/`none`), `periodic: bool`, `period: float|None`,
  `monotonic` (`increasing`/`decreasing`/`none`), `domain: (lo, hi)`, `range: (lo, hi)`.
- `SearchBounds` - for each of `c1, c2, a, b, d`: `(lower, upper)`; optional fixed values.
- `MRCandidate` - `coefficients: dict`, `residual: float`, `valid: bool`.
- `PSOResult` - `best_params`, `best_fitness`, `iterations`, `converged`, `history`.
- `RunConfig` - model name, input sample size, tolerances, PSO hyperparameters,
  random seed, output directory, `use_llm: bool`.

### 4.3 Fitness function

Given candidate parameters and `N` sampled inputs `x1` (drawn within the function's
valid domain), the fitness is the mean squared (or mean absolute) residual of the
template equation, plus a penalty term that grows as `(c1, c2)` approach zero, to reject
the trivial solution. `eps` is treated as a small fixed jitter (default 0). The fitness
callable is constructed in Part 4 from Part 1's functions and passed into Part 3's PSO,
keeping the PSO engine generic.

## 5. Module breakdown (the 4-way split)

Four modules of comparable size. Each owner implements their module plus its test file.
Owners are swappable.

### Part 1 - Subjects and Data (owner: Kuba)

Files: `functions.py`, `data.py`, `mutants.py`, `tests/test_functions.py`,
`tests/test_data.py`, `tests/test_mutants.py`.

- `functions.py`: registry mapping each of the 8 names to its numpy implementation and
  its valid input domain.
- `data.py`: domain-aware input sampling (uniform or log-spaced as appropriate),
  reproducible via seed.
- `mutants.py`: load and parse the 625 mutants / experimental data from the AutoMR
  source; expose a way to evaluate a given input set against each mutant. Mutants that
  cannot be reproduced numerically are skipped and logged.

Interface provided: `get_function(name)`, `get_domain(name)`, `sample_inputs(name, n, seed)`,
`load_mutants(name)`, `mutant_outputs(mutant, inputs)`.

### Part 2 - LLM Reconnaissance (owner: Ada)

Files: `recon.py`, `bounds.py`, `tests/test_recon.py`, `tests/test_bounds.py`.

- `recon.py`: Ollama client wrapper, prompt templates that ask the model to describe a
  named function's properties and return JSON, parsing/validation into a
  `FunctionProfile`, and on-disk caching of responses keyed by `(model, function)`.
- `bounds.py`: deterministic `profile_to_bounds(profile) -> SearchBounds`, plus
  `default_bounds()` used by the `--no-llm` path.

Interface provided: `analyze_function(name, config) -> FunctionProfile`,
`profile_to_bounds(profile) -> SearchBounds`, `default_bounds() -> SearchBounds`.

### Part 3 - PSO Engine (owner: Hugo)

Files: `pso.py`, `tests/test_pso.py`.

- Generic custom PSO: particle initialization within bounds, velocity and position
  update with inertia and cognitive/social terms, clamping to bounds, convergence
  detection (no improvement over a window), and iteration/convergence history capture.
- Accepts a fitness callable and a `SearchBounds`; returns a `PSOResult`.
- No knowledge of MRs or functions - validated against simple analytic test objectives
  (for example, recovering a known minimum) so it can be tested in isolation.

Interface provided: `optimize(fitness_fn, bounds, config) -> PSOResult`.

### Part 4 - Orchestration, Metrics, and Runner (owner: Yeskendir)

Files: `mr.py`, `metrics.py`, `run.py`, `cli.py`, `tests/test_mr.py`,
`tests/test_metrics.py`.

- `mr.py`: build the fitness function from a target function and the template equation;
  validate a candidate against fresh inputs; normalize and de-duplicate MRs.
- `metrics.py`: compute mutation kill rate, MR precision, PSO iterations to convergence,
  MR count, and runtime.
- `run.py`: orchestrate the full per-function pipeline across all 8 functions; write
  `results/<func>.json` and aggregate `results/summary.csv`; resumable (skip functions
  already completed); structured logging to `logs/`.
- `cli.py`: argument parsing (`--functions`, `--model`, `--use-llm/--no-llm`, `--out`,
  `--seed`, PSO hyperparameters).

Interface consumed: everything above. This part is the glue and the entry point.

## 6. Ollama model

Target server: GPU with 8-12GB VRAM.

- Primary model: `qwen2.5:7b-instruct` - strong at structured JSON output and reasoning
  about mathematical properties; roughly 6-7GB at Q4 quantization, fits comfortably.
- Alternative: `llama3.1:8b-instruct`.
- Small fallback: `qwen2.5:3b-instruct`.

Setup: `ollama pull qwen2.5:7b-instruct`, `ollama serve` (background), called through
the `ollama` Python package. Responses are cached to disk so reruns do not re-query the
model and so results are reproducible.

## 7. Background execution and result storage

- Entry point: `python -m amr.run --functions all --model qwen2.5:7b-instruct --out results/`.
- Launchable in background on the remote server via `nohup python -m amr.run ... &` or a
  tmux session; the runner logs to `logs/run-<timestamp>.log`.
- Results: one `results/<func>.json` per function (profile, discovered MRs, metrics) and
  an aggregate `results/summary.csv`.
- Resumable: on restart, functions with an existing result file are skipped unless
  `--force` is given.

## 8. Metrics and success criteria

| Metric | Formula | Purpose |
|--------|---------|---------|
| Mutation Kill Rate | killed mutants / 625 * 100% | Primary - did we find real bugs |
| MR Precision | valid MRs / total inferred MRs * 100% | Did PSO return real relations or noise |
| PSO Iterations to Convergence | avg iterations until stable solution | Did the LLM shrink the search space |
| MR Count | unique MRs per function | Diversity / non-redundancy |
| Runtime | seconds per function | Practical efficiency gain from LLM guidance |

Targets (from the comparison table): kill rate > 15% (MRI baseline), MR precision >= 95%,
fewer PSO iterations than the 500x350 baseline, 0 false detections, runtime faster than
the AutoMR 4.7s/MR reference. AutoMR's 50.6% kill rate is the practical upper bound.

## 9. Testing strategy

- Each part ships unit tests for its own module (test-driven where practical).
- Part 1: function outputs match known identities; sampling stays within domains;
  mutant loader parses the expected count.
- Part 2: profile parsing handles valid and malformed LLM output; bounds mapping is
  deterministic for fixed profiles (LLM calls mocked).
- Part 3: PSO recovers known optima on analytic objectives within tolerance.
- Part 4: fitness rejects the trivial solution; known MRs (for example
  `sin(x) + sin(-x) = 0`) are recovered end-to-end on a small run; metrics math is
  correct on hand-computed cases.

## 10. Repository layout

```
amr/
  __init__.py
  interfaces.py        # shared (joint)
  functions.py         # Part 1
  data.py              # Part 1
  mutants.py           # Part 1
  recon.py             # Part 2
  bounds.py            # Part 2
  pso.py               # Part 3
  mr.py                # Part 4
  metrics.py           # Part 4
  run.py               # Part 4
  cli.py               # Part 4
tests/
  test_*.py            # one per module, owned by the module's owner
results/               # generated
logs/                  # generated
docs/superpowers/specs/
requirements.txt
README.md
```

The repository is initialized with git so each owner commits and pushes their own
module, yielding roughly equal contribution.
