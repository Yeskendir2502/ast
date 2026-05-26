# Automatic Discovery of Metamorphic Relations

LLM-guided Particle Swarm Optimization that discovers metamorphic relations of the form
`c1*P(x1) + c2*P(a*x1 + b) + d = 0` for 8 numerical functions, scored by mutation kill
rate against the AutoMR benchmark.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/build_mutants.py        # writes data/mutants/<func>.json
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
python3 -m amr.cli --functions all --model qwen2.5:7b-instruct --out results

# pure-PSO ablation (control, no LLM)
python3 -m amr.cli --functions all --no-llm --out results_nollm
```

## Run in background on a remote server

```bash
nohup python3 -m amr.cli --functions all --out results > logs/nohup.out 2>&1 &
echo $! > run.pid
tail -f logs/nohup.out
```

Or with tmux:

```bash
tmux new -s amr 'python3 -m amr.cli --functions all --out results'
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

## Code structure

| Module | Owner | Responsibility |
|--------|-------|----------------|
| `amr/interfaces.py` | joint | shared dataclasses |
| `amr/functions.py` | Kuba | 8 numpy functions and domains |
| `amr/data.py` | Kuba | domain-aware input sampling |
| `amr/mutants.py` | Kuba | mutant operators and loader |
| `amr/bounds.py` | Ada | profile-to-bounds mapper |
| `amr/recon.py` | Ada | Ollama reconnaissance |
| `amr/pso.py` | Hugo | custom PSO engine |
| `amr/mr.py` | Yeskendir | MR fitness and validation |
| `amr/metrics.py` | Yeskendir | mutation kill rate and precision |
| `amr/run.py` | Yeskendir | pipeline orchestration |
| `amr/cli.py` | Yeskendir | CLI entry point |
