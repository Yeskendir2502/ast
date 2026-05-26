import argparse
import logging
import os
import time

from .interfaces import RunConfig
from .functions import FUNCTION_NAMES
from .run import run_all


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
    config = RunConfig(
        model=args.model,
        n_inputs=args.n_inputs,
        tolerance=args.tolerance,
        n_particles=args.particles,
        max_iterations=args.iterations,
        seed=args.seed,
        use_llm=args.use_llm,
        out_dir=args.out,
    )
    run_all(config, functions=functions, n_restarts=args.restarts, force=args.force)


if __name__ == "__main__":
    main()
