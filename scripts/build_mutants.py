import argparse, json, os, sys

# make sure project root is on the path when running this directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amr.functions import FUNCTION_NAMES
from amr.mutants import generate_mutant_specs

TOTAL_MUTANTS = 625  # matches the automr benchmark count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mutants")
    ap.add_argument("--from-automr", default=None,
                    help="path to a cloned automr checkout, not wired yet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    per_func = TOTAL_MUTANTS // len(FUNCTION_NAMES)  # 78 per function

    if args.from_automr:
        # I havent wired real mutant parsing yet, fallback to generated specs
        print(f"[build_mutants] AutoMR source given: {args.from_automr}", file=sys.stderr)
        print("[build_mutants] Real-mutant parsing not yet wired; falling back to generated specs.", file=sys.stderr)

    for i, name in enumerate(FUNCTION_NAMES):
        specs = generate_mutant_specs(count=per_func, seed=args.seed + i)
        with open(os.path.join(args.out, f"{name}.json"), "w") as fh:
            json.dump(specs, fh, indent=2)
        print(f"[build_mutants] wrote {per_func} mutants for {name}")


if __name__ == "__main__":
    main()
