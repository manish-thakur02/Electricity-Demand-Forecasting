import argparse
from .experiment import MODELS, compare_runs, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Corrected rolling one-hour electricity forecasting")
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--data", help="Path to the electricity demand CSV")
    parser.add_argument("--output", default="results")
    parser.add_argument("--demo", action="store_true", help="Use explicitly synthetic test data")
    parser.add_argument("--smoke", action="store_true", help="Small networks, 3 training batches, not research results")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--encoder-length", type=int)
    parser.add_argument("--max-rows", type=int, help="Explicitly restrict input for software testing")
    args = parser.parse_args()
    if not args.demo and not args.data:
        parser.error("--data is required unless --demo is selected")
    selected = MODELS if args.model == "all" else [args.model]
    runs = []
    for name in selected:
        print(f"Running {name}...", flush=True)
        result = run_experiment(name, args.data, args.output, demo=args.demo,
            smoke=args.smoke, epochs=args.epochs, encoder_length=args.encoder_length,
            max_rows=args.max_rows)
        print(result["metrics"].to_string(index=False), flush=True)
        print(f"Saved: {result['run_dir']}", flush=True)
        runs.append(result)
    if len(runs) > 1:
        print(compare_runs(runs, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
