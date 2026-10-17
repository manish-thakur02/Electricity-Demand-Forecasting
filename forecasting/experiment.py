"""Run experiments and save auditable predictions, metrics and configuration."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
import pandas as pd
from .data import load_data, prepare_data, split_boundaries, synthetic_data
from .evaluation import align_predictions, metrics

MODELS = ("random_forest", "arimax", "lstm", "lstm_rf", "tft", "tft_xgboost")


def run_experiment(model, data_path=None, output_dir="results", *, demo=False,
                   smoke=False, epochs=None, encoder_length=None, seed=42, max_rows=None):
    if model not in MODELS:
        raise ValueError(f"Choose one of {MODELS}")
    if not demo and data_path is None:
        raise ValueError("Supply data_path or explicitly choose demo=True.")
    raw = synthetic_data(seed=seed) if demo else load_data(data_path)
    if max_rows is not None:
        if max_rows < 400:
            raise ValueError("max_rows must be at least 400.")
        raw = raw.iloc[:max_rows].copy()
    # Hash the actual input content used in this run, including row order.
    fingerprint = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    df = prepare_data(raw)
    encoder_length = encoder_length or (24 if smoke else 168)
    epochs = epochs if epochs is not None else (1 if smoke else (100 if model == "tft_xgboost" else 50))
    if encoder_length < 1 or epochs < 1:
        raise ValueError("encoder_length and epochs must be positive.")
    cuts = split_boundaries(df, encoder_length)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir) / model / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()
    from . import models
    base = None
    if model == "random_forest":
        predictions, fitted = models.random_forest(df, cuts, seed, smoke)
    elif model == "arimax":
        predictions, fitted = models.arimax(df, cuts, smoke)
    else:
        family = "lstm" if model.startswith("lstm") else "tft"
        calibration, base, fitted = models.train_neural(df, cuts, family, run_dir,
            encoder_length, epochs, seed, smoke, upgraded=model == "tft_xgboost")
        predictions = base
        if model in ("lstm_rf", "tft_xgboost"):
            predictions, corrector = models.correct_residuals(df, calibration, base,
                "rf" if model == "lstm_rf" else "xgb", seed, smoke)
            # Calibration forecasts help audit the residual training alignment.
            calibration.to_csv(run_dir / "calibration_predictions.csv", index=False)
    _, _, test_start, end = cuts
    seasonal = align_predictions(df, np.arange(test_start, end),
        df.iloc[test_start:end].lag_168, test_start, end)
    rows = [{"model": model, **metrics(predictions)},
            {"model": "seasonal_naive_168h", **metrics(seasonal)}]
    if base is not None and model in ("lstm_rf", "tft_xgboost"):
        rows.append({"model": model + "_base", **metrics(base)})
    scores = pd.DataFrame(rows)
    scores.to_csv(run_dir / "metrics.csv", index=False)
    # Do not present an imputed target as a measured value in exported results.
    predictions = predictions.copy()
    predictions.loc[~predictions.observed_target, "actual"] = np.nan
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    libraries = ["numpy", "pandas", "scikit-learn", "statsmodels", "torch",
                 "lightning", "pytorch-forecasting", "xgboost"]
    versions = {}
    for library in libraries:
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            versions[library] = None
    manifest = {"model": model, "data_kind": "SYNTHETIC_DEMO" if demo else "provided_csv",
        "input_sha256": fingerprint, "input_rows": len(raw), "prepared_rows": len(df),
        "test_start": str(df.iloc[test_start].datetime), "test_end": str(df.iloc[-1].datetime),
        "forecast_horizon_hours": 1, "evaluation": "rolling_one_step_observed_history",
        "split_boundaries": cuts, "encoder_length": encoder_length, "seed": seed,
        "max_epochs": epochs if model not in ("random_forest", "arimax") else None,
        "smoke_test": smoke, "max_rows": max_rows,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "python": platform.python_version(), "libraries": versions,
        "input_missing_counts": {k: int(v) for k, v in raw.isna().sum().items()},
        "inferred_hour_labels": int(raw["hour"].isna().sum()) if "hour" in raw else 0,
        "missing_target_policy": "causal forward-fill for input history; excluded from fit labels and scoring",
        "weather_policy": "trees/ARIMAX use prior-hour weather; TFT uses encoder weather only; LSTM calendar and target history"}
    if model == "arimax":
        manifest["optimizer_converged"] = bool(fitted.mle_retvals.get("converged", False))
    if hasattr(fitted, "run_training_metadata"):
        manifest["training"] = fitted.run_training_metadata
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_predictions(predictions, model, run_dir / "forecast.png", demo)
    return {"model": model, "metrics": scores, "predictions": predictions,
            "manifest": manifest, "run_dir": run_dir}


def plot_predictions(frame, model, path, demo=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    view = frame.iloc[:168]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(view.datetime, view.actual, label="Observed demand", color="black", linewidth=1.5)
    ax.plot(view.datetime, view.prediction, label=model, linewidth=1.2)
    if "base_prediction" in view:
        ax.plot(view.datetime, view.base_prediction, label="Base network", linestyle="--", alpha=.7)
    ax.set(title=("SYNTHETIC TEST - " if demo else "") + "Rolling one-hour-ahead predictions (first 168 test hours)",
           xlabel="Target timestamp", ylabel="Demand (source units)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_runs(results, output_dir):
    """Reject misleading comparisons across datasets, timestamps or protocols."""
    reference = results[0]
    keys = ["input_sha256", "forecast_horizon_hours", "evaluation", "split_boundaries",
            "smoke_test", "data_kind", "encoder_length"]
    for result in results[1:]:
        if any(result["manifest"][k] != reference["manifest"][k] for k in keys):
            raise ValueError("Runs differ in input data, split or evaluation protocol.")
        if not result["predictions"].datetime.equals(reference["predictions"].datetime):
            raise ValueError("Test timestamps differ.")
    rows = [r["metrics"].iloc[0] for r in results]
    rows.append(reference["metrics"].iloc[1])
    comparison = pd.DataFrame(rows).reset_index(drop=True)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(folder / "comparison.csv", index=False)
    return comparison
