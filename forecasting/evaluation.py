"""Strict timestamp alignment and metrics computed from current predictions."""
import numpy as np
import pandas as pd


def align_predictions(df, indices, predictions, start, stop):
    indices = np.asarray(indices, dtype=int).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(indices) != len(predictions) or len(np.unique(indices)) != len(indices):
        raise ValueError("Prediction count mismatch or duplicate target timestamps.")
    expected = np.arange(start, stop)
    if not np.array_equal(np.sort(indices), expected):
        raise ValueError("Predictions must cover every target timestamp in the requested split.")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")
    indexed = pd.Series(predictions, index=indices)
    rows = df.set_index("time_idx").loc[expected]
    return pd.DataFrame({"time_idx": expected, "datetime": rows.datetime.to_numpy(),
        "actual": rows.Demand.to_numpy(), "prediction": indexed.loc[expected].to_numpy(),
        "observed_target": rows.observed_target.to_numpy()})


def metrics(frame):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    observed = frame.loc[frame.observed_target]
    actual, prediction = observed.actual.to_numpy(), observed.prediction.to_numpy()
    if len(actual) < 2 or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("At least two finite observed targets and predictions are required.")
    nonzero = np.abs(actual) > 1e-8
    mse = mean_squared_error(actual, prediction)
    return {"MAE": float(mean_absolute_error(actual, prediction)), "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, prediction)) if np.ptp(actual) > 0 else None,
        "MAPE_percent": float(np.mean(np.abs((actual[nonzero] - prediction[nonzero]) /
                                                  actual[nonzero])) * 100) if nonzero.any() else None,
        "scored_rows": len(actual), "missing_targets_excluded": len(frame) - len(actual),
        "zero_targets_excluded_from_MAPE": int((~nonzero).sum())}


def residual_features(df, predictions):
    from .data import TREE_FEATURES
    # Explicit key join, never tail slicing or truncating arrays to minimum length.
    features = df.set_index("time_idx").loc[predictions.time_idx, TREE_FEATURES].copy()
    features["base_prediction"] = predictions.prediction.to_numpy()
    return features
