"""Hourly input validation and features available before each target hour."""
from pathlib import Path
import numpy as np
import pandas as pd

CALENDAR = ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2",
            "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
TREE_FEATURES = CALENDAR + ["hour", "dayofweek", "month", "dayofyear",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    "rolling_24h_mean", "rolling_24h_std", "rolling_168h_mean", "rolling_168h_std",
    "ewma_12h", "ewma_24h", "ewma_168h", "demand_diff_1", "demand_diff_24",
    "temperature_lag1", "humidity_lag1", "temp_humidity_lag1"]


def synthetic_data(rows=960, seed=42):
    """Artificial test fixture; never a replacement for the research dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="h")
    t = np.arange(rows)
    temperature = 25 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, .5, rows)
    return pd.DataFrame({"Timestamp": dates, "Temperature": temperature,
        "Humidity": 60 + rng.normal(0, 3, rows),
        "Demand": 4000 + 400 * np.sin(2 * np.pi * t / 24)
        + 100 * np.cos(2 * np.pi * t / 168) + rng.normal(0, 30, rows)})


def load_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Set DATA_PATH to your CSV. "
                                "Use --demo only for synthetic software testing.")
    return pd.read_csv(path)


def prepare_data(raw):
    if raw.empty:
        raise ValueError("The input dataset is empty.")
    required = {"Timestamp", "Demand", "Temperature", "Humidity"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    raw = raw.copy()
    stamp = pd.to_datetime(raw["Timestamp"], format="mixed", errors="raise")
    if stamp.isna().any():
        raise ValueError("Timestamp must not be missing.")
    if stamp.dt.tz is not None:
        raise ValueError("Use timezone-naive hourly timestamps in one documented timezone.")
    if "hour" in raw:
        hour = pd.to_numeric(raw["hour"], errors="raise")
        if hour.isna().any() and (stamp == stamp.dt.normalize()).all():
            # The supplied CSV has complete, ordered 24-row days with a few
            # missing hour labels. Infer only if every known label validates
            # that structure. Ambiguous/incomplete days fail explicitly.
            within_day = raw.groupby(stamp, sort=False).cumcount()
            day_sizes = raw.groupby(stamp, sort=False)["Timestamp"].transform("size")
            valid_order = (hour.dropna() == within_day.loc[hour.notna()]).all()
            if (day_sizes == 24).all() and valid_order:
                hour = hour.fillna(within_day)
        if hour.isna().any() or not (hour.between(0, 23) & (hour % 1 == 0)).all():
            raise ValueError("hour must contain integers from 0 to 23.")
        if (stamp == stamp.dt.normalize()).all():
            stamp = stamp + pd.to_timedelta(hour, unit="h")
        elif not (stamp.dt.hour == hour).all():
            raise ValueError("Timestamp time and separate hour column disagree.")
    if not (stamp == stamp.dt.floor("h")).all():
        raise ValueError("Timestamps must fall on exact hourly boundaries.")
    if stamp.duplicated().any():
        raise ValueError("Duplicate hourly timestamps; fix duplicates before training.")
    df = raw.assign(datetime=stamp).set_index("datetime").sort_index()
    df = df[["Demand", "Temperature", "Humidity"]].apply(pd.to_numeric, errors="raise")
    if np.isinf(df.to_numpy()).any():
        raise ValueError("Input contains infinite values.")
    # Preserve gaps on the hourly clock. Never compress missing hours with dropna().
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="datetime"))
    df["observed_target"] = df["Demand"].notna()
    df[["Demand", "Temperature", "Humidity"]] = df[["Demand", "Temperature", "Humidity"]].ffill()
    # Forward filling is causal; no backward filling or interpolation from future rows.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    for name, values, period in [("hour", df.hour, 24), ("dow", df.dayofweek, 7),
                                  ("month", df.month, 12)]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    df["hour_sin2"] = np.sin(4 * np.pi * df.hour / 24)
    df["hour_cos2"] = np.cos(4 * np.pi * df.hour / 24)
    df["is_weekend"] = (df.dayofweek >= 5).astype(float)
    past = df["Demand"].shift(1)
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f"lag_{lag}"] = df["Demand"].shift(lag)
    for window in [24, 168]:
        df[f"rolling_{window}h_mean"] = past.rolling(window).mean()
        df[f"rolling_{window}h_std"] = past.rolling(window).std()
    for span in [12, 24, 168]:
        df[f"ewma_{span}h"] = past.ewm(span=span, adjust=False).mean()
    df["demand_diff_1"] = past.diff(1)
    df["demand_diff_24"] = past.diff(24)
    df["temperature_lag1"] = df.Temperature.shift(1)
    df["humidity_lag1"] = df.Humidity.shift(1)
    df["temp_humidity_lag1"] = df.temperature_lag1 * df.humidity_lag1
    ready = df[TREE_FEATURES + ["Demand"]].notna().all(axis=1)
    if not ready.any():
        raise ValueError("Need more than 168 hours with usable historical demand and weather.")
    df = df.loc[ready.idxmax():].copy()
    if df[TREE_FEATURES + ["Demand"]].isna().any().any():
        raise ValueError("Unresolved missing values after the initial warm-up.")
    df = df.reset_index()
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    df["series"] = "electricity"
    return df


def split_boundaries(df, encoder_length=168):
    """70% base fit, 7.5% stopping, 7.5% residual fit, 15% final test."""
    n = len(df)
    cuts = (n * 70 // 100, n * 775 // 1000, n * 85 // 100, n)
    if cuts[0] <= encoder_length + 8 or min(np.diff((0,) + cuts)[1:]) < 8:
        raise ValueError("Insufficient history for encoder and chronological holdouts.")
    for lo, hi in zip((0,) + cuts[:-1], cuts):
        if df.iloc[lo:hi].observed_target.sum() < 2:
            raise ValueError("Each split needs at least two observed demand values.")
    return cuts
