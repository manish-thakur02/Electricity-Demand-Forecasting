"""Original model families with a consistent rolling one-step evaluation."""
from pathlib import Path
import numpy as np
from .data import CALENDAR, TREE_FEATURES
from .evaluation import align_predictions, residual_features


def random_forest(df, cuts, seed=42, smoke=False):
    from sklearn.ensemble import RandomForestRegressor
    train_end, _, test_start, end = cuts
    train = df.iloc[:train_end].loc[lambda d: d.observed_target]
    model = RandomForestRegressor(n_estimators=12 if smoke else 300, max_depth=15,
                                 min_samples_leaf=4, random_state=seed, n_jobs=-1)
    model.fit(train[TREE_FEATURES], train.Demand)
    prediction = model.predict(df.iloc[test_start:end][TREE_FEATURES])
    return align_predictions(df, np.arange(test_start, end), prediction, test_start, end), model


def arimax(df, cuts, smoke=False):
    """ARIMA(2,0,2) + calendar/lagged weather, as in the supplied SARIMA file.

    There is no explicit seasonal ARIMA order in that notebook. Retain the
    actual family and label it accurately rather than silently changing it.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.preprocessing import StandardScaler
    train_end, _, test_start, end = cuts
    features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "temperature_lag1", "humidity_lag1"]
    scaler = StandardScaler().fit(df.iloc[:train_end][features])
    exog = scaler.transform(df[features])
    # A missing target stays missing in the state-space observation equation.
    y = df.Demand.where(df.observed_target).to_numpy()
    mean, scale = np.nanmean(y[:train_end]), np.nanstd(y[:train_end])
    scale = max(float(scale), 1.0)
    y_scaled = (y - mean) / scale
    model = ARIMA(y_scaled[:train_end], exog=exog[:train_end], order=(2, 0, 2))
    # Powell avoids deprecated SciPy L-BFGS arguments and the precision-loss
    # convergence failure seen with BFGS on the supplied full training period.
    fitted = model.fit(method_kwargs={"method": "powell", "maxiter": 30 if smoke else 300,
                                     "xtol": 1e-5, "ftol": 1e-7})
    if not smoke and not fitted.mle_retvals.get("converged", False):
        raise RuntimeError("ARIMAX optimizer did not converge. Inspect training data and fit settings.")
    # One-step predictions from fixed fitted parameters. Kalman filtering uses
    # an observation only AFTER predicting that observation, and does not refit.
    updated = fitted.extend(y_scaled[train_end:], exog=exog[train_end:])
    one_step = np.asarray(updated.get_prediction().predicted_mean) * scale + mean
    prediction = one_step[test_start - train_end:]
    result = align_predictions(df, np.arange(test_start, end), prediction, test_start, end)
    return result, fitted


def neural_datasets(df, cuts, encoder_length, family):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    train_end, stop_end, residual_end, end = cuts
    # RecurrentNetwork requires matching encoder/decoder covariates (except
    # target) and does not support add_relative_time_idx=True.
    known = ["time_idx"] + CALENDAR
    unknown = ["Demand"]
    if family == "tft":
        unknown += ["Temperature", "Humidity"]
    training = TimeSeriesDataSet(df.iloc[:train_end], time_idx="time_idx", target="Demand",
        group_ids=["series"], static_categoricals=["series"],
        min_encoder_length=encoder_length, max_encoder_length=encoder_length,
        min_prediction_length=1, max_prediction_length=1,
        time_varying_known_reals=known, time_varying_unknown_reals=unknown,
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=(family == "tft"), add_target_scales=True,
        add_encoder_length=False, allow_missing_timesteps=False)
    # Holdouts include historical encoder context, but targets start at the
    # boundary. Every hour is predicted, including the first holdout hour.
    def holdout(start, stop):
        return TimeSeriesDataSet.from_dataset(training,
            df.iloc[start - encoder_length:stop], min_prediction_idx=start,
            stop_randomization=True, predict=False)
    stopping = holdout(train_end, stop_end)
    residual = holdout(stop_end, residual_end)
    test = holdout(residual_end, end)
    observed = set(df.loc[df.observed_target, "time_idx"])
    def observed_only(index):
        return index.time_idx_first_prediction.isin(observed)
    # Do not fit/early-stop against imputed target labels.
    training = training.filter(observed_only)
    stopping = stopping.filter(observed_only)
    return training, stopping, residual, test


def train_neural(df, cuts, family, output_dir, encoder_length=168, epochs=50,
                 seed=42, smoke=False, upgraded=False):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from .neural import FixedWindowRecurrentNetwork, FixedWindowTemporalFusionTransformer
    from pytorch_forecasting.metrics import RMSE
    pl.seed_everything(seed, workers=True)
    # Prevent CPU thread oversubscription on small sequence batches.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    datasets = neural_datasets(df, cuts, encoder_length, family)
    train, stopping, residual, test = datasets
    def loader(dataset, train=False):
        return dataset.to_dataloader(train=train, batch_size=32 if smoke else 64,
                                     num_workers=0, drop_last=False)
    train_loader, stop_loader = loader(train, True), loader(stopping)
    if family == "lstm":
        cls = FixedWindowRecurrentNetwork
        model = cls.from_dataset(train, cell_type="LSTM", hidden_size=8 if smoke else 128,
            rnn_layers=2, dropout=.2, loss=RMSE(), learning_rate=.001, optimizer="adam")
    else:
        cls = FixedWindowTemporalFusionTransformer
        model = cls.from_dataset(train, hidden_size=8 if smoke else (64 if upgraded else 32),
            attention_head_size=2 if smoke else 4, hidden_continuous_size=4 if smoke else (32 if upgraded else 16),
            dropout=.15 if upgraded else .2, output_size=1, loss=RMSE(),
            learning_rate=.003 if upgraded else .001, optimizer="adamw" if upgraded else "adam")
    checkpoint = ModelCheckpoint(dirpath=str(Path(output_dir) / "checkpoints"),
        filename=family + "-best-{epoch:02d}", monitor="val_loss", mode="min", save_top_k=1)
    class TrainingProgress(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
                print(f"{family}: epoch {trainer.current_epoch + 1}, batch {batch_idx + 1}/{trainer.num_training_batches}", flush=True)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", devices=1,
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        gradient_clip_val=.1,
        callbacks=[checkpoint, EarlyStopping(monitor="val_loss", patience=10, mode="min"), TrainingProgress()],
        # TFT interpolation backward has no deterministic CUDA implementation.
        deterministic="warn" if family == "tft" and torch.cuda.is_available() else True,
        num_sanity_val_steps=0,
        limit_train_batches=3 if smoke else 1.0)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=stop_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("No best checkpoint was produced; inspect training losses.")
    # Only load the checkpoint just created in this run.
    model = cls.load_from_checkpoint(checkpoint.best_model_path)
    model.run_training_metadata = {
        "epochs_completed": int(trainer.current_epoch),
        "best_validation_loss": float(checkpoint.best_model_score.detach().cpu()),
        "training_windows": len(train), "early_stopping_windows": len(stopping),
        "architecture": family, "upgraded_architecture": bool(upgraded),
    }
    def predict(dataset, start, stop):
        result = model.predict(loader(dataset), mode="prediction", return_index=True,
            trainer_kwargs={"accelerator": "cpu", "devices": 1, "logger": False,
                            "enable_progress_bar": False})
        values = result.output.detach().cpu().numpy()
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(f"Expected one-step output, received {values.shape}")
        return align_predictions(df, result.index.time_idx.to_numpy(), values[:, 0], start, stop)
    _, residual_start, test_start, end = cuts
    return predict(residual, residual_start, test_start), predict(test, test_start, end), model


def correct_residuals(df, calibration, test, kind, seed=42, smoke=False):
    X = residual_features(df, calibration)
    y = calibration.actual.to_numpy() - calibration.prediction.to_numpy()
    observed = calibration.observed_target.to_numpy()
    X, y = X.loc[observed], y[observed]
    if len(y) < 8:
        raise ValueError("Residual training needs at least 8 observed targets.")
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=12 if smoke else 300, max_depth=10,
            min_samples_leaf=4, random_state=seed, n_jobs=-1)
        model.fit(X, y)
    else:
        from xgboost import XGBRegressor
        cut = max(1, int(.8 * len(y)))
        model = XGBRegressor(n_estimators=12 if smoke else 1000, max_depth=5,
            learning_rate=.02, subsample=.8, colsample_bytree=.7, min_child_weight=5,
            reg_alpha=.1, reg_lambda=1, objective="reg:squarederror", tree_method="hist",
            early_stopping_rounds=5 if smoke else 50, eval_metric="rmse", random_state=seed,
            n_jobs=4)
        model.fit(X.iloc[:cut], y[:cut], eval_set=[(X.iloc[cut:], y[cut:])], verbose=False)
    result = test.copy()
    result["base_prediction"] = result.prediction
    result["prediction"] += model.predict(residual_features(df, test))
    return result, model
