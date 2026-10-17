import numpy as np
import pytest
from forecasting.data import prepare_data, synthetic_data, split_boundaries
from forecasting.models import neural_datasets, random_forest, arimax, correct_residuals
from forecasting.evaluation import align_predictions


@pytest.mark.parametrize("family", ["lstm", "tft"])
def test_neural_holdout_boundaries_and_missing_training_targets(family):
    raw = synthetic_data()
    raw.loc[400, "Demand"] = np.nan
    df = prepare_data(raw)
    cuts = split_boundaries(df, 24)
    training, stopping, calibration, test = neural_datasets(df, cuts, 24, family)
    assert 400 - 168 not in training.decoded_index.time_idx_first_prediction.tolist()
    for dataset, start, stop in [(stopping, cuts[0], cuts[1]),
                                (calibration, cuts[1], cuts[2]), (test, cuts[2], cuts[3])]:
        np.testing.assert_array_equal(dataset.decoded_index.time_idx_first_prediction,
                                      np.arange(start, stop))


def test_tree_cannot_fit_test_targets():
    df = prepare_data(synthetic_data())
    cuts = split_boundaries(df, 24)
    first, _ = random_forest(df, cuts, smoke=True)
    mutated = df.copy()
    mutated.loc[cuts[2]:, "Demand"] += 100000
    second, _ = random_forest(mutated, cuts, smoke=True)
    np.testing.assert_allclose(first.prediction, second.prediction, rtol=1e-12, atol=1e-10)


def test_arimax_one_step_does_not_use_current_or_future_observation():
    df = prepare_data(synthetic_data(600))
    cuts = split_boundaries(df, 24)
    first, _ = arimax(df, cuts, smoke=True)
    mutated = df.copy()
    mutated.loc[cuts[2]:, "Demand"] += 10000
    second, _ = arimax(mutated, cuts, smoke=True)
    assert first.prediction.iloc[0] == pytest.approx(second.prediction.iloc[0])
    assert not np.allclose(first.prediction.iloc[1:], second.prediction.iloc[1:])


@pytest.mark.parametrize("kind", ["rf", "xgb"])
def test_residual_corrector_cannot_fit_test_labels(kind):
    df = prepare_data(synthetic_data())
    cuts = split_boundaries(df, 24)
    def forecasts(start, stop):
        return align_predictions(df, np.arange(start, stop), df.iloc[start:stop].lag_24, start, stop)
    calibration = forecasts(cuts[1], cuts[2])
    test = forecasts(cuts[2], cuts[3])
    first, _ = correct_residuals(df, calibration, test, kind, smoke=True)
    test["actual"] += 100000
    second, _ = correct_residuals(df, calibration, test, kind, smoke=True)
    np.testing.assert_allclose(first.prediction, second.prediction, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("family", ["lstm", "tft"])
def test_neural_first_prediction_ignores_unobserved_decoder_values(family):
    import torch
    from forecasting.neural import FixedWindowTemporalFusionTransformer as TemporalFusionTransformer
    from forecasting.neural import FixedWindowRecurrentNetwork as RecurrentNetwork
    from pytorch_forecasting.metrics import RMSE
    torch.set_num_threads(2)
    df = prepare_data(synthetic_data(600))
    cuts = split_boundaries(df, 24)
    training, _, _, test = neural_datasets(df, cuts, 24, family)
    if family == "lstm":
        model = RecurrentNetwork.from_dataset(training, hidden_size=4, rnn_layers=1,
                                              dropout=0, loss=RMSE())
    else:
        model = TemporalFusionTransformer.from_dataset(training, hidden_size=4,
            hidden_continuous_size=4, attention_head_size=1, dropout=0, output_size=1, loss=RMSE())
    changed = df.copy()
    changed.loc[cuts[2]:, ["Demand", "Temperature", "Humidity"]] += 10000
    _, _, _, changed_test = neural_datasets(changed, cuts, 24, family)
    def first(dataset):
        model.eval()
        x, _ = next(iter(dataset.to_dataloader(train=False, batch_size=1, num_workers=0)))
        with torch.no_grad():
            return model.to_prediction(model(x)).detach().cpu().numpy()
    np.testing.assert_allclose(first(test), first(changed_test), rtol=1e-6)
