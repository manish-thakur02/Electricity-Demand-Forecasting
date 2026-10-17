import numpy as np
import pandas as pd
import pytest
from forecasting.data import prepare_data, synthetic_data, TREE_FEATURES, split_boundaries
from forecasting.evaluation import align_predictions, metrics, residual_features


def test_current_and_future_demand_cannot_change_target_features():
    raw = synthetic_data()
    before = prepare_data(raw)
    raw.loc[500:, "Demand"] += 10000
    after = prepare_data(raw)
    # At target hour 500, only hours <=499 may enter a tree feature.
    pd.testing.assert_frame_equal(before.loc[:332, TREE_FEATURES], after.loc[:332, TREE_FEATURES])
    assert before.loc[333, "lag_1"] != after.loc[333, "lag_1"]


def test_current_weather_cannot_change_tree_features():
    raw = synthetic_data()
    before = prepare_data(raw)
    raw.loc[500:, ["Temperature", "Humidity"]] += 1000
    after = prepare_data(raw)
    pd.testing.assert_frame_equal(before.loc[:332, TREE_FEATURES], after.loc[:332, TREE_FEATURES])


def test_date_plus_hour_and_validated_missing_hour_recovery():
    raw = synthetic_data()
    expected = prepare_data(raw)
    raw["hour"] = raw.Timestamp.dt.hour.astype(float)
    raw["Timestamp"] = raw.Timestamp.dt.strftime("%d-%b-%y")
    raw.loc[500, "hour"] = np.nan
    pd.testing.assert_frame_equal(expected, prepare_data(raw))


def test_ambiguous_missing_hour_is_rejected():
    raw = synthetic_data()
    raw["hour"] = raw.Timestamp.dt.hour.astype(float)
    raw["Timestamp"] = raw.Timestamp.dt.strftime("%d-%b-%y")
    raw.loc[500, "hour"] = np.nan
    with pytest.raises(ValueError, match="hour"):
        prepare_data(raw.drop(index=501))


def test_calendar_columns_are_recomputed():
    raw = synthetic_data()
    expected = prepare_data(raw)
    raw["dayofweek"] = 999
    raw["month"] = np.nan
    pd.testing.assert_frame_equal(expected, prepare_data(raw))


def test_missing_hour_keeps_time_and_excludes_unobserved_label():
    raw = synthetic_data().drop(index=500)
    df = prepare_data(raw)
    assert (df.datetime.diff().dropna() == pd.Timedelta(hours=1)).all()
    assert not df.loc[332, "observed_target"]
    assert df.loc[332, "Demand"] == df.loc[331, "Demand"]


def test_duplicates_rejected():
    raw = synthetic_data()
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_data(pd.concat([raw, raw.iloc[:1]]))


def test_predictions_align_by_key_not_input_order():
    df = prepare_data(synthetic_data())
    pred = align_predictions(df, [702, 700, 701], [2, 0, 1], 700, 703)
    assert pred.prediction.tolist() == [0, 1, 2]
    X = residual_features(df, pred)
    np.testing.assert_array_equal(X.lag_1, df.iloc[700:703].lag_1)
    with pytest.raises(ValueError, match="every target"):
        align_predictions(df, [701, 702], [1, 2], 700, 703)
    with pytest.raises(ValueError, match="duplicate"):
        align_predictions(df, [700, 700], [1, 2], 700, 702)


def test_missing_and_zero_targets_have_explicit_metric_policy():
    frame = pd.DataFrame({"actual": [0., 2., 100.], "prediction": [1., 1., -100.],
                          "observed_target": [True, True, False]})
    scores = metrics(frame)
    assert scores["MAE"] == 1
    assert scores["MAPE_percent"] == 50
    assert scores["missing_targets_excluded"] == 1
    assert scores["zero_targets_excluded_from_MAPE"] == 1


def test_too_short_history_is_rejected():
    with pytest.raises(ValueError, match="168"):
        prepare_data(synthetic_data(100))
    with pytest.raises(ValueError, match="Insufficient"):
        split_boundaries(prepare_data(synthetic_data(220)))
