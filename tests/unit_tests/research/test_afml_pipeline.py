# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nautilus_trader.research.afml_pipeline import MovingAverageCrossoverSideModel
from nautilus_trader.research.afml_pipeline import SequentialBootstrapBaggingClassifier
from nautilus_trader.research.afml_pipeline import add_vertical_barrier
from nautilus_trader.research.afml_pipeline import average_uniqueness
from nautilus_trader.research.afml_pipeline import average_uniqueness_from_indicator
from nautilus_trader.research.afml_pipeline import bet_size_from_probability
from nautilus_trader.research.afml_pipeline import build_afml_dataset
from nautilus_trader.research.afml_pipeline import build_afml_meta_dataset
from nautilus_trader.research.afml_pipeline import fit_afml_meta_model
from nautilus_trader.research.afml_pipeline import fit_afml_model
from nautilus_trader.research.afml_pipeline import fracdiff_ffd
from nautilus_trader.research.afml_pipeline import get_bins
from nautilus_trader.research.afml_pipeline import get_daily_vol
from nautilus_trader.research.afml_pipeline import get_indicator_matrix
from nautilus_trader.research.afml_pipeline import get_triple_barrier_events
from nautilus_trader.research.afml_pipeline import load_afml_bar_csv
from nautilus_trader.research.afml_pipeline import make_bet_size_frame
from nautilus_trader.research.afml_pipeline import make_default_features
from nautilus_trader.research.afml_pipeline import make_ewma_1bar_log_return_volatility_target
from nautilus_trader.research.afml_pipeline import make_ewma_volatility_target
from nautilus_trader.research.afml_pipeline import make_horizon_log_return_volatility_target
from nautilus_trader.research.afml_pipeline import make_pipeline_features
from nautilus_trader.research.afml_pipeline import make_rolling_cusum_threshold
from nautilus_trader.research.afml_pipeline import mda_feature_importance
from nautilus_trader.research.afml_pipeline import mdi_feature_importance
from nautilus_trader.research.afml_pipeline import moving_average_crossover_side
from nautilus_trader.research.afml_pipeline import num_concurrent_events
from nautilus_trader.research.afml_pipeline import sample_weights_by_return
from nautilus_trader.research.afml_pipeline import sequential_bootstrap
from nautilus_trader.research.afml_pipeline import sequential_bootstrap_indices
from nautilus_trader.research.afml_pipeline import sfi_feature_importance
from nautilus_trader.research.afml_pipeline import symmetric_cusum_filter
from nautilus_trader.research.afml_pipeline import write_signal_csv


def test_get_daily_vol_returns_ewm_daily_returns():
    index = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)

    daily_vol = get_daily_vol(close, span=5)

    assert daily_vol.index[0] == index[25]
    assert daily_vol.dropna().shape[0] > 0
    assert daily_vol.index.is_monotonic_increasing


def test_make_horizon_log_return_volatility_target_uses_rolling_horizon_returns():
    index = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    log_close = pd.Series((np.arange(len(index), dtype=float) ** 2) * 0.001, index=index)
    close = np.exp(log_close)

    target = make_horizon_log_return_volatility_target(
        close,
        horizon_bars=2,
        window_bars=3,
        min_periods=3,
        floor_quantile=None,
    )

    expected = np.log(close).diff(2).rolling(3, min_periods=3).std().rename("trgt")
    pd.testing.assert_series_equal(target, expected)


def test_make_ewma_1bar_log_return_volatility_target_uses_ewma_log_returns():
    index = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    log_close = pd.Series((np.arange(len(index), dtype=float) ** 2) * 0.001, index=index)
    close = np.exp(log_close)

    target = make_ewma_1bar_log_return_volatility_target(
        close,
        span=4,
        min_periods=2,
        floor_quantile=None,
    )

    expected = np.log(close).diff().ewm(span=4, adjust=False, min_periods=2).std().rename("trgt")
    pd.testing.assert_series_equal(target, expected)


def test_symmetric_cusum_filter_resets_after_events():
    index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    series = pd.Series([0.00, 0.01, 0.03, 0.02, -0.02], index=index)

    events = symmetric_cusum_filter(series, threshold=0.015)

    assert list(events) == [index[2], index[4]]


def test_rolling_cusum_threshold_uses_training_window_floor():
    index = pd.date_range("2024-01-01", periods=80, freq="15min", tz="UTC")
    close = pd.Series(
        100.0 * np.exp(np.cumsum(np.sin(np.arange(len(index)) / 5.0) * 0.001)),
        index=index,
    )

    threshold = make_rolling_cusum_threshold(
        close,
        window=12,
        min_periods=4,
        train_start=index[10],
        train_end=index[50],
        floor_quantile=0.25,
    )

    train_values = threshold.loc[index[10] : index[50]].dropna()
    assert threshold.name == "cusum_threshold"
    assert train_values.shape[0] > 0
    assert threshold.dropna().min() >= train_values.quantile(0.25) - 1e-12


def test_build_dataset_accepts_explicit_cusum_and_barrier_target():
    index = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    steps = np.sin(np.arange(len(index)) / 11.0) * 0.001
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)
    cusum_threshold = make_rolling_cusum_threshold(
        close,
        window=20,
        min_periods=5,
        train_start=index[20],
        train_end=index[180],
    )
    target = make_ewma_volatility_target(
        close,
        span=20,
        train_start=index[20],
        train_end=index[180],
    )

    dataset = build_afml_dataset(
        close,
        cusum_threshold=cusum_threshold,
        target=target,
        vertical_barrier_bars=12,
        vertical_barrier_days=None,
    )

    assert list(dataset.t_events) == list(symmetric_cusum_filter(np.log(close), cusum_threshold))
    assert dataset.cusum_threshold is not None
    assert dataset.events["trgt"].equals(target.reindex(dataset.events.index))


def test_triple_barrier_labels_first_profit_touch():
    index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    close = pd.Series([100.0, 103.0, 104.0, 102.0, 101.0], index=index)
    t_events = pd.DatetimeIndex([index[0]])
    target = pd.Series(0.02, index=index)
    vertical = add_vertical_barrier(t_events, close, num_bars=4, num_days=None)

    events = get_triple_barrier_events(close, t_events, (1.0, 1.0), target, t1=vertical)
    labels = get_bins(events, close)

    assert events.iloc[0]["barrier"] == "pt"
    assert events.iloc[0]["t1"] == index[1]
    assert labels.iloc[0]["bin"] == 1


def test_triple_barrier_can_zero_vertical_touch():
    index = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    close = pd.Series([100.0, 101.0, 100.5], index=index)
    t_events = pd.DatetimeIndex([index[0]])
    target = pd.Series(0.05, index=index)
    vertical = add_vertical_barrier(t_events, close, num_bars=2, num_days=None)

    events = get_triple_barrier_events(close, t_events, (1.0, 1.0), target, t1=vertical)
    labels = get_bins(events, close, zero_on_vertical=True)

    assert events.iloc[0]["barrier"] == "t1"
    assert labels.iloc[0]["bin"] == 0


def test_meta_labels_require_return_above_cost_hurdle():
    index = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    close = pd.Series([100.0, 100.5, 101.0], index=index)
    events = pd.DataFrame(
        {
            "t1": [index[1]],
            "trgt": [0.01],
            "side": [1],
            "barrier": ["t1"],
        },
        index=[index[0]],
    )

    labels = get_bins(events, close, meta_label=True, meta_label_min_ret=0.01)

    assert np.isclose(labels.iloc[0]["ret"], 0.005)
    assert labels.iloc[0]["bin"] == 0


def test_sample_weights_reflect_overlap_and_return_attribution():
    index = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0, 104.0], index=index)
    t1 = pd.Series([index[3], index[4]], index=[index[0], index[2]])

    concurrency = num_concurrent_events(index, t1)
    uniqueness = average_uniqueness(t1, concurrency)
    weights = sample_weights_by_return(t1, concurrency, close)

    assert concurrency.loc[index[2]] == 2.0
    assert 0.0 < uniqueness.loc[index[0]] < 1.0
    assert np.isclose(weights.sum(), len(weights))


def test_sample_weights_are_capped_after_extreme_return_shock():
    index = pd.date_range("2024-01-01", periods=101, freq="min", tz="UTC")
    close = pd.Series(100.0, index=index)
    close.iloc[50:] = 1_000.0
    t1 = pd.Series(index[1:], index=index[:-1])

    concurrency = num_concurrent_events(index, t1)
    weights = sample_weights_by_return(t1, concurrency, close)

    assert np.isclose(weights.sum(), len(weights))
    assert weights.max() <= 10.0
    assert weights.max() / weights.sum() < 0.11


def test_sequential_bootstrap_samples_indicator_columns():
    bar_index = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    t1 = pd.Series(
        [bar_index[2], bar_index[3], bar_index[5]], index=[bar_index[0], bar_index[2], bar_index[4]]
    )
    indicator = get_indicator_matrix(bar_index, t1)

    avg_uniqueness = average_uniqueness_from_indicator(indicator)
    sampled_positions = sequential_bootstrap_indices(indicator, sample_length=4, random_state=7)
    sampled_labels = sequential_bootstrap(indicator, sample_length=4, random_state=7)

    assert indicator.shape == (6, 3)
    assert np.isclose(avg_uniqueness.loc[bar_index[0]], 5.0 / 6.0)
    assert sampled_positions.shape == (4,)
    assert sampled_labels.shape == (4,)
    assert set(sampled_labels).issubset(set(indicator.columns))


def test_fracdiff_default_features_replace_log_returns():
    index = pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.05, len(index))), index=index)

    features = make_default_features(close, fast_span=4, slow_span=8)
    fracdiff = fracdiff_ffd(np.log(close), d=0.4, threshold=0.01)

    assert fracdiff.first_valid_index() == index[10]
    assert features["ffd_d_0_40_lag_1"].first_valid_index() == index[10]
    assert any(column.startswith("ffd_d_0_40") for column in features.columns)
    assert not any(column.startswith("log_ret_") for column in features.columns)


def test_load_afml_bar_csv_preserves_generation_metadata(tmp_path):
    path = tmp_path / "bars.csv"
    frame = pd.DataFrame(
        {
            "bar_type": ["AFML_DRB"],
            "ts_event": ["2024-01-01T00:00:00Z"],
            "ts_event_ns": [1_704_067_200_000_000_000],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
            "buy_notional": [700.0],
            "sell_notional": [300.0],
            "signed_notional": [400.0],
            "theta": [700.0],
        },
    )
    frame.to_csv(path, index=False)

    loaded = load_afml_bar_csv(path)

    assert loaded.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert loaded.iloc[0]["signed_notional"] == 400.0
    assert "bar_type" not in loaded.columns


def test_pipeline_features_include_microstructure_information_and_sadf():
    index = pd.date_range("2024-01-01", periods=180, freq="15min", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.sin(np.arange(len(index)) / 12.0) * 0.02), index=index)
    flow = np.sin(np.arange(len(index)) / 5.0)
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 100.0 + np.arange(len(index)) % 11,
            "buy_ticks": 40 + np.arange(len(index)) % 7,
            "sell_ticks": 35 + np.arange(len(index)) % 5,
            "buy_notional": close * (55.0 + 15.0 * flow),
            "sell_notional": close * (45.0 - 15.0 * flow),
            "signed_notional": close * (10.0 + 30.0 * flow),
            "theta": close * (55.0 + 15.0 * flow),
            "threshold": close * 10.0,
            "threshold_scale": 1.0,
            "expected_ticks": 75.0,
            "bid_depth": 1_000.0 + np.arange(len(index)) % 13,
            "ask_depth": 900.0 + np.arange(len(index)) % 11,
            "bid_price": close * 0.9999,
            "ask_price": close * 1.0001,
        },
        index=index,
    )

    features = make_pipeline_features(
        frame,
        microstructure_window=10,
        information_window=12,
        sadf_windows=(16, 24),
    )

    expected = {
        "adf_t_16",
        "amihud_lambda",
        "cvd",
        "cvd_slope",
        "flow_shannon_entropy",
        "hasbrouck_lambda",
        "lob_depth_hhi",
        "lob_depth_imbalance",
        "kyle_lambda",
        "lz_complexity",
        "notional_side_hhi",
        "order_flow_imbalance",
        "relative_spread",
        "spread_elasticity",
        "sadf",
        "sadf_zscore",
        "shannon_entropy",
        "signed_notional_cumsum_ffd",
        "signed_notional_cumsum_ffd_zscore",
        "vpin",
        "vpin_zscore",
    }
    assert expected.issubset(features.columns)
    assert features[list(expected)].notna().any().all()


def test_moving_average_crossover_side_model_emits_cross_signals():
    index = pd.date_range("2024-01-01", periods=40, freq="min", tz="UTC")
    close = pd.Series(
        [5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 2] * 2,
        index=index,
        dtype=float,
    )

    side = moving_average_crossover_side(close, fast_window=10, slow_window=20)
    model = MovingAverageCrossoverSideModel(close, fast_window=10, slow_window=20)
    proba = model.predict_proba(pd.DataFrame({"x": np.arange(len(index))}, index=index))

    assert 1 in set(side)
    assert -1 in set(side)
    assert set(model.predict(proba).unique()).issubset({-1, 0, 1})
    assert (proba.loc[side == 1, 1] == 1.0).all()
    assert (proba.loc[side == -1, -1] == 1.0).all()


def test_make_bet_size_frame_adds_signed_bet_size():
    index = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    signals = pd.DataFrame(
        {
            "signal": [1, -1, 0],
            "confidence": [0.5, 0.8, 0.9],
        },
        index=index,
    )

    sized = make_bet_size_frame(signals, min_abs_size=0.1)

    assert bet_size_from_probability(0.5) == 0.0
    assert sized.iloc[0]["bet_size"] == 0.1
    assert sized.iloc[1]["bet_size"] < -0.1
    assert sized.iloc[2]["bet_size"] == 0.0


def test_make_bet_size_frame_discretizes_signed_bet_size():
    index = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    signals = pd.DataFrame(
        {
            "signal": [1, -1],
            "confidence": [0.7, 0.9],
        },
        index=index,
    )

    sized = make_bet_size_frame(signals, step_size=0.25)

    assert sized.iloc[0]["bet_size"] == 0.25
    assert sized.iloc[1]["bet_size"] == -0.75
    assert sized.iloc[1]["bet_size_abs"] == 0.75


def test_build_fit_and_write_signal_csv():
    index = pd.date_range("2024-01-01", periods=600, freq="15min", tz="UTC")
    steps = np.sin(np.arange(len(index)) / 13.0) * 0.001
    steps += np.cos(np.arange(len(index)) / 7.0) * 0.0005
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)

    dataset = build_afml_dataset(
        close,
        volatility_span=20,
        cusum_threshold_mult=0.35,
        vertical_barrier_bars=24,
        vertical_barrier_days=None,
    )
    model = SequentialBootstrapBaggingClassifier(n_estimators=3, max_samples=0.8, random_state=3)
    result = fit_afml_model(
        dataset,
        train_fraction=0.7,
        probability_threshold=0.5,
        model=model,
    )
    output_dir = Path("build") / "afml_test_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = write_signal_csv(result.signals, output_dir / f"signals_{uuid4().hex}.csv")

    expected_threshold = dataset.volatility.reindex(close.index).ffill() * 0.35
    expected_events = symmetric_cusum_filter(np.log(close), threshold=expected_threshold)
    assert list(dataset.t_events) == list(expected_events)
    assert dataset.labels["bin"].nunique() >= 2
    assert set(dataset.y.unique()).issubset({-1, 1})
    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0
    assert {"timestamp_ns", "signal", "confidence"}.issubset(pd.read_csv(path).columns)


def test_build_afml_dataset_can_keep_neutral_labels():
    index = pd.date_range("2024-01-01", periods=700, freq="15min", tz="UTC")
    close = pd.Series(100.0 + np.sin(np.arange(len(index)) / 11.0), index=index)

    dataset = build_afml_dataset(
        close,
        volatility_span=20,
        cusum_threshold_mult=0.2,
        pt_sl=(2.0, 2.0),
        vertical_barrier_bars=8,
        vertical_barrier_days=None,
        zero_on_vertical=True,
        drop_neutral=False,
    )
    dropped = build_afml_dataset(
        close,
        volatility_span=20,
        cusum_threshold_mult=0.2,
        pt_sl=(2.0, 2.0),
        vertical_barrier_bars=8,
        vertical_barrier_days=None,
        zero_on_vertical=True,
    )

    assert 0 in set(dataset.y.unique())
    assert 0 not in set(dropped.y.unique())


def test_build_meta_dataset_labels_primary_sides_as_trade_or_pass():
    index = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.sin(np.arange(len(index)) / 9.0) * 0.02), index=index)
    side_index = index[120:240:3]
    side = pd.Series(np.where(np.arange(len(side_index)) % 2 == 0, 1, -1), index=side_index)

    dataset = build_afml_meta_dataset(
        close,
        side,
        volatility_span=10,
        pt_sl=(1.0, 1.0),
        meta_label_min_ret=0.001,
        vertical_barrier_bars=8,
        vertical_barrier_days=None,
    )

    assert set(dataset.y.unique()).issubset({0, 1})
    assert set(dataset.events["side"].unique()).issubset({-1, 1})


def test_fit_afml_meta_model_outputs_primary_filtered_signals():
    index = pd.date_range("2024-01-01", periods=750, freq="15min", tz="UTC")
    steps = np.sin(np.arange(len(index)) / 11.0) * 0.001
    steps += np.cos(np.arange(len(index)) / 19.0) * 0.0007
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)
    dataset = build_afml_dataset(
        close,
        volatility_span=20,
        cusum_threshold_mult=0.35,
        vertical_barrier_bars=24,
        vertical_barrier_days=None,
    )

    result = fit_afml_meta_model(
        dataset,
        train_fraction=0.7,
        primary_probability_threshold=0.0,
        meta_probability_threshold=0.5,
        primary_model=SequentialBootstrapBaggingClassifier(
            n_estimators=3, max_samples=0.8, random_state=5
        ),
        meta_model=SequentialBootstrapBaggingClassifier(
            n_estimators=3, max_samples=0.8, random_state=7
        ),
        volatility_span=20,
        meta_label_min_ret=0.001,
        vertical_barrier_bars=24,
        vertical_barrier_days=None,
    )

    assert set(result.primary_dataset.y.unique()).issubset({-1, 1})
    assert set(result.meta_dataset.y.unique()).issubset({0, 1})
    assert {"signal", "primary_side", "meta_probability", "confidence"}.issubset(
        result.signals.columns
    )
    assert set(result.signals["signal"].unique()).issubset({-1, 0, 1})
    assert 0.0 <= result.meta_test_precision <= 1.0
    assert 0.0 <= result.meta_test_recall <= 1.0


def test_fit_afml_model_with_sequential_bootstrap_bagging():
    index = pd.date_range("2024-01-01", periods=700, freq="15min", tz="UTC")
    steps = np.sin(np.arange(len(index)) / 11.0) * 0.001
    steps += np.cos(np.arange(len(index)) / 19.0) * 0.0007
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)
    dataset = build_afml_dataset(
        close,
        volatility_span=20,
        cusum_threshold_mult=0.35,
        vertical_barrier_bars=24,
        vertical_barrier_days=None,
    )
    model = SequentialBootstrapBaggingClassifier(
        n_estimators=5,
        max_samples=0.8,
        min_samples_leaf=2,
        n_jobs=2,
        random_state=11,
    )

    result = fit_afml_model(
        dataset,
        train_fraction=0.7,
        probability_threshold=0.5,
        model=model,
    )

    assert len(result.model.estimators_) == 5
    assert len(result.model.bootstrap_indices_) == 5
    assert result.model.estimators_[0].__class__.__name__ == "DecisionTreeClassifier"
    assert result.model.estimators_[0].criterion == "entropy"
    assert result.model.estimators_[0].max_features == 1
    assert 0.0 <= result.test_accuracy <= 1.0


def test_mdi_mda_and_sfi_feature_importance():
    index = pd.date_range("2024-01-01", periods=700, freq="15min", tz="UTC")
    steps = np.sin(np.arange(len(index)) / 11.0) * 0.001
    steps += np.cos(np.arange(len(index)) / 19.0) * 0.0007
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)
    features = pd.DataFrame(
        {
            "signal_feature": np.sin(np.arange(len(index)) / 10.0),
            "noise_feature": np.cos(np.arange(len(index)) / 17.0),
        },
        index=index,
    )
    dataset = build_afml_dataset(
        close,
        features=features,
        volatility_span=20,
        cusum_threshold_mult=0.35,
        vertical_barrier_bars=24,
        vertical_barrier_days=None,
    )
    result = fit_afml_model(
        dataset,
        train_fraction=0.7,
        probability_threshold=0.5,
        model=SequentialBootstrapBaggingClassifier(
            n_estimators=3, max_samples=0.8, random_state=19
        ),
    )

    mdi = mdi_feature_importance(result.model)
    mda = mda_feature_importance(
        result.model,
        dataset.X.loc[result.test_index],
        dataset.y.loc[result.test_index],
        sample_weight=dataset.sample_weight.loc[result.test_index],
        n_repeats=2,
        random_state=23,
    )
    sfi = sfi_feature_importance(
        dataset,
        train_index=result.train_index,
        test_index=result.test_index,
        model_factory=lambda: SequentialBootstrapBaggingClassifier(
            n_estimators=2,
            max_samples=0.8,
            random_state=29,
        ),
    )

    assert set(mdi["feature"]) == {"signal_feature", "noise_feature"}
    assert set(mda["feature"]) == {"signal_feature", "noise_feature"}
    assert set(sfi["feature"]) == {"signal_feature", "noise_feature"}
    assert {"method", "feature", "importance", "std"}.issubset(mdi.columns)
    assert {"baseline_score"}.issubset(mda.columns)
    assert mdi["importance"].notna().all()
    assert mda["importance"].notna().all()
    assert sfi["importance"].notna().all()


def test_sequential_bootstrap_bagging_defaults_max_samples_to_average_uniqueness():
    bar_index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    event_index = bar_index[:4]
    t1 = pd.Series([bar_index[-1]] * len(event_index), index=event_index)
    indicator = get_indicator_matrix(bar_index, t1)
    expected_sample_length = int(
        np.ceil(average_uniqueness_from_indicator(indicator).mean() * len(event_index))
    )
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}, index=event_index)
    y = pd.Series([0, 1, 0, 1], index=event_index)
    model = SequentialBootstrapBaggingClassifier(n_estimators=3, random_state=13)

    model.fit(X, y, t1=t1, bar_index=bar_index)

    assert expected_sample_length < len(event_index)
    assert {len(sampled) for sampled in model.bootstrap_indices_} == {expected_sample_length}


def test_fit_afml_model_uses_explicit_date_ranges():
    index = pd.date_range("2024-01-01", periods=900, freq="15min", tz="UTC")
    steps = np.sin(np.arange(len(index)) / 17.0) * 0.001
    steps += np.cos(np.arange(len(index)) / 23.0) * 0.0007
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=index)
    dataset = build_afml_dataset(
        close,
        volatility_span=20,
        cusum_threshold_mult=0.35,
        vertical_barrier_bars=12,
        vertical_barrier_days=None,
    )

    result = fit_afml_model(
        dataset,
        train_start="2024-01-03",
        train_end="2024-01-05",
        test_start="2024-01-07",
        test_end="2024-01-09",
        probability_threshold=0.5,
        model=SequentialBootstrapBaggingClassifier(
            n_estimators=3, max_samples=0.8, random_state=17
        ),
    )

    assert result.train_index.min() >= pd.Timestamp("2024-01-03", tz="UTC")
    assert result.train_index.max() <= pd.Timestamp("2024-01-05 23:59:59.999999999", tz="UTC")
    assert result.test_index.min() >= pd.Timestamp("2024-01-07", tz="UTC")
    assert result.test_index.max() <= pd.Timestamp("2024-01-09 23:59:59.999999999", tz="UTC")
