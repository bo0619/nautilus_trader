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

import numpy as np
import pandas as pd
import pytest

import afml_scripts.backtest_synthetic_barriers_ml as synthetic_ml
from afml_scripts.backtest_synthetic_barriers_ml import SyntheticCandidateOutcome
from afml_scripts.backtest_synthetic_barriers_ml import barrier_outcome_for_candidate
from afml_scripts.backtest_synthetic_barriers_ml import build_primary_events
from afml_scripts.backtest_synthetic_barriers_ml import candidate_exit_levels
from afml_scripts.backtest_synthetic_barriers_ml import candidate_screen_row
from afml_scripts.backtest_synthetic_barriers_ml import event_state_features
from afml_scripts.backtest_synthetic_barriers_ml import rolling_slope_r2
from afml_scripts.backtest_synthetic_barriers_ml import round_trip_cost_from_labeling_config
from afml_scripts.backtest_synthetic_barriers_ml import trade_path_metrics


pytest.importorskip("numba")


def test_event_state_features_uses_decomposed_trend_for_trend_filters(monkeypatch):
    log_trend = np.linspace(10.0, 10.11, 12)
    residual = np.array([0.0, 0.04, -0.03, 0.05, -0.04, 0.03, -0.02, 0.06, -0.05, 0.04, -0.03, 0.05])
    log_close = log_trend + residual
    df = pd.DataFrame(
        {
            "close": np.exp(log_close),
            "buy_notional": np.arange(12, dtype=float) + 20.0,
            "sell_notional": np.ones(12, dtype=float),
        },
    )

    def fake_kalman_local_level(frame, price_column, q_over_r):
        assert price_column == "log_price"
        assert q_over_r == pytest.approx(0.25)
        return pd.Series(log_trend, index=frame.index)

    monkeypatch.setattr(synthetic_ml, "kalman_local_level", fake_kalman_local_level)

    features, observed_trend, observed_residual = event_state_features(
        df,
        price_column="close",
        fair_value_config={"q_over_r": 0.25},
        event_config={
            "trend_window_bars": 4,
            "micro_slope_window_bars": 3,
            "cvd_z_window_bars": 4,
        },
    )

    expected_slope, _ = rolling_slope_r2(log_trend, 3)
    _, expected_r2 = rolling_slope_r2(log_trend, 4)
    raw_slope, _ = rolling_slope_r2(log_close, 3)

    assert observed_trend[-1] == pytest.approx(log_trend[-1])
    assert observed_residual[-1] == pytest.approx(residual[-1])
    assert features["micro_slope"].iloc[-1] == pytest.approx(expected_slope[-1])
    assert features["trend_r2"].iloc[-1] == pytest.approx(expected_r2[-1])
    assert features["micro_slope"].iloc[-1] != pytest.approx(raw_slope[-1])


def test_event_state_features_defaults_cvd_z_to_200_bars(monkeypatch):
    n = 199
    log_trend = np.linspace(10.0, 10.1, n)
    df = pd.DataFrame(
        {
            "close": np.exp(log_trend),
            "signed_notional": np.arange(1, n + 1, dtype=float),
        },
    )

    monkeypatch.setattr(
        synthetic_ml,
        "kalman_local_level",
        lambda frame, price_column, q_over_r: pd.Series(log_trend, index=frame.index),
    )

    features, _, _ = event_state_features(
        df,
        price_column="close",
        fair_value_config={},
        event_config={
            "trend_window_bars": 4,
            "micro_slope_window_bars": 3,
        },
    )

    assert np.isnan(features["cvd_z"].iloc[-1])


def test_build_primary_events_requires_strict_cvd_z_thresholds():
    event_config = {
        "r2_threshold": 0.488,
        "long_micro_slope_min": 0.0,
        "short_micro_slope_max": 0.0,
        "long_cvd_z_min": 1.5,
        "short_cvd_z_max": -1.5,
        "min_event_gap_bars": 1,
    }
    base_features = {
        "trend_r2": [0.9, 0.9],
        "micro_slope": [0.01, -0.01],
        "residual_z": [0.0, 0.0],
        "log_ret_1": [0.0, 0.0],
        "log_ret_5": [0.0, 0.0],
        "log_ret_20": [0.0, 0.0],
        "realized_vol_20": [0.1, 0.1],
        "signed_notional_z": [0.0, 0.0],
    }
    log_trend = np.array([10.0, 10.01])
    log_residual = np.array([0.0, 0.0])

    equality_features = pd.DataFrame({**base_features, "cvd_z": [1.5, -1.5]})
    with pytest.raises(ValueError, match="No primary model events found"):
        build_primary_events(
            "TEST",
            df=pd.DataFrame(index=equality_features.index),
            feature_frame=equality_features,
            log_trend=log_trend,
            log_residual=log_residual,
            event_config=event_config,
            horizon=0,
        )

    crossing_features = pd.DataFrame({**base_features, "cvd_z": [1.500001, -1.500001]})
    events = build_primary_events(
        "TEST",
        df=pd.DataFrame(index=crossing_features.index),
        feature_frame=crossing_features,
        log_trend=log_trend,
        log_residual=log_residual,
        event_config=event_config,
        horizon=0,
    )

    assert events["side"].to_list() == [1, -1]


def test_build_primary_events_supports_asymmetric_long_short_thresholds():
    event_config = {
        "long": {
            "r2_threshold": 0.60,
            "micro_slope_min": 0.0,
            "cvd_z_min": 1.5,
        },
        "short": {
            "r2_threshold": 0.45,
            "micro_slope_max": 0.0,
            "cvd_z_min": 1.0,
        },
        "min_event_gap_bars": 1,
    }
    features = pd.DataFrame(
        {
            "trend_r2": [0.65, 0.50, 0.50, 0.65],
            "micro_slope": [0.01, -0.01, 0.01, -0.01],
            "cvd_z": [1.8, -1.2, 1.8, -1.2],
            "residual_z": [0.0, 0.0, 0.0, 0.0],
            "log_ret_1": [0.0, 0.0, 0.0, 0.0],
            "log_ret_5": [0.0, 0.0, 0.0, 0.0],
            "log_ret_20": [0.0, 0.0, 0.0, 0.0],
            "realized_vol_20": [0.1, 0.1, 0.1, 0.1],
            "signed_notional_z": [0.0, 0.0, 0.0, 0.0],
        },
    )

    events = build_primary_events(
        "TEST",
        df=pd.DataFrame(index=features.index),
        feature_frame=features,
        log_trend=np.array([10.0, 10.01, 10.02, 10.03]),
        log_residual=np.array([0.0, 0.0, 0.0, 0.0]),
        event_config=event_config,
        horizon=0,
    )

    assert events["event_idx"].to_list() == [0, 1, 3]
    assert events["side"].to_list() == [1, -1, -1]


def test_round_trip_cost_uses_component_taker_costs():
    config = {
        "slippage_per_side_rate": 0.0002,
        "maker_fee_rate": 0.0002,
        "taker_fee_rate": 0.0005,
        "fee_liquidity": "taker",
    }

    cost = round_trip_cost_from_labeling_config(config)

    assert cost == pytest.approx(0.0014)


def test_round_trip_cost_requires_explicit_component_costs():
    with pytest.raises(ValueError, match="missing explicit cost fields"):
        round_trip_cost_from_labeling_config({})


def test_candidate_exit_levels_apply_directional_pt_sl():
    events = pd.DataFrame({"side": [1, -1, 1, -1]})

    pt_levels, sl_levels = candidate_exit_levels(
        events,
        {
            "long_profit_taking_mult": 0.5,
            "long_stop_loss_mult": 1.5,
            "short_profit_taking_mult": 0.75,
            "short_stop_loss_mult": 2.0,
        },
        unit=0.01,
    )

    assert pt_levels.tolist() == pytest.approx([0.005, 0.0075, 0.005, 0.0075])
    assert sl_levels.tolist() == pytest.approx([0.015, 0.020, 0.015, 0.020])


def test_barrier_outcome_for_candidate_scores_payoff_hits_and_path_error():
    signed_returns = np.array(
        [
            [
                [0.005, 0.011, 0.020],
                [-0.005, -0.011, -0.020],
                [0.001, 0.002, 0.003],
            ],
        ],
        dtype=np.float32,
    )

    expected, payoff_std, pt_rate, sl_rate, vertical_rate, mean_exit_step = (
        barrier_outcome_for_candidate(
            signed_returns,
            pt_levels=np.array([0.01], dtype=np.float32),
            sl_levels=np.array([0.01], dtype=np.float32),
            horizon=3,
            cost=0.001,
            tie_policy="stop_loss_first",
        )
    )

    path_payoffs = np.array([0.009, -0.011, 0.002])
    assert expected[0] == pytest.approx(float(np.mean(path_payoffs)), abs=1e-7)
    assert payoff_std[0] == pytest.approx(float(np.std(path_payoffs)), rel=1e-6)
    assert pt_rate[0] == pytest.approx(1.0 / 3.0)
    assert sl_rate[0] == pytest.approx(1.0 / 3.0)
    assert vertical_rate[0] == pytest.approx(1.0 / 3.0)
    assert mean_exit_step[0] == pytest.approx(7.0 / 3.0)


def test_barrier_outcome_for_candidate_honors_tie_policy():
    signed_returns = np.array([[[0.0]]], dtype=np.float32)

    stop_loss_first = barrier_outcome_for_candidate(
        signed_returns,
        pt_levels=np.array([0.0], dtype=np.float32),
        sl_levels=np.array([0.0], dtype=np.float32),
        horizon=1,
        cost=0.0,
        tie_policy="stop_loss_first",
    )
    profit_taking_first = barrier_outcome_for_candidate(
        signed_returns,
        pt_levels=np.array([0.0], dtype=np.float32),
        sl_levels=np.array([0.0], dtype=np.float32),
        horizon=1,
        cost=0.0,
        tie_policy="profit_taking_first",
    )

    assert stop_loss_first[2][0] == 0.0
    assert stop_loss_first[3][0] == 1.0
    assert profit_taking_first[2][0] == 1.0
    assert profit_taking_first[3][0] == 0.0


def test_candidate_screen_row_reports_directional_runtime_parameters():
    expected = np.array(
        [0.010, 0.008, -0.004, 0.007],
        dtype=np.float32,
    )
    outcome = SyntheticCandidateOutcome(
        expected_payoff=expected,
        payoff_std=np.ones_like(expected) * 0.01,
        payoff_standard_error=np.ones_like(expected) * 0.001,
        pt_hit_rate=np.zeros_like(expected),
        sl_hit_rate=np.zeros_like(expected),
        vertical_rate=np.ones_like(expected),
        mean_exit_step=np.ones_like(expected) * 80.0,
        unit=0.01,
        n_paths=100,
    )

    row = candidate_screen_row(
        candidate_id="trial_0001",
        event_definition={
            "long": {"r2_threshold": 0.52, "cvd_z_min": 1.7},
            "short": {"r2_threshold": 0.44, "cvd_z_max": -1.2},
        },
        exit_definition={
            "long_profit_taking_mult": 0.5,
            "long_stop_loss_mult": 1.5,
            "short_profit_taking_mult": 0.75,
            "short_stop_loss_mult": 2.0,
        },
        outcome=outcome,
        label_threshold=0.0,
        n_events=len(expected),
    )

    assert row["candidate_id"] == "trial_0001"
    assert row["long_R2"] == pytest.approx(0.52)
    assert row["short_Z"] == pytest.approx(-1.2)
    assert row["long_PT"] == pytest.approx(0.5)
    assert row["short_SL"] == pytest.approx(2.0)
    assert row["screen_eligible_for_meta"]
    assert row["screen_label_positive_rate"] == pytest.approx(0.75)


def test_trade_path_metrics_reports_streaks_and_drawdown_in_event_order():
    returns = np.array([0.03, -0.02, 0.10, -0.05, -0.01, 0.20, -0.02])
    event_starts = np.array([4, 6, 0, 2, 5, 1, 3])

    metrics = trade_path_metrics(returns, event_starts=event_starts)

    assert metrics["trade_count"] == 7
    assert metrics["win_count"] == 3
    assert metrics["loss_count"] == 4
    assert metrics["max_consecutive_wins"] == 2
    assert metrics["max_consecutive_losses"] == 2
    assert metrics["ending_loss_streak"] == 2
    assert metrics["max_drawdown_log_return"] == pytest.approx(-0.07)
    assert metrics["total_expected_log_return"] == pytest.approx(0.23)
    assert metrics["profit_factor"] == pytest.approx(0.33 / 0.10)
