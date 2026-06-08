import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from afml_strategies.cvdslope_main import CvdSlopeRuleSideModel
from afml_strategies.cvdslope_main import apply_directional_exit_columns
from afml_strategies.cvdslope_main import apply_json_defaults
from afml_strategies.cvdslope_main import barrier_settings
from afml_strategies.cvdslope_main import directional_barrier_settings
from afml_strategies.cvdslope_main import make_rule_meta_features
from afml_strategies.cvdslope_main import make_rule_meta_signal_frame
from afml_strategies.cvdslope_main import primary_side_frame_from_rule
from afml_strategies.cvdslope_main import runtime_event_definition
from afml_strategies.cvdslope_main import validate_runtime_candidate_volatility


class AlwaysTradeMetaModel:
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                0: np.full(len(features), 0.2),
                1: np.full(len(features), 0.8),
            },
            index=features.index,
        )


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=None,
        start=None,
        end=None,
        train_start=None,
        train_end=None,
        test_start=None,
        test_end=None,
        test_days=None,
        train_days=None,
        no_backtest=False,
        no_progress=False,
        volatility_span=None,
        volatility_lookback_days=None,
        volatility_ewma_span=None,
        volatility_horizon_bars=None,
        volatility_window_bars=None,
        volatility_floor_quantile=None,
        pt_mult=None,
        sl_mult=None,
        vertical_barrier_bars=None,
        cusum_window=None,
        cusum_threshold_mult=None,
        cusum_floor_quantile=None,
        min_ret=None,
        oldest_weight=None,
        rare_label_min_pct=None,
        meta_label_min_ret=None,
        feature_fracdiff_d=None,
        feature_fracdiff_threshold=None,
        microstructure_window=None,
        information_window=None,
        primary_probability_threshold=None,
        meta_probability_threshold=None,
        min_weight_fraction_leaf=None,
        n_jobs=None,
        progress_interval=None,
        pca_components=None,
        pca_random_state=None,
        bet_size_step=None,
        trade_notional=None,
        starting_balance=None,
        default_leverage=None,
        confidence_threshold=None,
        min_position_change=None,
        log_level=None,
    )


def test_apply_json_defaults_uses_ewma_span_for_real_volatility():
    args = apply_json_defaults(
        default_args(),
        {
            "real_data_labeling": {
                "volatility_ewma_span": 80,
            },
        },
    )

    assert args.volatility_ewma_span == 80


def test_barrier_settings_uses_synthetic_runtime_candidate(tmp_path):
    symbol = "BTCUSDT"
    symbol_dir = tmp_path / symbol
    symbol_dir.mkdir()
    (symbol_dir / f"{symbol}_synthetic_ml_summary.json").write_text(
        json.dumps(
            {
                "runtime_strategy_candidate": {
                    "profit_taking_mult": 0.5,
                    "stop_loss_mult": 1.5,
                    "vertical_barrier_bars": 80,
                },
            },
        ),
        encoding="utf-8",
    )

    pt_sl, vertical_bars = barrier_settings(
        {
            "runtime_strategy": {
                "source": "synthetic_ml_summary",
                "synthetic_summary_dir": str(tmp_path),
            },
        },
        default_args(),
        symbol=symbol,
    )

    assert pt_sl == (0.5, 1.5)
    assert vertical_bars == 80


def test_directional_barrier_settings_uses_synthetic_runtime_candidate_exit_definition(tmp_path):
    symbol = "BTCUSDT"
    symbol_dir = tmp_path / symbol
    symbol_dir.mkdir()
    (symbol_dir / f"{symbol}_synthetic_ml_summary.json").write_text(
        json.dumps(
            {
                "runtime_strategy_candidate": {
                    "exit_definition": {
                        "long_profit_taking_mult": 0.5,
                        "long_stop_loss_mult": 1.5,
                        "short_profit_taking_mult": 0.75,
                        "short_stop_loss_mult": 2.0,
                    },
                    "vertical_barrier_bars": 80,
                },
            },
        ),
        encoding="utf-8",
    )

    exit_config, vertical_bars = directional_barrier_settings(
        {
            "runtime_strategy": {
                "source": "synthetic_ml_summary",
                "synthetic_summary_dir": str(tmp_path),
            },
        },
        default_args(),
        symbol=symbol,
    )

    assert exit_config == {
        "profit_taking_mult": 1.0,
        "stop_loss_mult": 1.0,
        "long_profit_taking_mult": 0.5,
        "long_stop_loss_mult": 1.5,
        "short_profit_taking_mult": 0.75,
        "short_stop_loss_mult": 2.0,
    }
    assert vertical_bars == 80


def test_runtime_event_definition_uses_synthetic_candidate_event_override(tmp_path):
    symbol = "BTCUSDT"
    symbol_dir = tmp_path / symbol
    symbol_dir.mkdir()
    (symbol_dir / f"{symbol}_synthetic_ml_summary.json").write_text(
        json.dumps(
            {
                "runtime_strategy_candidate": {
                    "event_definition": {
                        "long": {"r2_threshold": 0.62, "cvd_z_min": 1.8},
                        "short": {"r2_threshold": 0.43, "cvd_z_min": 1.2},
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    event_definition = runtime_event_definition(
        {
            "event_definition": {
                "trend_window_bars": 20,
                "r2_threshold": 0.5,
                "long_cvd_z_min": 1.5,
                "short_cvd_z_max": -1.5,
            },
            "runtime_strategy": {
                "source": "synthetic_ml_summary",
                "synthetic_summary_dir": str(tmp_path),
            },
        },
        symbol,
    )

    assert event_definition["trend_window_bars"] == 20
    assert event_definition["long"]["r2_threshold"] == 0.62
    assert event_definition["short"]["r2_threshold"] == 0.43
    assert event_definition["short"]["cvd_z_min"] == 1.2


def test_validate_runtime_candidate_volatility_requires_matching_ewma_span():
    args = default_args()
    args.volatility_ewma_span = 80

    validate_runtime_candidate_volatility(
        {
            "volatility_target": {
                "kind": "ewma_1bar_log_return_std",
                "span": 80,
            },
        },
        args=args,
        symbol="BTCUSDT",
    )

    with pytest.raises(ValueError, match="configured for span 80"):
        validate_runtime_candidate_volatility(
            {
                "volatility_target": {
                    "kind": "ewma_1bar_log_return_std",
                    "span": 40,
                },
            },
            args=args,
            symbol="BTCUSDT",
        )


def test_apply_directional_exit_columns_writes_signal_side_multipliers():
    index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    signals = pd.DataFrame({"signal": [1, -1, 0]}, index=index)

    out = apply_directional_exit_columns(
        signals,
        {
            "profit_taking_mult": 1.0,
            "stop_loss_mult": 1.0,
            "long_profit_taking_mult": 0.5,
            "long_stop_loss_mult": 1.5,
            "short_profit_taking_mult": 0.75,
            "short_stop_loss_mult": 2.0,
        },
    )

    assert out["profit_taking_mult"].to_list() == [0.5, 0.75, 1.0]
    assert out["stop_loss_mult"].to_list() == [1.5, 2.0, 1.0]


def test_cvdslope_rule_primary_feeds_meta_signals_only_for_rule_candidates():
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    features = pd.DataFrame(
        {
            "cvdslope_trend_r2": [0.9, 0.9, 0.9, 0.9],
            "cvdslope_micro_slope": [0.01, -0.01, 0.0, 0.01],
            "cvdslope_cvd_z": [1.500001, -1.500001, 2.0, 1.5],
            "other_feature": [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )
    model = CvdSlopeRuleSideModel(
        {
            "r2_threshold": 0.488,
            "long_micro_slope_min": 0.0,
            "short_micro_slope_max": 0.0,
            "long_cvd_z_min": 1.5,
            "short_cvd_z_max": -1.5,
        },
    )

    primary = primary_side_frame_from_rule(model, features, probability_threshold=0.0)
    meta_features = make_rule_meta_features(features, primary)
    signals = make_rule_meta_signal_frame(
        primary,
        AlwaysTradeMetaModel(),
        meta_features,
        meta_probability_threshold=0.5,
        active_index=pd.DatetimeIndex(index[:3]),
    )

    assert primary["primary_side"].to_list() == [1, -1, 0, 0]
    assert primary["primary_prob_0"].to_list() == [0.0, 0.0, 1.0, 1.0]
    assert signals["signal"].to_list() == [1, -1, 0, 0]
    assert signals["meta_probability"].to_list() == [0.8, 0.8, 0.0, 0.0]


def test_cvdslope_rule_supports_asymmetric_long_short_thresholds():
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    features = pd.DataFrame(
        {
            "cvdslope_trend_r2": [0.65, 0.50, 0.50, 0.65],
            "cvdslope_micro_slope": [0.01, -0.01, 0.01, -0.01],
            "cvdslope_cvd_z": [1.8, -1.2, 1.8, -1.2],
        },
        index=index,
    )
    model = CvdSlopeRuleSideModel(
        {
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
        },
    )

    primary = primary_side_frame_from_rule(model, features, probability_threshold=0.0)

    assert primary["primary_side"].to_list() == [1, -1, 0, -1]
