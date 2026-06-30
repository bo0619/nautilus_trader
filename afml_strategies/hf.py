from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_scripts.real_data_utils import configured_symbols  # noqa: E402
from afml_scripts.real_data_utils import event_state_features  # noqa: E402
from afml_scripts.real_data_utils import input_csv_for_symbol  # noqa: E402
from afml_scripts.real_data_utils import normalize_symbol  # noqa: E402
from afml_scripts.real_data_utils import read_real_bars  # noqa: E402
from afml_scripts.real_data_utils import round_trip_cost_from_execution_config  # noqa: E402
from afml_strategies.config_loader import load_afml_data_config  # noqa: E402
from afml_strategies.config_loader import resolve_repo_path  # noqa: E402
from afml_strategies.config_loader import section  # noqa: E402
from nautilus_trader.backtest.config import BacktestEngineConfig  # noqa: E402
from nautilus_trader.backtest.engine import BacktestEngine  # noqa: E402
from nautilus_trader.config import LoggingConfig  # noqa: E402
from nautilus_trader.examples.strategies.afml_signal import AfmlSignalStrategy  # noqa: E402
from nautilus_trader.examples.strategies.afml_signal import AfmlSignalStrategyConfig  # noqa: E402
from nautilus_trader.model.currencies import USDT  # noqa: E402
from nautilus_trader.model.data import Bar  # noqa: E402
from nautilus_trader.model.data import BarAggregation  # noqa: E402
from nautilus_trader.model.data import BarSpecification  # noqa: E402
from nautilus_trader.model.data import BarType  # noqa: E402
from nautilus_trader.model.enums import AccountType  # noqa: E402
from nautilus_trader.model.enums import BookType  # noqa: E402
from nautilus_trader.model.enums import OmsType  # noqa: E402
from nautilus_trader.model.enums import OrderSide  # noqa: E402
from nautilus_trader.model.enums import PriceType  # noqa: E402
from nautilus_trader.model.enums import TimeInForce  # noqa: E402
from nautilus_trader.model.identifiers import InstrumentId  # noqa: E402
from nautilus_trader.model.identifiers import Symbol  # noqa: E402
from nautilus_trader.model.identifiers import TraderId  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.instruments import CryptoPerpetual  # noqa: E402
from nautilus_trader.model.objects import Currency  # noqa: E402
from nautilus_trader.model.objects import Money  # noqa: E402
from nautilus_trader.model.objects import Price  # noqa: E402
from nautilus_trader.model.objects import Quantity  # noqa: E402
from nautilus_trader.research.afml_pipeline import BANNED_PRIMARY_EXPERT_NAMES  # noqa: E402
from nautilus_trader.research.afml_pipeline import PRIMARY_MOE_MODEL_NAME  # noqa: E402
from nautilus_trader.research.afml_pipeline import AfmlDataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import AfmlMoeWeightedVotePrimarySideModel  # noqa: E402
from nautilus_trader.research.afml_pipeline import RegimeSwitchingPrimarySideModel  # noqa: E402
from nautilus_trader.research.afml_pipeline import (  # noqa: E402
    SequentialBootstrapBaggingClassifier,
)
from nautilus_trader.research.afml_pipeline import _pca_dataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import afml_validation_diagnostics  # noqa: E402
from nautilus_trader.research.afml_pipeline import build_afml_dataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import build_afml_meta_dataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import calibrate_binary_probabilities  # noqa: E402
from nautilus_trader.research.afml_pipeline import fit_regime_state_model  # noqa: E402
from nautilus_trader.research.afml_pipeline import make_bet_size_frame  # noqa: E402
from nautilus_trader.research.afml_pipeline import make_ewma_volatility_target  # noqa: E402
from nautilus_trader.research.afml_pipeline import (  # noqa: E402
    make_meta_features as make_primary_meta_features,
)
from nautilus_trader.research.afml_pipeline import make_pipeline_features  # noqa: E402
from nautilus_trader.research.afml_pipeline import make_rolling_cusum_threshold  # noqa: E402
from nautilus_trader.research.afml_pipeline import mda_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import mdi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import pca_mdi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import sfi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import sharpe_ratio_diagnostics  # noqa: E402
from nautilus_trader.research.afml_pipeline import write_signal_csv  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("afml_strategies/output/hf")
DEFAULT_REAL_STRATEGY_CONFIG = Path("afml_strategies/hf.json")
DEFAULT_CALENDAR_WINDOWS = ("1D", "3D", "7D", "14D", "30D")
VOLATILITY_TARGET_KIND = "afml_daily_volatility_ewm_std"
BINANCE = Venue("BINANCE")
PRIMARY_CONTEXT_PREFIXES = (
    "primary_prob_",
    "primary_expert_",
    "primary_moe_",
    "primary_driver_",
    "primary_policy_",
    "primary_market_state_",
    "trend_",
    "reversion_",
    "selected_policy_",
    "volatility_state_",
    "flow_state_",
    "entropy_state_",
)
PRIMARY_CONTEXT_COLUMNS = {
    "primary_confidence",
    "primary_uncertainty",
    "primary_disagreement",
    "primary_expert_count",
    "primary_policy",
    "primary_market_state",
    "trend_score",
    "reversion_score",
    "volatility_state",
    "flow_state",
    "entropy_state",
}
PRIMARY_CVDSLOPE_RULE_NAME = "cvdslope_rule"
PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME = "order_flow_shock_rule"
PRIMARY_PROFIT_SESSION_FLOW_RULE_NAME = "profit_session_flow_rule"
PRIMARY_MAKER_PRECISION_RULE_NAME = "maker_precision_primary"
META_SEQUENTIAL_BOOTSTRAP_RF = "sequential_bootstrap_rf"
META_PROFIT_WEIGHTED_SEQUENTIAL_BOOTSTRAP_RF = "profit_weighted_sequential_bootstrap_rf"


def format_elapsed(seconds: float) -> str:
    return f"{max(0.0, float(seconds)):.1f}s"


def progress_enabled(args: argparse.Namespace | None) -> bool:
    return args is None or not bool(getattr(args, "no_progress", False))


def progress_print(args: argparse.Namespace | None, message: str) -> None:
    if progress_enabled(args):
        print(message, flush=True)


def progress_start(args: argparse.Namespace | None, message: str) -> float | None:
    if not progress_enabled(args):
        return None
    print(f"{message} ...", flush=True)
    return time.perf_counter()


def progress_done(args: argparse.Namespace | None, message: str, started: float | None) -> None:
    if started is None or not progress_enabled(args):
        return
    print(f"{message} done elapsed={format_elapsed(time.perf_counter() - started)}", flush=True)


@dataclass(frozen=True)
class TrainTestWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_bars: int


@dataclass(frozen=True)
class MetaTrainingResult:
    model: Any
    train_pred: pd.Series
    test_pred: pd.Series
    feature_columns: list[str]
    diagnostics: dict[str, Any]
    probability_threshold: float


@dataclass(frozen=True)
class PrimaryTrainingResult:
    model: Any
    dataset: AfmlDataset | None
    frame: pd.DataFrame
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    diagnostics: dict[str, Any]


def window_to_dict(window: TrainTestWindow) -> dict[str, Any]:
    return {
        "train_start": window.train_start,
        "train_end": window.train_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "embargo_bars": window.embargo_bars,
    }


class CvdSlopeAfmlSignalStrategy(AfmlSignalStrategy):
    """
    AFML signal strategy with intrabar high/low triple-barrier exits.
    """

    def __init__(self, config: AfmlSignalStrategyConfig) -> None:
        super().__init__(config)
        barrier_config = self._barrier_config()
        self._tie_policy = str(barrier_config.get("tie_policy", "stop_loss_first"))

    def _check_triple_barrier_exit(self, bar: Bar) -> bool:  # noqa: C901
        if self._entry_side is None or self._entry_price is None:
            return False
        if self._close_pending or self.portfolio.is_flat(self.config.instrument_id):
            return False

        self._entry_bar_count += 1
        entry_price = float(self._entry_price)
        if entry_price <= 0.0:
            return False

        if self._entry_target is not None and self._entry_target > 0.0:
            profit_taking_mult = self._entry_profit_taking_multiplier()
            stop_loss_mult = self._entry_stop_loss_multiplier()
            pt_level = (
                profit_taking_mult * self._entry_target
                if profit_taking_mult is not None
                else None
            )
            sl_level = (
                stop_loss_mult * self._entry_target
                if stop_loss_mult is not None
                else None
            )

            high = float(bar.high)
            low = float(bar.low)
            if self._entry_side == OrderSide.BUY:
                favorable = math.log(high / entry_price)
                adverse = math.log(low / entry_price)
            else:
                favorable = -math.log(low / entry_price)
                adverse = -math.log(high / entry_price)

            pt_hit = pt_level is not None and pt_level > 0.0 and favorable >= pt_level
            sl_hit = sl_level is not None and sl_level > 0.0 and adverse <= -sl_level
            if pt_hit and sl_hit:
                self._flatten()
                return True
            if sl_hit:
                self._flatten()
                return True
            if pt_hit:
                self._flatten()
                return True

        if (
            self._vertical_barrier_bars is not None
            and self._entry_bar_count >= self._vertical_barrier_bars
        ):
            self._flatten()
            return True

        if (
            self._vertical_barrier_days is not None
            and self._entry_ts_event is not None
            and int(bar.ts_event) - self._entry_ts_event
            >= int(self._vertical_barrier_days * 86_400_000_000_000)
        ):
            self._flatten()
            return True

        return False


class CvdSlopeRuleSideModel:
    """
    Deterministic CVD-slope primary side model.
    """

    requires_event_spans = False

    def __init__(self, event_config: dict[str, Any]) -> None:
        long_config = event_config.get("long")
        short_config = event_config.get("short")
        if not isinstance(long_config, dict):
            raise TypeError("event_definition.long must be a JSON object")
        if not isinstance(short_config, dict):
            raise TypeError("event_definition.short must be a JSON object")

        self.long_trend_r2_min = self._required_float(long_config, "r2_threshold", "event_definition.long")
        self.short_trend_r2_min = self._required_float(short_config, "r2_threshold", "event_definition.short")
        self.long_slope_min = self._required_float(long_config, "micro_slope_min", "event_definition.long")
        self.short_slope_max = self._required_float(short_config, "micro_slope_max", "event_definition.short")
        self.long_cvd_z_min = self._required_float(long_config, "cvd_z_min", "event_definition.long")
        self.short_cvd_z_max = self._required_float(short_config, "cvd_z_max", "event_definition.short")
        self.classes_ = np.array([-1, 0, 1], dtype=int)
        self.feature_names_: list[str] | None = None

    @staticmethod
    def _required_float(config: dict[str, Any], key: str, path: str) -> float:
        if key not in config:
            raise ValueError(f"{path}.{key} is required")
        return float(config[key])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> CvdSlopeRuleSideModel:
        self.feature_names_ = list(X.columns)
        return self

    def _feature(self, X: pd.DataFrame, name: str) -> pd.Series:
        prefixed = f"cvdslope_{name}"
        if prefixed in X:
            return pd.to_numeric(X[prefixed], errors="coerce")
        if name in X:
            return pd.to_numeric(X[name], errors="coerce")
        raise ValueError(f"CVD-slope primary rule requires feature {prefixed!r}.")

    def primary_side(self, X: pd.DataFrame) -> pd.Series:
        trend_r2 = self._feature(X, "trend_r2")
        micro_slope = self._feature(X, "micro_slope")
        cvd_z = self._feature(X, "cvd_z")
        side = pd.Series(0, index=X.index, dtype=int)
        long_mask = (
            (trend_r2 > self.long_trend_r2_min)
            & (micro_slope > self.long_slope_min)
            & (cvd_z > self.long_cvd_z_min)
        )
        short_mask = (
            (trend_r2 > self.short_trend_r2_min)
            & (micro_slope < self.short_slope_max)
            & (cvd_z < self.short_cvd_z_max)
        )
        side.loc[long_mask.fillna(False)] = 1
        side.loc[short_mask.fillna(False)] = -1
        return side.rename("prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        side = self.primary_side(X)
        proba = pd.DataFrame(0.0, index=X.index, columns=self.classes_)
        proba.loc[side < 0, -1] = 1.0
        proba.loc[side == 0, 0] = 1.0
        proba.loc[side > 0, 1] = 1.0
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.primary_side(X)


class MeanReversionRuleSideModel:
    """
    Deterministic overextension primary side model for the reversion policy.
    """

    requires_event_spans = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config if isinstance(config, dict) else {}
        self.price_z_abs_min = float(config.get("price_z_abs_min", 1.0))
        self.max_trend_r2 = float(config.get("max_trend_r2", 0.80))
        self.min_flow_exhaustion_z = float(config.get("min_flow_exhaustion_z", 0.0))
        self.classes_ = np.array([-1, 0, 1], dtype=int)
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> MeanReversionRuleSideModel:
        self.feature_names_ = list(X.columns)
        return self

    @staticmethod
    def _first_feature(X: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in X:
                return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(0.0, index=X.index, dtype=float)

    def primary_side(self, X: pd.DataFrame) -> pd.Series:
        price_z = self._first_feature(
            X,
            ("zscore_slow", "calendar_log_return_robust_z_1d", "log_return_1"),
        )
        trend_r2 = self._first_feature(X, ("cvdslope_trend_r2", "trend_r2"))
        flow_exhaustion = self._first_feature(
            X,
            ("vpin_zscore", "theta_to_threshold_robust_zscore", "signed_dollar_imbalance_abs"),
        ).abs()
        side = pd.Series(0, index=X.index, dtype=int)
        regime_ok = trend_r2 <= self.max_trend_r2
        if self.min_flow_exhaustion_z > 0.0:
            regime_ok &= flow_exhaustion >= self.min_flow_exhaustion_z
        side.loc[(price_z <= -self.price_z_abs_min) & regime_ok] = 1
        side.loc[(price_z >= self.price_z_abs_min) & regime_ok] = -1
        return side.rename("prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        side = self.primary_side(X)
        proba = pd.DataFrame(0.0, index=X.index, columns=self.classes_)
        proba.loc[side < 0, -1] = 1.0
        proba.loc[side == 0, 0] = 1.0
        proba.loc[side > 0, 1] = 1.0
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.primary_side(X)


class OrderFlowShockRuleSideModel:
    """
    Deterministic order-flow shock primary side model.

    The rule uses contemporaneous bar-level order-flow and liquidity proxies,
    not CVD-slope trend fields. It is intended for higher-frequency bars where
    flow imbalance and adverse-selection pressure should dominate slower trend
    descriptors.
    """

    requires_event_spans = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config if isinstance(config, dict) else {}
        self.min_signed_dollar_imbalance = self._optional_float(
            config,
            "min_signed_dollar_imbalance",
            0.0,
        )
        self.min_signed_dollar_imbalance_quantile = self._optional_quantile(
            config,
            "min_signed_dollar_imbalance_quantile",
        )
        self.min_order_flow_imbalance = self._optional_float(
            config,
            "min_order_flow_imbalance",
            0.0,
        )
        self.min_order_flow_imbalance_quantile = self._optional_quantile(
            config,
            "min_order_flow_imbalance_quantile",
        )
        self.max_order_flow_imbalance = self._optional_float(
            config,
            "max_order_flow_imbalance",
            None,
        )
        self.max_order_flow_imbalance_quantile = self._optional_quantile(
            config,
            "max_order_flow_imbalance_quantile",
        )
        self.min_buy_notional_share = self._optional_float(config, "min_buy_notional_share", None)
        self.min_buy_notional_share_quantile = self._optional_quantile(
            config,
            "min_buy_notional_share_quantile",
        )
        self.min_tick_imbalance = self._optional_float(config, "min_tick_imbalance", None)
        self.min_tick_imbalance_quantile = self._optional_quantile(config, "min_tick_imbalance_quantile")
        self.min_vpin_zscore = self._optional_float(config, "min_vpin_zscore", None)
        self.min_vpin_zscore_quantile = self._optional_quantile(config, "min_vpin_zscore_quantile")
        self.max_vpin_zscore = self._optional_float(config, "max_vpin_zscore", None)
        self.max_vpin_zscore_quantile = self._optional_quantile(config, "max_vpin_zscore_quantile")
        self.min_vpin = self._optional_float(config, "min_vpin", None)
        self.min_vpin_quantile = self._optional_quantile(config, "min_vpin_quantile")
        self.max_vpin = self._optional_float(config, "max_vpin", None)
        self.max_vpin_quantile = self._optional_quantile(config, "max_vpin_quantile")
        self.min_notional_side_hhi_zscore = self._optional_float(
            config,
            "min_notional_side_hhi_zscore",
            None,
        )
        self.min_notional_side_hhi_zscore_quantile = self._optional_quantile(
            config,
            "min_notional_side_hhi_zscore_quantile",
        )
        self.max_bar_spread_proxy = self._optional_float(config, "max_bar_spread_proxy", None)
        self.max_bar_spread_proxy_quantile = self._optional_quantile(
            config,
            "max_bar_spread_proxy_quantile",
        )
        self.max_spread_zscore = self._optional_float(config, "max_spread_zscore", None)
        self.max_spread_zscore_quantile = self._optional_quantile(config, "max_spread_zscore_quantile")
        self.min_abs_log_return = self._optional_float(config, "min_abs_log_return", None)
        self.min_abs_log_return_quantile = self._optional_quantile(config, "min_abs_log_return_quantile")
        self.require_positive_return = bool(config.get("require_positive_return", False))
        self.allow_short = bool(config.get("allow_short", False))
        pullback_config = config.get("pullback_continuation", {})
        self.pullback_config = pullback_config if isinstance(pullback_config, dict) else {}
        self.pullback_enabled = bool(self.pullback_config.get("enabled", False))
        self.pullback_thresholds = self._pullback_thresholds_from_config(self.pullback_config)
        self.pullback_quantiles = self._pullback_quantiles_from_config(self.pullback_config)
        self.classes_ = np.array([-1, 0, 1], dtype=int)
        self.feature_names_: list[str] | None = None
        self.calibration_rows_: int = 0

    @staticmethod
    def _optional_float(
        config: dict[str, Any],
        key: str,
        default: float | None,
    ) -> float | None:
        value = config.get(key, default)
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _optional_quantile(config: dict[str, Any], key: str) -> float | None:
        value = config.get(key)
        if value is None:
            return None
        quantile = float(value)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"primary_model.order_flow_shock.{key} must be between 0 and 1")
        return quantile

    @classmethod
    def _pullback_thresholds_from_config(cls, config: dict[str, Any]) -> dict[str, float | None]:
        return {
            "min_signed_dollar_imbalance": cls._optional_float(
                config,
                "min_signed_dollar_imbalance",
                None,
            ),
            "min_order_flow_imbalance": cls._optional_float(config, "min_order_flow_imbalance", None),
            "max_order_flow_imbalance": cls._optional_float(config, "max_order_flow_imbalance", None),
            "min_buy_notional_share": cls._optional_float(config, "min_buy_notional_share", None),
            "max_vpin_zscore": cls._optional_float(config, "max_vpin_zscore", None),
            "max_log_return": cls._optional_float(config, "max_log_return", 0.0),
            "min_notional_side_hhi_zscore": cls._optional_float(
                config,
                "min_notional_side_hhi_zscore",
                None,
            ),
            "max_bar_spread_proxy": cls._optional_float(config, "max_bar_spread_proxy", None),
        }

    @classmethod
    def _pullback_quantiles_from_config(cls, config: dict[str, Any]) -> dict[str, float | None]:
        return {
            "min_signed_dollar_imbalance": cls._optional_quantile(
                config,
                "min_signed_dollar_imbalance_quantile",
            ),
            "min_order_flow_imbalance": cls._optional_quantile(
                config,
                "min_order_flow_imbalance_quantile",
            ),
            "max_order_flow_imbalance": cls._optional_quantile(
                config,
                "max_order_flow_imbalance_quantile",
            ),
            "min_buy_notional_share": cls._optional_quantile(
                config,
                "min_buy_notional_share_quantile",
            ),
            "max_vpin_zscore": cls._optional_quantile(config, "max_vpin_zscore_quantile"),
            "max_log_return": cls._optional_quantile(config, "max_log_return_quantile"),
            "min_notional_side_hhi_zscore": cls._optional_quantile(
                config,
                "min_notional_side_hhi_zscore_quantile",
            ),
            "max_bar_spread_proxy": cls._optional_quantile(config, "max_bar_spread_proxy_quantile"),
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> OrderFlowShockRuleSideModel:
        self.feature_names_ = list(X.columns)
        self._fit_quantile_thresholds(X)
        return self

    @staticmethod
    def _first_feature(X: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in X:
                return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(0.0, index=X.index, dtype=float)

    def _feature_components(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "signed_dollar": self._first_feature(
                X,
                ("signed_dollar_imbalance", "calendar_signed_dollar_imbalance_1d"),
            ),
            "order_flow": self._first_feature(
                X,
                ("order_flow_imbalance", "calendar_order_flow_imbalance_1d"),
            ),
            "buy_share": self._first_feature(X, ("buy_notional_share",)),
            "tick_imbalance": self._first_feature(X, ("tick_imbalance", "buy_tick_ratio")),
            "vpin_zscore": self._first_feature(X, ("vpin_zscore", "theta_to_threshold_robust_zscore")),
            "vpin": self._first_feature(X, ("vpin",)),
            "hhi_zscore": self._first_feature(X, ("notional_side_hhi_zscore",)),
            "bar_spread": self._first_feature(X, ("bar_spread_proxy",)),
            "spread_zscore": self._first_feature(X, ("spread_zscore",)),
            "log_return": self._first_feature(X, ("log_return_1", "oc_return", "close_return")),
        }

    def _fit_quantile_thresholds(self, X: pd.DataFrame) -> None:
        components = self._feature_components(X)
        self.calibration_rows_ = len(X)
        self._set_quantile_threshold(
            "min_signed_dollar_imbalance",
            components["signed_dollar"],
            self.min_signed_dollar_imbalance_quantile,
        )
        self._set_quantile_threshold(
            "min_order_flow_imbalance",
            components["order_flow"],
            self.min_order_flow_imbalance_quantile,
        )
        self._set_quantile_threshold(
            "max_order_flow_imbalance",
            components["order_flow"],
            self.max_order_flow_imbalance_quantile,
        )
        self._set_quantile_threshold(
            "min_buy_notional_share",
            components["buy_share"],
            self.min_buy_notional_share_quantile,
        )
        self._set_quantile_threshold(
            "min_tick_imbalance",
            components["tick_imbalance"],
            self.min_tick_imbalance_quantile,
        )
        self._set_quantile_threshold(
            "min_vpin_zscore",
            components["vpin_zscore"],
            self.min_vpin_zscore_quantile,
        )
        self._set_quantile_threshold(
            "max_vpin_zscore",
            components["vpin_zscore"],
            self.max_vpin_zscore_quantile,
        )
        self._set_quantile_threshold("min_vpin", components["vpin"], self.min_vpin_quantile)
        self._set_quantile_threshold("max_vpin", components["vpin"], self.max_vpin_quantile)
        self._set_quantile_threshold(
            "min_notional_side_hhi_zscore",
            components["hhi_zscore"],
            self.min_notional_side_hhi_zscore_quantile,
        )
        self._set_quantile_threshold(
            "max_bar_spread_proxy",
            components["bar_spread"],
            self.max_bar_spread_proxy_quantile,
        )
        self._set_quantile_threshold(
            "max_spread_zscore",
            components["spread_zscore"],
            self.max_spread_zscore_quantile,
        )
        self._set_quantile_threshold(
            "min_abs_log_return",
            components["log_return"].abs(),
            self.min_abs_log_return_quantile,
        )
        self._fit_pullback_quantile_thresholds(components)

    @staticmethod
    def _valid_threshold_values(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    def _set_quantile_threshold(
        self,
        attribute: str,
        values: pd.Series,
        quantile: float | None,
    ) -> None:
        if quantile is None:
            return
        valid = self._valid_threshold_values(values)
        if valid.empty:
            return
        setattr(self, attribute, float(valid.quantile(quantile)))

    def _set_pullback_quantile_threshold(
        self,
        key: str,
        values: pd.Series,
    ) -> None:
        quantile = self.pullback_quantiles.get(key)
        if quantile is None:
            return
        valid = self._valid_threshold_values(values)
        if valid.empty:
            return
        self.pullback_thresholds[key] = float(valid.quantile(quantile))

    def _fit_pullback_quantile_thresholds(self, components: dict[str, pd.Series]) -> None:
        if not self.pullback_enabled:
            return
        self._set_pullback_quantile_threshold("min_signed_dollar_imbalance", components["signed_dollar"])
        self._set_pullback_quantile_threshold("min_order_flow_imbalance", components["order_flow"])
        self._set_pullback_quantile_threshold("max_order_flow_imbalance", components["order_flow"])
        self._set_pullback_quantile_threshold("min_buy_notional_share", components["buy_share"])
        self._set_pullback_quantile_threshold("max_vpin_zscore", components["vpin_zscore"])
        self._set_pullback_quantile_threshold("max_log_return", components["log_return"])
        self._set_pullback_quantile_threshold("min_notional_side_hhi_zscore", components["hhi_zscore"])
        self._set_pullback_quantile_threshold("max_bar_spread_proxy", components["bar_spread"])

    @staticmethod
    def _threshold_mask(
        values: pd.Series,
        threshold: float | None,
        *,
        side: int,
        greater_is_long: bool = True,
    ) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        signed_values = values if (side > 0) == greater_is_long else -values
        return signed_values >= float(threshold)

    def _flow_mask(
        self,
        *,
        side: int,
        signed_dollar: pd.Series,
        order_flow: pd.Series,
        buy_share: pd.Series,
        tick_imbalance: pd.Series,
    ) -> pd.Series:
        mask = self._threshold_mask(
            signed_dollar,
            self.min_signed_dollar_imbalance,
            side=side,
        )
        mask &= self._threshold_mask(
            order_flow,
            self.min_order_flow_imbalance,
            side=side,
        )
        if self.max_order_flow_imbalance is not None:
            signed_order_flow = order_flow if side > 0 else -order_flow
            mask &= signed_order_flow <= self.max_order_flow_imbalance
        if self.min_buy_notional_share is not None:
            share = buy_share if side > 0 else 1.0 - buy_share
            mask &= share >= self.min_buy_notional_share
        if self.min_tick_imbalance is not None:
            mask &= self._threshold_mask(tick_imbalance, self.min_tick_imbalance, side=side)
        return mask

    def _adverse_selection_mask(
        self,
        *,
        vpin_zscore: pd.Series,
        vpin: pd.Series,
        hhi_zscore: pd.Series,
    ) -> pd.Series:
        mask = pd.Series(True, index=vpin_zscore.index)
        if self.min_vpin_zscore is not None:
            mask &= vpin_zscore >= self.min_vpin_zscore
        if self.max_vpin_zscore is not None:
            mask &= vpin_zscore <= self.max_vpin_zscore
        if self.min_vpin is not None:
            mask &= vpin >= self.min_vpin
        if self.max_vpin is not None:
            mask &= vpin <= self.max_vpin
        if self.min_notional_side_hhi_zscore is not None:
            mask &= hhi_zscore >= self.min_notional_side_hhi_zscore
        return mask

    def _liquidity_mask(
        self,
        *,
        bar_spread: pd.Series,
        spread_zscore: pd.Series,
    ) -> pd.Series:
        mask = pd.Series(True, index=bar_spread.index)
        if self.max_bar_spread_proxy is not None:
            mask &= bar_spread <= self.max_bar_spread_proxy
        if self.max_spread_zscore is not None:
            mask &= spread_zscore <= self.max_spread_zscore
        return mask

    def _return_mask(self, *, side: int, log_return: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=log_return.index)
        if self.min_abs_log_return is not None:
            mask &= log_return.abs() >= self.min_abs_log_return
        if self.require_positive_return:
            same_direction = log_return > 0.0 if side > 0 else log_return < 0.0
            mask &= same_direction
        return mask

    @staticmethod
    def _optional_ge_mask(values: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        return values >= float(threshold)

    @staticmethod
    def _optional_le_mask(values: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        return values <= float(threshold)

    def _pullback_continuation_mask(self, components: dict[str, pd.Series]) -> pd.Series:
        if not self.pullback_enabled:
            return pd.Series(False, index=components["log_return"].index)
        mask = self._optional_ge_mask(
            components["signed_dollar"],
            self.pullback_thresholds.get("min_signed_dollar_imbalance"),
        )
        mask &= self._optional_ge_mask(
            components["order_flow"],
            self.pullback_thresholds.get("min_order_flow_imbalance"),
        )
        mask &= self._optional_le_mask(
            components["order_flow"],
            self.pullback_thresholds.get("max_order_flow_imbalance"),
        )
        mask &= self._optional_ge_mask(
            components["buy_share"],
            self.pullback_thresholds.get("min_buy_notional_share"),
        )
        mask &= self._optional_le_mask(
            components["vpin_zscore"],
            self.pullback_thresholds.get("max_vpin_zscore"),
        )
        mask &= self._optional_le_mask(
            components["log_return"],
            self.pullback_thresholds.get("max_log_return"),
        )
        mask &= self._optional_ge_mask(
            components["hhi_zscore"],
            self.pullback_thresholds.get("min_notional_side_hhi_zscore"),
        )
        mask &= self._optional_le_mask(
            components["bar_spread"],
            self.pullback_thresholds.get("max_bar_spread_proxy"),
        )
        return mask.fillna(False)

    def primary_side(self, X: pd.DataFrame) -> pd.Series:
        components = self._feature_components(X)

        def shock_mask(side: int) -> pd.Series:
            mask = self._flow_mask(
                side=side,
                signed_dollar=components["signed_dollar"],
                order_flow=components["order_flow"],
                buy_share=components["buy_share"],
                tick_imbalance=components["tick_imbalance"],
            )
            mask &= self._adverse_selection_mask(
                vpin_zscore=components["vpin_zscore"],
                vpin=components["vpin"],
                hhi_zscore=components["hhi_zscore"],
            )
            mask &= self._liquidity_mask(
                bar_spread=components["bar_spread"],
                spread_zscore=components["spread_zscore"],
            )
            mask &= self._return_mask(side=side, log_return=components["log_return"])
            return mask.fillna(False)

        side = pd.Series(0, index=X.index, dtype=int)
        side.loc[shock_mask(1) | self._pullback_continuation_mask(components)] = 1
        if self.allow_short:
            side.loc[shock_mask(-1)] = -1
        return side.rename("prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        side = self.primary_side(X)
        proba = pd.DataFrame(0.0, index=X.index, columns=self.classes_)
        proba.loc[side < 0, -1] = 1.0
        proba.loc[side == 0, 0] = 1.0
        proba.loc[side > 0, 1] = 1.0
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.primary_side(X)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "calibration_rows": self.calibration_rows_,
            "min_signed_dollar_imbalance": self.min_signed_dollar_imbalance,
            "min_signed_dollar_imbalance_quantile": self.min_signed_dollar_imbalance_quantile,
            "min_order_flow_imbalance": self.min_order_flow_imbalance,
            "min_order_flow_imbalance_quantile": self.min_order_flow_imbalance_quantile,
            "max_order_flow_imbalance": self.max_order_flow_imbalance,
            "max_order_flow_imbalance_quantile": self.max_order_flow_imbalance_quantile,
            "min_buy_notional_share": self.min_buy_notional_share,
            "min_buy_notional_share_quantile": self.min_buy_notional_share_quantile,
            "min_tick_imbalance": self.min_tick_imbalance,
            "min_tick_imbalance_quantile": self.min_tick_imbalance_quantile,
            "min_vpin_zscore": self.min_vpin_zscore,
            "min_vpin_zscore_quantile": self.min_vpin_zscore_quantile,
            "max_vpin_zscore": self.max_vpin_zscore,
            "max_vpin_zscore_quantile": self.max_vpin_zscore_quantile,
            "min_vpin": self.min_vpin,
            "min_vpin_quantile": self.min_vpin_quantile,
            "max_vpin": self.max_vpin,
            "max_vpin_quantile": self.max_vpin_quantile,
            "min_notional_side_hhi_zscore": self.min_notional_side_hhi_zscore,
            "min_notional_side_hhi_zscore_quantile": self.min_notional_side_hhi_zscore_quantile,
            "max_bar_spread_proxy": self.max_bar_spread_proxy,
            "max_bar_spread_proxy_quantile": self.max_bar_spread_proxy_quantile,
            "max_spread_zscore": self.max_spread_zscore,
            "max_spread_zscore_quantile": self.max_spread_zscore_quantile,
            "min_abs_log_return": self.min_abs_log_return,
            "min_abs_log_return_quantile": self.min_abs_log_return_quantile,
            "require_positive_return": self.require_positive_return,
            "allow_short": self.allow_short,
            "pullback_continuation": {
                "enabled": self.pullback_enabled,
                "thresholds": self.pullback_thresholds,
                "quantiles": self.pullback_quantiles,
            },
        }


class ProfitSessionFlowRuleSideModel:
    """
    High-frequency primary side rule built from train-calibrated flow/session filters.

    The rule is intentionally narrow: it proposes trades only when signed flow,
    participation, liquidity, and UTC session filters agree. The meta model then
    decides whether those opportunities are worth taking.
    """

    requires_event_spans = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config if isinstance(config, dict) else {}
        self.allowed_sessions_utc = tuple(str(item) for item in config.get("allowed_sessions_utc", []))
        self.allowed_weekdays = tuple(int(item) for item in config.get("allowed_weekdays", []))
        self.allow_short = bool(config.get("allow_short", False))
        self.require_directional_return = bool(config.get("require_directional_return", True))
        self.long_signed_dollar_quantile = self._optional_quantile(
            config,
            "long_signed_dollar_quantile",
            0.60,
        )
        self.short_signed_dollar_quantile = self._optional_quantile(
            config,
            "short_signed_dollar_quantile",
            0.40,
        )
        self.long_order_flow_quantile = self._optional_quantile(config, "long_order_flow_quantile", 0.60)
        self.short_order_flow_quantile = self._optional_quantile(config, "short_order_flow_quantile", 0.40)
        self.min_buy_notional_share_quantile = self._optional_quantile(
            config,
            "min_buy_notional_share_quantile",
            0.55,
        )
        self.min_abs_log_return_quantile = self._optional_quantile(
            config,
            "min_abs_log_return_quantile",
            0.50,
        )
        self.max_vpin_zscore_quantile = self._optional_quantile(config, "max_vpin_zscore_quantile", 0.90)
        self.max_bar_spread_proxy_quantile = self._optional_quantile(
            config,
            "max_bar_spread_proxy_quantile",
            0.90,
        )
        self.min_theta_to_threshold_quantile = self._optional_quantile(
            config,
            "min_theta_to_threshold_quantile",
            0.40,
        )
        self.min_bar_density_quantile = self._optional_quantile(
            config,
            "min_bar_density_quantile",
            None,
        )
        self.thresholds_: dict[str, float | None] = {}
        self.classes_ = np.array([-1, 0, 1], dtype=int)
        self.feature_names_: list[str] | None = None
        self.calibration_rows_: int = 0

    @staticmethod
    def _optional_quantile(
        config: dict[str, Any],
        key: str,
        default: float | None,
    ) -> float | None:
        value = config.get(key, default)
        if value is None:
            return None
        quantile = float(value)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"primary_model.profit_session_flow.{key} must be in [0, 1]")
        return quantile

    @staticmethod
    def _first_feature(X: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in X:
                return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(0.0, index=X.index, dtype=float)

    def _feature_components(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "signed_dollar": self._first_feature(
                X,
                ("signed_dollar_imbalance", "calendar_signed_notional_robust_z_1d"),
            ),
            "order_flow": self._first_feature(
                X,
                ("order_flow_imbalance", "calendar_order_flow_imbalance_1d"),
            ),
            "buy_share": self._first_feature(X, ("buy_notional_share",)),
            "vpin_zscore": self._first_feature(X, ("vpin_zscore", "theta_to_threshold_robust_zscore")),
            "bar_spread": self._first_feature(X, ("bar_spread_proxy",)),
            "log_return": self._first_feature(X, ("log_return_1", "oc_return", "cvdslope_log_ret_1")),
            "theta_to_threshold": self._first_feature(X, ("theta_to_threshold",)),
            "bar_density": self._first_feature(
                X,
                ("calendar_bar_density_per_day_1d", "calendar_bar_density_per_day_3d"),
            ),
        }

    @staticmethod
    def _valid_values(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    def _quantile_value(
        self,
        values: pd.Series,
        quantile: float | None,
    ) -> float | None:
        if quantile is None:
            return None
        valid = self._valid_values(values)
        if valid.empty:
            return None
        return float(valid.quantile(quantile))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> ProfitSessionFlowRuleSideModel:
        self.feature_names_ = list(X.columns)
        components = self._feature_components(X)
        self.calibration_rows_ = len(X)
        self.thresholds_ = {
            "long_signed_dollar": self._quantile_value(
                components["signed_dollar"],
                self.long_signed_dollar_quantile,
            ),
            "short_signed_dollar": self._quantile_value(
                components["signed_dollar"],
                self.short_signed_dollar_quantile,
            ),
            "long_order_flow": self._quantile_value(
                components["order_flow"],
                self.long_order_flow_quantile,
            ),
            "short_order_flow": self._quantile_value(
                components["order_flow"],
                self.short_order_flow_quantile,
            ),
            "min_buy_notional_share": self._quantile_value(
                components["buy_share"],
                self.min_buy_notional_share_quantile,
            ),
            "min_abs_log_return": self._quantile_value(
                components["log_return"].abs(),
                self.min_abs_log_return_quantile,
            ),
            "max_vpin_zscore": self._quantile_value(
                components["vpin_zscore"],
                self.max_vpin_zscore_quantile,
            ),
            "max_bar_spread_proxy": self._quantile_value(
                components["bar_spread"],
                self.max_bar_spread_proxy_quantile,
            ),
            "min_theta_to_threshold": self._quantile_value(
                components["theta_to_threshold"],
                self.min_theta_to_threshold_quantile,
            ),
            "min_bar_density": self._quantile_value(
                components["bar_density"],
                self.min_bar_density_quantile,
            ),
        }
        return self

    @staticmethod
    def _ge(values: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        return values >= float(threshold)

    @staticmethod
    def _le(values: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        return values <= float(threshold)

    def _session_mask(self, index: pd.Index) -> pd.Series:
        if not self.allowed_sessions_utc:
            return pd.Series(True, index=index)
        hours = pd.Series(pd.DatetimeIndex(index).hour, index=index)
        mask = pd.Series(False, index=index)
        for raw in self.allowed_sessions_utc:
            text = raw.strip().replace("-", "_")
            if not text:
                continue
            if "_" in text:
                start_text, end_text = text.split("_", 1)
                start_hour = int(start_text)
                end_hour = int(end_text)
                if start_hour <= end_hour:
                    mask |= (hours >= start_hour) & (hours <= end_hour)
                else:
                    mask |= (hours >= start_hour) | (hours <= end_hour)
            else:
                mask |= hours == int(text)
        return mask

    def _weekday_mask(self, index: pd.Index) -> pd.Series:
        if not self.allowed_weekdays:
            return pd.Series(True, index=index)
        weekdays = pd.Series(pd.DatetimeIndex(index).weekday, index=index)
        return weekdays.isin(self.allowed_weekdays)

    def primary_side(self, X: pd.DataFrame) -> pd.Series:
        components = self._feature_components(X)
        session_ok = self._session_mask(X.index)
        weekday_ok = self._weekday_mask(X.index)
        common = (
            session_ok
            & weekday_ok
            & self._le(components["vpin_zscore"], self.thresholds_.get("max_vpin_zscore"))
            & self._le(components["bar_spread"], self.thresholds_.get("max_bar_spread_proxy"))
            & self._ge(components["theta_to_threshold"], self.thresholds_.get("min_theta_to_threshold"))
            & self._ge(components["bar_density"], self.thresholds_.get("min_bar_density"))
            & self._ge(components["log_return"].abs(), self.thresholds_.get("min_abs_log_return"))
        )
        long_mask = (
            common
            & self._ge(components["signed_dollar"], self.thresholds_.get("long_signed_dollar"))
            & self._ge(components["order_flow"], self.thresholds_.get("long_order_flow"))
            & self._ge(components["buy_share"], self.thresholds_.get("min_buy_notional_share"))
        )
        if self.require_directional_return:
            long_mask &= components["log_return"] >= 0.0

        side = pd.Series(0, index=X.index, dtype=int)
        side.loc[long_mask.fillna(False)] = 1
        if self.allow_short:
            short_mask = (
                common
                & self._le(components["signed_dollar"], self.thresholds_.get("short_signed_dollar"))
                & self._le(components["order_flow"], self.thresholds_.get("short_order_flow"))
                & self._ge(1.0 - components["buy_share"], self.thresholds_.get("min_buy_notional_share"))
            )
            if self.require_directional_return:
                short_mask &= components["log_return"] <= 0.0
            side.loc[short_mask.fillna(False)] = -1
        return side.rename("prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        side = self.primary_side(X)
        proba = pd.DataFrame(0.0, index=X.index, columns=self.classes_)
        proba.loc[side < 0, -1] = 1.0
        proba.loc[side == 0, 0] = 1.0
        proba.loc[side > 0, 1] = 1.0
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.primary_side(X)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "calibration_rows": self.calibration_rows_,
            "allowed_sessions_utc": list(self.allowed_sessions_utc),
            "allowed_weekdays": list(self.allowed_weekdays),
            "allow_short": self.allow_short,
            "require_directional_return": self.require_directional_return,
            "quantiles": {
                "long_signed_dollar_quantile": self.long_signed_dollar_quantile,
                "short_signed_dollar_quantile": self.short_signed_dollar_quantile,
                "long_order_flow_quantile": self.long_order_flow_quantile,
                "short_order_flow_quantile": self.short_order_flow_quantile,
                "min_buy_notional_share_quantile": self.min_buy_notional_share_quantile,
                "min_abs_log_return_quantile": self.min_abs_log_return_quantile,
                "max_vpin_zscore_quantile": self.max_vpin_zscore_quantile,
                "max_bar_spread_proxy_quantile": self.max_bar_spread_proxy_quantile,
                "min_theta_to_threshold_quantile": self.min_theta_to_threshold_quantile,
                "min_bar_density_quantile": self.min_bar_density_quantile,
            },
            "thresholds": self.thresholds_,
        }


class MakerPrecisionPrimarySideModel:
    """
    Maker-style high-frequency primary model without hard-coded time filters.

    The model keeps only liquid, low-adverse-selection flow events. It is designed
    for small profit-taking / larger stop-loss profiles where hit rate matters
    more than per-trade payoff symmetry.
    """

    requires_event_spans = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config if isinstance(config, dict) else {}
        self.allow_short = bool(config.get("allow_short", False))
        self.long_signed_dollar_quantile = self._optional_quantile(
            config,
            "long_signed_dollar_quantile",
            0.55,
        )
        self.short_signed_dollar_quantile = self._optional_quantile(
            config,
            "short_signed_dollar_quantile",
            0.45,
        )
        self.long_order_flow_quantile = self._optional_quantile(
            config,
            "long_order_flow_quantile",
            0.52,
        )
        self.short_order_flow_quantile = self._optional_quantile(
            config,
            "short_order_flow_quantile",
            0.48,
        )
        self.min_buy_notional_share_quantile = self._optional_quantile(
            config,
            "min_buy_notional_share_quantile",
            0.52,
        )
        self.max_vpin_zscore_quantile = self._optional_quantile(
            config,
            "max_vpin_zscore_quantile",
            0.70,
        )
        self.max_bar_spread_proxy_quantile = self._optional_quantile(
            config,
            "max_bar_spread_proxy_quantile",
            0.65,
        )
        self.min_theta_to_threshold_quantile = self._optional_quantile(
            config,
            "min_theta_to_threshold_quantile",
            0.35,
        )
        self.min_bar_density_quantile = self._optional_quantile(
            config,
            "min_bar_density_quantile",
            0.55,
        )
        self.max_abs_log_return_quantile = self._optional_quantile(
            config,
            "max_abs_log_return_quantile",
            0.80,
        )
        self.min_flow_score = float(config.get("min_flow_score", 0.67))
        self.min_liquidity_score = float(config.get("min_liquidity_score", 0.50))
        self.min_absorption_score = float(config.get("min_absorption_score", 0.60))
        self.min_maker_score = float(config.get("min_maker_score", 0.62))
        self.max_adverse_selection_risk = float(config.get("max_adverse_selection_risk", 0.40))
        self.thresholds_: dict[str, float | None] = {}
        self.classes_ = np.array([-1, 0, 1], dtype=int)
        self.feature_names_: list[str] | None = None
        self.calibration_rows_: int = 0

    @staticmethod
    def _optional_quantile(
        config: dict[str, Any],
        key: str,
        default: float | None,
    ) -> float | None:
        value = config.get(key, default)
        if value is None:
            return None
        quantile = float(value)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"primary_model.maker_precision.{key} must be in [0, 1]")
        return quantile

    @staticmethod
    def _first_feature(X: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in X:
                return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(0.0, index=X.index, dtype=float)

    @staticmethod
    def _valid_values(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    def _quantile_value(self, values: pd.Series, quantile: float | None) -> float | None:
        if quantile is None:
            return None
        valid = self._valid_values(values)
        if valid.empty:
            return None
        return float(valid.quantile(quantile))

    def _feature_components(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "signed_dollar": self._first_feature(
                X,
                ("signed_dollar_imbalance", "calendar_signed_notional_robust_z_1d"),
            ),
            "order_flow": self._first_feature(
                X,
                ("order_flow_imbalance", "calendar_order_flow_imbalance_1d"),
            ),
            "buy_share": self._first_feature(X, ("buy_notional_share",)),
            "vpin_zscore": self._first_feature(
                X,
                ("vpin_zscore", "theta_to_threshold_robust_zscore"),
            ),
            "bar_spread": self._first_feature(X, ("bar_spread_proxy",)),
            "log_return": self._first_feature(X, ("log_return_1", "oc_return", "cvdslope_log_ret_1")),
            "theta_to_threshold": self._first_feature(X, ("theta_to_threshold",)),
            "bar_density": self._first_feature(
                X,
                ("calendar_bar_density_per_day_1d", "calendar_bar_density_per_day_3d"),
            ),
        }

    @staticmethod
    def _ge(values: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        return values >= float(threshold)

    @staticmethod
    def _le(values: pd.Series, threshold: float | None) -> pd.Series:
        if threshold is None:
            return pd.Series(True, index=values.index)
        return values <= float(threshold)

    @staticmethod
    def _condition(values: pd.Series) -> pd.Series:
        return values.fillna(False).astype(float)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> MakerPrecisionPrimarySideModel:
        self.feature_names_ = list(X.columns)
        components = self._feature_components(X)
        self.calibration_rows_ = len(X)
        self.thresholds_ = {
            "long_signed_dollar": self._quantile_value(
                components["signed_dollar"],
                self.long_signed_dollar_quantile,
            ),
            "short_signed_dollar": self._quantile_value(
                components["signed_dollar"],
                self.short_signed_dollar_quantile,
            ),
            "long_order_flow": self._quantile_value(
                components["order_flow"],
                self.long_order_flow_quantile,
            ),
            "short_order_flow": self._quantile_value(
                components["order_flow"],
                self.short_order_flow_quantile,
            ),
            "min_buy_notional_share": self._quantile_value(
                components["buy_share"],
                self.min_buy_notional_share_quantile,
            ),
            "max_vpin_zscore": self._quantile_value(
                components["vpin_zscore"],
                self.max_vpin_zscore_quantile,
            ),
            "max_bar_spread_proxy": self._quantile_value(
                components["bar_spread"],
                self.max_bar_spread_proxy_quantile,
            ),
            "min_theta_to_threshold": self._quantile_value(
                components["theta_to_threshold"],
                self.min_theta_to_threshold_quantile,
            ),
            "min_bar_density": self._quantile_value(
                components["bar_density"],
                self.min_bar_density_quantile,
            ),
            "max_abs_log_return": self._quantile_value(
                components["log_return"].abs(),
                self.max_abs_log_return_quantile,
            ),
        }
        return self

    def score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        components = self._feature_components(X)
        long_flow_score = (
            self._condition(self._ge(components["signed_dollar"], self.thresholds_.get("long_signed_dollar")))
            + self._condition(self._ge(components["order_flow"], self.thresholds_.get("long_order_flow")))
            + self._condition(self._ge(components["buy_share"], self.thresholds_.get("min_buy_notional_share")))
        ) / 3.0
        liquidity_score = (
            self._condition(self._le(components["bar_spread"], self.thresholds_.get("max_bar_spread_proxy")))
            + self._condition(self._ge(components["bar_density"], self.thresholds_.get("min_bar_density")))
        ) / 2.0
        low_adverse_score = (
            self._condition(self._le(components["vpin_zscore"], self.thresholds_.get("max_vpin_zscore")))
            + self._condition(self._le(components["log_return"].abs(), self.thresholds_.get("max_abs_log_return")))
            + self._condition(self._ge(components["theta_to_threshold"], self.thresholds_.get("min_theta_to_threshold")))
        ) / 3.0
        adverse_risk = (1.0 - low_adverse_score).clip(0.0, 1.0)
        absorption_score = (
            0.45 * long_flow_score
            + 0.30 * liquidity_score
            + 0.25 * low_adverse_score
        ).clip(0.0, 1.0)
        maker_score = (
            0.40 * long_flow_score
            + 0.35 * liquidity_score
            + 0.25 * low_adverse_score
        ).clip(0.0, 1.0)
        return pd.DataFrame(
            {
                "primary_maker_score": maker_score,
                "primary_absorption_score": absorption_score,
                "primary_adverse_selection_risk": adverse_risk,
                "primary_flow_score": long_flow_score,
                "primary_liquidity_score": liquidity_score,
            },
            index=X.index,
        )

    def primary_side(self, X: pd.DataFrame) -> pd.Series:
        scores = self.score_frame(X)
        long_mask = (
            (scores["primary_flow_score"] >= self.min_flow_score)
            & (scores["primary_liquidity_score"] >= self.min_liquidity_score)
            & (scores["primary_absorption_score"] >= self.min_absorption_score)
            & (scores["primary_maker_score"] >= self.min_maker_score)
            & (scores["primary_adverse_selection_risk"] <= self.max_adverse_selection_risk)
        )
        side = pd.Series(0, index=X.index, dtype=int)
        side.loc[long_mask.fillna(False)] = 1
        if self.allow_short:
            components = self._feature_components(X)
            short_flow_score = (
                self._condition(self._le(components["signed_dollar"], self.thresholds_.get("short_signed_dollar")))
                + self._condition(self._le(components["order_flow"], self.thresholds_.get("short_order_flow")))
                + self._condition(self._ge(1.0 - components["buy_share"], self.thresholds_.get("min_buy_notional_share")))
            ) / 3.0
            short_mask = (
                (short_flow_score >= self.min_flow_score)
                & (scores["primary_liquidity_score"] >= self.min_liquidity_score)
                & (scores["primary_adverse_selection_risk"] <= self.max_adverse_selection_risk)
            )
            side.loc[short_mask.fillna(False)] = -1
        return side.rename("prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        side = self.primary_side(X)
        proba = pd.DataFrame(0.0, index=X.index, columns=self.classes_)
        proba.loc[side < 0, -1] = 1.0
        proba.loc[side == 0, 0] = 1.0
        proba.loc[side > 0, 1] = 1.0
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.primary_side(X)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "calibration_rows": self.calibration_rows_,
            "allow_short": self.allow_short,
            "quantiles": {
                "long_signed_dollar_quantile": self.long_signed_dollar_quantile,
                "short_signed_dollar_quantile": self.short_signed_dollar_quantile,
                "long_order_flow_quantile": self.long_order_flow_quantile,
                "short_order_flow_quantile": self.short_order_flow_quantile,
                "min_buy_notional_share_quantile": self.min_buy_notional_share_quantile,
                "max_vpin_zscore_quantile": self.max_vpin_zscore_quantile,
                "max_bar_spread_proxy_quantile": self.max_bar_spread_proxy_quantile,
                "min_theta_to_threshold_quantile": self.min_theta_to_threshold_quantile,
                "min_bar_density_quantile": self.min_bar_density_quantile,
                "max_abs_log_return_quantile": self.max_abs_log_return_quantile,
            },
            "score_thresholds": {
                "min_flow_score": self.min_flow_score,
                "min_liquidity_score": self.min_liquidity_score,
                "min_absorption_score": self.min_absorption_score,
                "min_maker_score": self.min_maker_score,
                "max_adverse_selection_risk": self.max_adverse_selection_risk,
            },
            "thresholds": self.thresholds_,
        }


class CalibratedBinaryProbabilityModel:
    """
    Apply a binary probability calibrator to an already fitted classifier.
    """

    classes_ = np.array([0, 1], dtype=int)

    def __init__(
        self,
        model: Any,
        *,
        calibration_probability: pd.Series,
        calibration_truth: pd.Series,
        method: str,
    ) -> None:
        self.model = model
        self.calibration_probability = calibration_probability.astype(float)
        self.calibration_truth = calibration_truth.astype(int)
        self.method = method

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = self.model.predict_proba(X)
        if 1 not in raw.columns:
            calibrated = pd.Series(0.0, index=raw.index, name="calibrated_probability")
        else:
            calibrated = calibrate_binary_probabilities(
                self.calibration_probability,
                self.calibration_truth,
                target_probability=raw[1],
                method=self.method,
            ).reindex(raw.index).fillna(0.0)
        calibrated = calibrated.clip(0.0, 1.0)
        return pd.DataFrame(
            {
                0: 1.0 - calibrated,
                1: calibrated,
            },
            index=raw.index,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba[1] >= 0.5).astype(int).rename("prediction")


class ProfitWeightedSequentialBootstrapClassifier:
    """
    Sequential-bootstrap forest with return-aware sample weights for meta labels.

    Base AFML sample weights already combine uniqueness, return attribution, and
    time decay. This wrapper keeps those weights, then increases emphasis on
    high-impact winners and losers using the realized side-adjusted label return.
    """

    requires_dataset = True
    requires_event_spans = False

    def __init__(
        self,
        *,
        n_estimators: int,
        max_samples: float | str | None,
        max_features: float | str | None,
        max_depth: int | None,
        min_samples_leaf: int,
        min_weight_fraction_leaf: float,
        class_weight: str | dict[int, float] | None,
        random_state: int | None,
        n_jobs: int | None,
        verbose: bool,
        progress_interval: int,
        progress_label: str | None,
        return_floor: float = 0.0,
        return_weight_scale: float = 0.001,
        max_return_weight: float = 6.0,
        positive_return_multiplier: float = 1.0,
        negative_return_multiplier: float = 1.25,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.progress_interval = progress_interval
        self.progress_label = progress_label
        self.return_floor = return_floor
        self.return_weight_scale = return_weight_scale
        self.max_return_weight = max_return_weight
        self.positive_return_multiplier = positive_return_multiplier
        self.negative_return_multiplier = negative_return_multiplier

    def _base_model(self) -> SequentialBootstrapBaggingClassifier:
        return SequentialBootstrapBaggingClassifier(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            progress_interval=self.progress_interval,
            progress_label=self.progress_label,
        )

    def _profit_adjusted_weights(
        self,
        dataset: Any,
        train_index: pd.DatetimeIndex,
    ) -> pd.Series:
        base_weight = dataset.sample_weight.loc[train_index].fillna(0.0).clip(lower=0.0).astype(float)
        if "ret" not in dataset.labels.columns:
            return base_weight
        label_return = pd.to_numeric(dataset.labels.loc[train_index, "ret"], errors="coerce").fillna(0.0)
        net_return = label_return - float(self.return_floor)
        scale = max(1e-12, float(self.return_weight_scale))
        magnitude = (net_return.abs() / scale).clip(lower=0.0, upper=float(self.max_return_weight))
        side_multiplier = pd.Series(float(self.positive_return_multiplier), index=train_index)
        side_multiplier.loc[net_return < 0.0] = float(self.negative_return_multiplier)
        adjusted = base_weight * (1.0 + side_multiplier * magnitude)
        if adjusted.sum() <= 0.0:
            return base_weight
        self.weight_diagnostics_ = {
            "return_floor": float(self.return_floor),
            "return_weight_scale": float(self.return_weight_scale),
            "max_return_weight": float(self.max_return_weight),
            "positive_return_multiplier": float(self.positive_return_multiplier),
            "negative_return_multiplier": float(self.negative_return_multiplier),
            "base_weight_mean": float(base_weight.mean()) if len(base_weight) else 0.0,
            "adjusted_weight_mean": float(adjusted.mean()) if len(adjusted) else 0.0,
            "adjusted_weight_p95": float(adjusted.quantile(0.95)) if len(adjusted) else 0.0,
            "positive_events": int((net_return > 0.0).sum()),
            "negative_events": int((net_return < 0.0).sum()),
        }
        return adjusted.rename("sample_weight")

    def _copy_fitted_attributes(self) -> None:
        self.classes_ = self.model_.classes_
        self.feature_names_ = self.model_.feature_names_
        self.estimators_ = self.model_.estimators_
        self.estimator_classes_ = self.model_.estimator_classes_
        self.bootstrap_indices_ = self.model_.bootstrap_indices_

    def fit_dataset(
        self,
        dataset: Any,
        train_index: pd.DatetimeIndex,
    ) -> ProfitWeightedSequentialBootstrapClassifier:
        train_index = pd.DatetimeIndex(train_index).intersection(dataset.X.index)
        adjusted_weight = self._profit_adjusted_weights(dataset, train_index)
        self.model_ = self._base_model()
        self.model_.fit(
            dataset.X.loc[train_index],
            dataset.y.loc[train_index],
            sample_weight=adjusted_weight,
            t1=dataset.events.loc[train_index, "t1"],
            bar_index=dataset.close.index,
        )
        self._copy_fitted_attributes()
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        t1: pd.Series | None = None,
        bar_index: pd.DatetimeIndex | None = None,
    ) -> ProfitWeightedSequentialBootstrapClassifier:
        self.weight_diagnostics_ = {"status": "fallback_fit_without_label_returns"}
        self.model_ = self._base_model()
        self.model_.fit(X, y, sample_weight=sample_weight, t1=t1, bar_index=bar_index)
        self._copy_fitted_attributes()
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model_.predict(X)

    def diagnostics(self) -> dict[str, Any]:
        return getattr(self, "weight_diagnostics_", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the real-data CVD-slope AFML rule-primary/meta model and optionally "
            "run the real-data Nautilus backtest."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to AFML data config JSON.")
    parser.add_argument(
        "--strategy-config",
        default=None,
        help=f"Override strategy JSON path (default: {DEFAULT_REAL_STRATEGY_CONFIG}).",
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--input-csv", default=None, help="Only valid with one --symbols value.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start", default=None, help="Optional data start timestamp.")
    parser.add_argument("--end", default=None, help="Optional data end timestamp.")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None, help="Optional test end timestamp.")
    parser.add_argument("--test-days", type=float, default=None)
    parser.add_argument(
        "--train-days",
        type=float,
        default=None,
        help="Training lookback before the embargo. Use 0 for all available history.",
    )
    parser.add_argument("--price-column", default=None)
    parser.add_argument("--vertical-barrier-bars", type=int, default=None)
    parser.add_argument("--volatility-ewma-span", type=int, default=None)
    parser.add_argument("--volatility-lookback-days", type=int, default=None)
    parser.add_argument("--volatility-floor-quantile", type=float, default=None)
    parser.add_argument("--cusum-window", type=int, default=None)
    parser.add_argument("--cusum-threshold-mult", type=float, default=None)
    parser.add_argument("--cusum-floor-quantile", type=float, default=None)
    parser.add_argument("--min-ret", type=float, default=None)
    parser.add_argument("--oldest-weight", type=float, default=None)
    parser.add_argument("--rare-label-min-pct", type=float, default=None)
    parser.add_argument("--meta-label-min-ret", type=float, default=None)
    parser.add_argument("--meta-probability-threshold", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-features", default=None)
    parser.add_argument("--max-samples", default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)
    parser.add_argument("--min-weight-fraction-leaf", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--class-weight", default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--embargo-bars", type=int, default=None)
    parser.add_argument("--pca-components", type=float, default=None)
    parser.add_argument("--pca-random-state", type=int, default=None)
    parser.add_argument("--feature-fracdiff-d", type=float, default=None)
    parser.add_argument("--feature-fracdiff-threshold", type=float, default=None)
    parser.add_argument("--microstructure-window", type=int, default=None)
    parser.add_argument("--information-window", type=int, default=None)
    parser.add_argument(
        "--calendar-windows",
        default=None,
        help="Comma-separated calendar windows for regime features, or 'none' to disable.",
    )
    parser.add_argument("--calendar-min-periods", type=int, default=None)
    parser.add_argument("--tail-quantile", type=float, default=None)
    parser.add_argument("--robust-z-clip", type=float, default=None)
    parser.add_argument("--bet-size-step", type=float, default=None)
    parser.add_argument("--bet-size-min-abs", type=float, default=None)
    parser.add_argument("--bet-size-max-abs", type=float, default=None)
    parser.add_argument("--trade-notional", type=Decimal, default=None)
    parser.add_argument("--starting-balance", type=Decimal, default=None)
    parser.add_argument("--default-leverage", type=Decimal, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--min-position-change", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument(
        "--allow-rejected-backtest",
        action="store_true",
        help="Run the real Nautilus backtest even when AFML acceptance gates reject the signals.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def parse_calendar_windows(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null", "off", "false"}:
            return None
        windows = tuple(item.strip() for item in text.split(",") if item.strip())
    elif isinstance(value, (list, tuple)):
        windows = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise TypeError("calendar_windows must be a string, list, tuple, or null")
    return windows or None


def load_real_strategy_config(config: dict[str, Any], cli_path: str | None = None) -> dict[str, Any]:
    strategy_path_value = cli_path or DEFAULT_REAL_STRATEGY_CONFIG
    strategy_path = resolve_repo_path(strategy_path_value)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Real strategy config does not exist: {strategy_path}")

    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Real strategy config root must be a JSON object: {strategy_path}")
    payload["_strategy_config_path"] = str(strategy_path)
    return payload


def apply_json_defaults(
    args: argparse.Namespace,
    strategy_config: dict[str, Any],
) -> argparse.Namespace:
    workflow = section(strategy_config, "real_data_workflow")
    labeling = section(strategy_config, "real_data_labeling")
    features = section(strategy_config, "feature_engineering")
    meta_config = section(strategy_config, "meta_model")
    backtest = section(strategy_config, "backtest")

    args.output_dir = first_not_none(args.output_dir, workflow.get("output_dir"), str(DEFAULT_OUTPUT_DIR))
    args.input_csv = first_not_none(args.input_csv, workflow.get("input_csv"))
    args.start = first_not_none(args.start, workflow.get("data_start"))
    args.end = first_not_none(args.end, workflow.get("data_end"))
    args.train_start = first_not_none(args.train_start, workflow.get("train_start"))
    args.train_end = first_not_none(args.train_end, workflow.get("train_end"))
    args.test_start = first_not_none(args.test_start, workflow.get("test_start"))
    args.test_end = first_not_none(args.test_end, workflow.get("test_end"))
    args.test_days = float(first_not_none(args.test_days, workflow.get("test_days"), 2.0))
    args.train_days = float(first_not_none(args.train_days, workflow.get("train_days"), 90.0))
    args.run_backtest = bool(workflow.get("run_backtest", True)) and not args.no_backtest
    args.no_progress = args.no_progress or not bool(workflow.get("show_progress", True))

    args.volatility_ewma_span = int(
        first_not_none(
            args.volatility_ewma_span,
            labeling.get("volatility_ewma_span"),
            100,
        ),
    )
    args.volatility_lookback_days = int(
        first_not_none(
            args.volatility_lookback_days,
            labeling.get("volatility_lookback_days"),
            1,
        ),
    )
    args.volatility_floor_quantile = float(
        first_not_none(args.volatility_floor_quantile, labeling.get("volatility_floor_quantile"), 0.05),
    )
    args.cusum_window = int(first_not_none(args.cusum_window, labeling.get("cusum_window"), 100))
    args.cusum_threshold_mult = float(
        first_not_none(args.cusum_threshold_mult, labeling.get("cusum_threshold_mult"), 3.0),
    )
    args.cusum_floor_quantile = float(
        first_not_none(args.cusum_floor_quantile, labeling.get("cusum_floor_quantile"), 0.10),
    )
    args.min_ret = float(first_not_none(args.min_ret, labeling.get("min_ret"), 0.0))
    args.oldest_weight = float(first_not_none(args.oldest_weight, labeling.get("oldest_weight"), 1.0))
    args.rare_label_min_pct = first_not_none(args.rare_label_min_pct, labeling.get("rare_label_min_pct"))
    args.meta_label_min_ret = first_not_none(args.meta_label_min_ret, labeling.get("meta_label_min_ret"))

    args.feature_fracdiff_d = float(
        first_not_none(args.feature_fracdiff_d, features.get("fracdiff_d"), 0.4),
    )
    args.feature_fracdiff_threshold = float(
        first_not_none(args.feature_fracdiff_threshold, features.get("fracdiff_threshold"), 0.01),
    )
    args.microstructure_window = int(
        first_not_none(args.microstructure_window, features.get("microstructure_window"), 50),
    )
    args.information_window = int(
        first_not_none(args.information_window, features.get("information_window"), 64),
    )
    calendar_windows_value = args.calendar_windows
    if calendar_windows_value is None:
        calendar_windows_value = features.get("calendar_windows", DEFAULT_CALENDAR_WINDOWS)
    args.calendar_windows = parse_calendar_windows(calendar_windows_value)
    args.calendar_min_periods = int(
        first_not_none(args.calendar_min_periods, features.get("calendar_min_periods"), 8),
    )
    args.tail_quantile = float(
        first_not_none(args.tail_quantile, features.get("tail_quantile"), 0.99),
    )
    robust_z_clip_value = args.robust_z_clip
    if robust_z_clip_value is None:
        robust_z_clip_value = features.get("robust_z_clip", 5.0)
    args.robust_z_clip = None if robust_z_clip_value is None else float(robust_z_clip_value)

    args.meta_probability_threshold = first_not_none(
        args.meta_probability_threshold,
        meta_config.get("class_probability_threshold"),
    )
    args.min_weight_fraction_leaf = float(
        first_not_none(args.min_weight_fraction_leaf, meta_config.get("min_weight_fraction_leaf"), 0.05),
    )
    args.n_jobs = int(first_not_none(args.n_jobs, meta_config.get("n_jobs"), 1))
    args.progress_interval = int(
        first_not_none(args.progress_interval, meta_config.get("progress_interval"), 10),
    )
    args.embargo_bars = int(first_not_none(args.embargo_bars, meta_config.get("embargo_bars"), 80))
    args.pca_components = first_not_none(args.pca_components, meta_config.get("pca_components"))
    args.pca_random_state = int(first_not_none(args.pca_random_state, meta_config.get("pca_random_state"), 42))
    bet_size_step = args.bet_size_step if args.bet_size_step is not None else meta_config.get("bet_size_step")
    args.bet_size_step = None if bet_size_step is None else float(bet_size_step)
    args.bet_size_min_abs = float(
        first_not_none(args.bet_size_min_abs, meta_config.get("bet_size_min_abs"), 0.0),
    )
    args.bet_size_max_abs = float(
        first_not_none(args.bet_size_max_abs, meta_config.get("bet_size_max_abs"), 1.0),
    )

    args.trade_notional = Decimal(
        str(first_not_none(args.trade_notional, backtest.get("trade_notional"), "10000")),
    )
    args.starting_balance = Decimal(
        str(first_not_none(args.starting_balance, backtest.get("starting_balance"), "1000000")),
    )
    args.default_leverage = Decimal(
        str(first_not_none(args.default_leverage, backtest.get("default_leverage"), "20")),
    )
    args.confidence_threshold = float(
        first_not_none(args.confidence_threshold, backtest.get("confidence_threshold"), 0.0),
    )
    args.min_position_change = first_not_none(
        args.min_position_change,
        backtest.get("min_position_change"),
        "size_increment",
    )
    args.log_level = str(first_not_none(args.log_level, backtest.get("log_level"), "ERROR"))
    return args


def parse_time_bound(value: Any, *, is_end: bool = False) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if is_end and isinstance(value, str) and len(value) == 10:
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def class_counts(values: pd.Series) -> dict[str, int]:
    counts = values.astype(int).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def value_counts_dict(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    counts = values.fillna("unknown").astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def signal_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "signal" not in frame:
        return {}
    return class_counts(frame["signal"])


def diagnostic_backtest_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    validation = section(strategy_config, "validation")
    config = validation.get("diagnostic_backtest", {})
    if not isinstance(config, dict):
        config = {}
    return {
        "enabled_on_rejection": bool(config.get("enabled_on_rejection", True)),
    }


def diagnostic_backtest_enabled_on_rejection(strategy_config: dict[str, Any]) -> bool:
    return bool(diagnostic_backtest_config(strategy_config).get("enabled_on_rejection", True))


def diagnostic_active_signal_count(summary: dict[str, Any]) -> int:
    signals = summary.get("signals", {})
    if not isinstance(signals, dict):
        return 0
    try:
        return int(signals.get("diagnostic_active_signals", 0))
    except (TypeError, ValueError):
        return 0


def summary_approved_for_backtest(summary: dict[str, Any]) -> bool:
    gate = summary.get("afml_acceptance_gate", {})
    if not isinstance(gate, dict):
        return True
    return bool(gate.get("approved_for_backtest", True))


def sample_weight_summary(weights: pd.Series) -> dict[str, Any]:
    weights = weights.dropna().astype(float)
    if weights.empty:
        return {"count": 0}
    return {
        "count": len(weights),
        "mean": float(weights.mean()),
        "std": float(weights.std(ddof=1)) if len(weights) > 1 else 0.0,
        "min": float(weights.min()),
        "max": float(weights.max()),
        "q25": float(weights.quantile(0.25)),
        "q50": float(weights.quantile(0.50)),
        "q75": float(weights.quantile(0.75)),
    }


def dataset_event_span_arrays(dataset: AfmlDataset) -> tuple[np.ndarray, np.ndarray]:
    index = pd.DatetimeIndex(dataset.y.index)
    bar_index = pd.DatetimeIndex(dataset.close.index).sort_values()
    starts = bar_index.searchsorted(index, side="left").astype(np.int64)
    event_end = pd.to_datetime(dataset.events.loc[index, "t1"], utc=True)
    ends = bar_index.searchsorted(pd.DatetimeIndex(event_end), side="right").astype(np.int64)
    return starts, ends


def dataset_validation_diagnostics(
    dataset: AfmlDataset,
    test_index: pd.DatetimeIndex,
    *,
    strategy_config: dict[str, Any],
    n_trials: int,
    round_trip_cost: float = 0.0,
) -> dict[str, Any]:
    validation_config = section(strategy_config, "validation")
    returns = dataset.labels.loc[test_index, "ret"] if "ret" in dataset.labels else pd.Series(dtype=float)
    returns = returns - float(round_trip_cost)
    starts, ends = dataset_event_span_arrays(dataset)
    diagnostics = afml_validation_diagnostics(
        returns,
        validation_config=validation_config,
        n_trials=n_trials,
        event_starts=starts,
        event_ends=ends,
    )
    diagnostics["return_basis"] = "cost_adjusted_label_ret"
    diagnostics["round_trip_cost"] = float(round_trip_cost)
    return diagnostics


def feature_importance_completeness(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(diagnostics.get("enabled", False)),
        "pca_importance": bool(diagnostics.get("pca_importance_csv")),
        "mda_pruning": bool(diagnostics.get("mda_pruning_csv")),
        "final_mdi": bool(diagnostics.get("final_mdi_csv")),
        "final_mda": bool(diagnostics.get("final_mda_csv")),
        "final_sfi": bool(diagnostics.get("final_sfi_csv")),
    }


def classification_metrics(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    positive_label: int = 1,
) -> dict[str, Any]:
    truth = truth.astype(int)
    prediction = prediction.reindex(truth.index).fillna(0).astype(int)
    labels = sorted(set(truth.unique()).union(set(prediction.unique())))
    total = len(truth)
    accuracy = float((truth == prediction).mean()) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    for label in labels:
        true_positive = int(((truth == label) & (prediction == label)).sum())
        false_positive = int(((truth != label) & (prediction == label)).sum())
        false_negative = int(((truth == label) & (prediction != label)).sum())
        support = int((truth == label).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(int(label))] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        supports.append(support)

    support_array = np.asarray(supports, dtype=float)
    support_sum = float(support_array.sum())
    weighted_f1 = float(np.average(f1s, weights=support_array)) if support_sum > 0.0 else 0.0
    weighted_precision = (
        float(np.average(precisions, weights=support_array)) if support_sum > 0.0 else 0.0
    )
    weighted_recall = float(np.average(recalls, weights=support_array)) if support_sum > 0.0 else 0.0
    positive = per_class.get(str(int(positive_label)), {})
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "positive_label": int(positive_label),
        "positive_precision": positive.get("precision", 0.0),
        "positive_recall": positive.get("recall", 0.0),
        "positive_f1": positive.get("f1", 0.0),
        "per_class": per_class,
    }


def threshold_binary_prediction(
    model: Any,
    features: pd.DataFrame,
    *,
    threshold: float,
    upper_threshold: float | None = None,
) -> pd.Series:
    proba = model.predict_proba(features)
    if 1 not in proba.columns:
        return pd.Series(0, index=features.index, name="prediction")
    prediction = proba[1] >= threshold
    if upper_threshold is not None:
        prediction &= proba[1] <= float(upper_threshold)
    return prediction.astype(int).rename("prediction")


def primary_side_frame_from_rule(
    model: Any,
    features: pd.DataFrame,
    *,
    volatility: pd.Series | None = None,
    train_start: pd.Timestamp | None = None,
    train_end: pd.Timestamp | None = None,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.DataFrame:
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    side = model.primary_side(features).astype(int)
    frame = pd.DataFrame(
        {
            "primary_side": side,
            "primary_prediction": side,
        },
        index=features.index,
    )
    if hasattr(model, "score_frame"):
        score_frame = model.score_frame(features).reindex(frame.index)
        score_columns = [column for column in score_frame.columns if column not in frame.columns]
        if score_columns:
            frame = frame.join(score_frame[score_columns], how="left")
    if volatility is not None:
        frame["volatility_state"] = volatility_state_by_train_quantile(
            volatility,
            frame.index,
            train_start=train_start,
            train_end=train_end,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
        )
    return frame


def volatility_state_by_train_quantile(
    volatility: pd.Series,
    index: pd.Index,
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp | None,
    low_quantile: float,
    high_quantile: float,
) -> pd.Series:
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("volatility state quantiles must satisfy 0 < low < high < 1")
    aligned = pd.to_numeric(volatility, errors="coerce").reindex(pd.DatetimeIndex(index)).ffill()
    train_vol = pd.to_numeric(volatility, errors="coerce").dropna()
    if train_start is not None:
        train_vol = train_vol.loc[train_vol.index >= train_start]
    if train_end is not None:
        train_vol = train_vol.loc[train_vol.index <= train_end]
    if train_vol.empty:
        return pd.Series("unknown", index=pd.DatetimeIndex(index), dtype=object)

    low = float(train_vol.quantile(low_quantile))
    high = float(train_vol.quantile(high_quantile))
    state = pd.Series("mid_vol", index=pd.DatetimeIndex(index), dtype=object)
    state.loc[aligned <= low] = "low_vol"
    state.loc[aligned >= high] = "high_vol"
    state.loc[aligned.isna()] = "unknown"
    return state


def abstain_primary_side_frame(index: pd.Index, *, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "primary_side": 0,
            "primary_prediction": 0,
            "primary_abstain_reason": reason,
        },
        index=pd.DatetimeIndex(index),
    )


def add_event_session_utc_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    hour = pd.Series(pd.DatetimeIndex(out.index).hour, index=out.index)
    session = pd.Series("16_23", index=out.index, dtype=object)
    session.loc[hour < 8] = "00_07"
    session.loc[(hour >= 8) & (hour < 16)] = "08_15"
    four_hour_session = pd.Series(
        [f"{int(value // 4) * 4:02d}_{int(value // 4) * 4 + 3:02d}" for value in hour],
        index=out.index,
        dtype=object,
    )
    for column, values in pd.get_dummies(session, prefix="event_session_utc", dtype=float).items():
        if column not in out:
            out[column] = values
    for column, values in pd.get_dummies(four_hour_session, prefix="event_session_utc_4h", dtype=float).items():
        if column not in out:
            out[column] = values
    return out


def add_threshold_filter_features(
    features: pd.DataFrame,
    *,
    primary_side: pd.Series | None = None,
) -> pd.DataFrame:
    out = add_event_session_utc_features(features)
    if primary_side is None:
        return out
    side = pd.to_numeric(primary_side.reindex(out.index), errors="coerce").fillna(0).astype(int)
    out["primary_side_long"] = (side > 0).astype(float)
    out["primary_side_short"] = (side < 0).astype(float)
    return out


def excluded_feature_patterns(strategy_config: dict[str, Any]) -> tuple[str, ...]:
    config = section(strategy_config, "feature_engineering")
    raw = config.get("excluded_feature_patterns", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise TypeError("feature_engineering.excluded_feature_patterns must be a list or string")
    return tuple(str(pattern).strip().lower() for pattern in raw if str(pattern).strip())


def apply_feature_pattern_exclusions(
    features: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> pd.DataFrame:
    patterns = excluded_feature_patterns(strategy_config)
    if not patterns or features.empty:
        return features
    keep_columns = [
        column
        for column in features.columns
        if not any(pattern in str(column).lower() for pattern in patterns)
    ]
    return features.loc[:, keep_columns]


def additional_primary_meta_context(primary_frame: pd.DataFrame) -> pd.DataFrame:
    prefixes = (
        "primary_maker_",
        "primary_absorption_",
        "primary_adverse_",
        "primary_flow_",
        "primary_liquidity_",
    )
    columns = [
        column
        for column in primary_frame.columns
        if str(column).startswith(prefixes)
    ]
    if not columns:
        return pd.DataFrame(index=primary_frame.index)
    return (
        primary_frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def make_rule_meta_features(
    features: pd.DataFrame,
    primary_frame: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> pd.DataFrame:
    aligned = features.loc[features.index.intersection(primary_frame.index)].copy()
    primary_frame = primary_frame.reindex(aligned.index)
    meta_features = make_primary_meta_features(aligned, primary_frame)
    extra_context = additional_primary_meta_context(primary_frame)
    if not extra_context.empty:
        extra_context = extra_context.reindex(meta_features.index)
        missing_columns = [column for column in extra_context.columns if column not in meta_features.columns]
        if missing_columns:
            meta_features = meta_features.join(extra_context[missing_columns], how="left")
    return apply_feature_pattern_exclusions(meta_features, strategy_config)


def primary_model_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    return section(strategy_config, "primary_model")


def primary_rule_mode(strategy_config: dict[str, Any]) -> str:
    config = primary_model_config(strategy_config)
    return str(config.get("mode", PRIMARY_CVDSLOPE_RULE_NAME))


def make_primary_rule_side_model(
    *,
    strategy_config: dict[str, Any],
    event_config: dict[str, Any],
) -> tuple[Any, str, dict[str, Any]]:
    config = primary_model_config(strategy_config)
    mode = primary_rule_mode(strategy_config)
    if primary_model_enabled(strategy_config) and mode in {PRIMARY_MOE_MODEL_NAME, "regime_switching_dual_policy"}:
        mode = str(config.get("rule_mode", PRIMARY_CVDSLOPE_RULE_NAME))
    if mode == PRIMARY_CVDSLOPE_RULE_NAME:
        return CvdSlopeRuleSideModel(event_config), mode, {"event_definition": event_config}
    if mode == PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME:
        rule_config = config.get("order_flow_shock", {})
        if not isinstance(rule_config, dict):
            raise TypeError("primary_model.order_flow_shock must be a JSON object")
        model = OrderFlowShockRuleSideModel(rule_config)
        return model, mode, {"order_flow_shock": model.diagnostics()}
    if mode == PRIMARY_PROFIT_SESSION_FLOW_RULE_NAME:
        rule_config = config.get("profit_session_flow", {})
        if not isinstance(rule_config, dict):
            raise TypeError("primary_model.profit_session_flow must be a JSON object")
        model = ProfitSessionFlowRuleSideModel(rule_config)
        return model, mode, {"profit_session_flow": model.diagnostics()}
    if mode == PRIMARY_MAKER_PRECISION_RULE_NAME:
        rule_config = config.get("maker_precision", {})
        if not isinstance(rule_config, dict):
            raise TypeError("primary_model.maker_precision must be a JSON object")
        model = MakerPrecisionPrimarySideModel(rule_config)
        return model, mode, {"maker_precision": model.diagnostics()}
    raise ValueError(
        f"Unsupported primary rule mode: {mode}. Use {PRIMARY_CVDSLOPE_RULE_NAME!r}, "
        f"{PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME!r}, {PRIMARY_PROFIT_SESSION_FLOW_RULE_NAME!r}, "
        f"or {PRIMARY_MAKER_PRECISION_RULE_NAME!r}. "
        "For enabled MoE/regime primary models set primary_model.rule_mode.",
    )


def primary_rule_side_model_diagnostics(
    model: Any,
    *,
    mode: str,
    event_config: dict[str, Any],
) -> dict[str, Any]:
    if mode == PRIMARY_CVDSLOPE_RULE_NAME:
        return {"event_definition": event_config}
    if mode == PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME and hasattr(model, "diagnostics"):
        return {"order_flow_shock": model.diagnostics()}
    if mode == PRIMARY_PROFIT_SESSION_FLOW_RULE_NAME and hasattr(model, "diagnostics"):
        return {"profit_session_flow": model.diagnostics()}
    if mode == PRIMARY_MAKER_PRECISION_RULE_NAME and hasattr(model, "diagnostics"):
        return {"maker_precision": model.diagnostics()}
    return {}


def primary_rule_config_for_mode(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    mode = primary_rule_mode(strategy_config)
    if mode == PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME:
        rule_config = config.get("order_flow_shock", {})
    elif mode == PRIMARY_PROFIT_SESSION_FLOW_RULE_NAME:
        rule_config = config.get("profit_session_flow", {})
    elif mode == PRIMARY_MAKER_PRECISION_RULE_NAME:
        rule_config = config.get("maker_precision", {})
    else:
        rule_config = {}
    return rule_config if isinstance(rule_config, dict) else {}


def primary_rule_calibration_features(
    *,
    features: pd.DataFrame,
    volatility: pd.Series,
    window: TrainTestWindow,
    volatility_low_quantile: float,
    volatility_high_quantile: float,
    strategy_config: dict[str, Any],
) -> pd.DataFrame:
    fit_features = features.replace([np.inf, -np.inf], np.nan)
    fit_features = fit_features.loc[
        (fit_features.index >= window.train_start) & (fit_features.index <= window.train_end)
    ]
    rule_config = primary_rule_config_for_mode(strategy_config)
    calibration_lookback_days = rule_config.get("calibration_lookback_days")
    if calibration_lookback_days is not None:
        lookback_start = window.train_end - pd.Timedelta(days=float(calibration_lookback_days))
        fit_features = fit_features.loc[fit_features.index >= lookback_start]
    fit_state_filter = primary_candidate_filter_config(strategy_config).get("volatility_states", [])
    if isinstance(fit_state_filter, str):
        fit_state_filter = [fit_state_filter]
    fit_state_filter = [str(state) for state in fit_state_filter if str(state)]
    if not fit_state_filter or fit_features.empty:
        return fit_features

    fit_volatility_state = volatility_state_by_train_quantile(
        volatility,
        fit_features.index,
        train_start=window.train_start,
        train_end=window.train_end,
        low_quantile=volatility_low_quantile,
        high_quantile=volatility_high_quantile,
    )
    filtered_fit_features = fit_features.loc[fit_volatility_state.isin(fit_state_filter)]
    return filtered_fit_features if not filtered_fit_features.empty else fit_features


def _order_flow_walk_forward_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    rule_config = primary_model_config(strategy_config).get("order_flow_shock", {})
    if not isinstance(rule_config, dict):
        return {}
    walk_forward = rule_config.get("walk_forward", {})
    return walk_forward if isinstance(walk_forward, dict) else {}


def _order_flow_base_config_without_controls(strategy_config: dict[str, Any]) -> dict[str, Any]:
    rule_config = primary_model_config(strategy_config).get("order_flow_shock", {})
    rule_config = rule_config if isinstance(rule_config, dict) else {}
    return {
        key: value
        for key, value in rule_config.items()
        if key not in {"search", "walk_forward", "calibration_lookback_days"}
    }


def _candidate_filter_volatility_states(strategy_config: dict[str, Any]) -> list[str]:
    raw_states = primary_candidate_filter_config(strategy_config).get("volatility_states", [])
    if isinstance(raw_states, str):
        raw_states = [raw_states]
    return [str(state) for state in raw_states if str(state)]


def _order_flow_calibration_slice(
    *,
    features: pd.DataFrame,
    volatility_state: pd.Series,
    calibration_end: pd.Timestamp,
    lookback_days: float | None,
    allowed_states: list[str],
) -> pd.DataFrame:
    calibration = features.loc[features.index < calibration_end]
    if lookback_days is not None:
        calibration = calibration.loc[calibration.index >= calibration_end - pd.Timedelta(days=float(lookback_days))]
    if allowed_states and not calibration.empty:
        states = volatility_state.reindex(calibration.index)
        filtered = calibration.loc[states.isin(allowed_states)]
        if not filtered.empty:
            calibration = filtered
    return calibration


def _month_period_starts(index: pd.DatetimeIndex) -> pd.Series:
    normalized = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    periods = normalized.to_period("M")
    starts = periods.to_timestamp(how="start").tz_localize("UTC")
    return pd.Series(starts, index=index)


def primary_side_frame_from_walk_forward_order_flow(
    *,
    features: pd.DataFrame,
    volatility: pd.Series,
    window: TrainTestWindow,
    strategy_config: dict[str, Any],
    volatility_low_quantile: float,
    volatility_high_quantile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean_features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_features.empty:
        return abstain_primary_side_frame(features.index, reason="empty_walk_forward_features"), {
            "enabled": True,
            "status": "empty_features",
        }

    walk_config = _order_flow_walk_forward_config(strategy_config)
    lookback_days = first_not_none(
        walk_config.get("calibration_lookback_days"),
        primary_model_config(strategy_config).get("order_flow_shock", {}).get("calibration_lookback_days", None),
    )
    min_calibration_rows = int(walk_config.get("min_calibration_rows", 200))
    allowed_states = _candidate_filter_volatility_states(strategy_config)
    volatility_state = volatility_state_by_train_quantile(
        volatility,
        clean_features.index,
        train_start=window.train_start,
        train_end=window.train_end,
        low_quantile=volatility_low_quantile,
        high_quantile=volatility_high_quantile,
    )
    period_start = _month_period_starts(pd.DatetimeIndex(clean_features.index))
    side = pd.Series(0, index=clean_features.index, dtype=int)
    diagnostics_rows: list[dict[str, Any]] = []
    base_config = _order_flow_base_config_without_controls(strategy_config)

    for start, group_index in period_start.groupby(period_start).groups.items():
        group_index = pd.DatetimeIndex(group_index)
        calibration = _order_flow_calibration_slice(
            features=clean_features,
            volatility_state=volatility_state,
            calibration_end=pd.Timestamp(start),
            lookback_days=lookback_days,
            allowed_states=allowed_states,
        )
        if len(calibration) < min_calibration_rows:
            diagnostics_rows.append(
                {
                    "period_start": str(pd.Timestamp(start)),
                    "status": "skipped_insufficient_calibration_rows",
                    "calibration_rows": len(calibration),
                },
            )
            continue
        model = OrderFlowShockRuleSideModel(base_config).fit(calibration)
        period_side = model.primary_side(clean_features.loc[group_index]).astype(int)
        side.loc[group_index] = period_side
        diagnostics_rows.append(
            {
                "period_start": str(pd.Timestamp(start)),
                "status": "ok",
                "calibration_rows": len(calibration),
                "active_candidates": int((period_side != 0).sum()),
                "thresholds": model.diagnostics(),
            },
        )

    frame = pd.DataFrame({"primary_side": side, "primary_prediction": side}, index=clean_features.index)
    frame["volatility_state"] = volatility_state.reindex(frame.index)
    return frame, {
        "enabled": True,
        "status": "ok",
        "frequency": "monthly",
        "calibration_lookback_days": lookback_days,
        "min_calibration_rows": min_calibration_rows,
        "periods": diagnostics_rows,
    }


def _float_grid(config: dict[str, Any], key: str, default: list[float]) -> list[float]:
    raw = config.get(key, default)
    if isinstance(raw, (int, float, str)):
        raw = [raw]
    if not isinstance(raw, list):
        raise TypeError(f"primary_model.order_flow_shock.search.{key} must be a list")
    values = sorted({float(value) for value in raw})
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"primary_model.order_flow_shock.search.{key} values must be in [0, 1]")
    return values


def _order_flow_search_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    rule_config = primary_model_config(strategy_config).get("order_flow_shock", {})
    if not isinstance(rule_config, dict):
        return {}
    search_config = rule_config.get("search", {})
    return search_config if isinstance(search_config, dict) else {}


def _order_flow_candidate_rule_config(
    base_config: dict[str, Any],
    *,
    signed_quantile: float,
    ofi_min_quantile: float,
    ofi_max_quantile: float,
    buy_share_quantile: float,
    vpin_max_quantile: float,
) -> dict[str, Any] | None:
    if ofi_min_quantile >= ofi_max_quantile:
        return None
    config = {key: value for key, value in base_config.items() if key != "search"}
    config.update(
        {
            "min_signed_dollar_imbalance": None,
            "min_signed_dollar_imbalance_quantile": signed_quantile,
            "min_order_flow_imbalance": None,
            "min_order_flow_imbalance_quantile": ofi_min_quantile,
            "max_order_flow_imbalance": None,
            "max_order_flow_imbalance_quantile": ofi_max_quantile,
            "min_buy_notional_share": None,
            "min_buy_notional_share_quantile": buy_share_quantile,
            "max_vpin_zscore": None,
            "max_vpin_zscore_quantile": vpin_max_quantile,
        },
    )
    return config


def _order_flow_candidate_metrics(
    model: OrderFlowShockRuleSideModel,
    *,
    fit_features: pd.DataFrame,
    labels: pd.DataFrame,
    events: pd.DataFrame,
    round_trip_cost: float,
) -> dict[str, Any]:
    side = model.primary_side(fit_features)
    active_index = pd.DatetimeIndex(side.index[side.astype(int) != 0]).intersection(labels.index)
    returns = (pd.to_numeric(labels.loc[active_index, "ret"], errors="coerce") - float(round_trip_cost)).dropna()
    event_ends = pd.to_datetime(events.loc[returns.index, "t1"], utc=True, errors="coerce")
    position_proxy = position_aware_prediction_summary(
        pd.Series(1, index=returns.index, dtype=int),
        returns,
        event_ends=event_ends,
    )
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "candidate_count": int((side.astype(int) != 0).sum()),
        "trade_count": len(returns),
        "win_count": len(wins),
        "loss_count": len(losses),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "mean_return": float(returns.mean()) if not returns.empty else None,
        "position_aware": position_proxy,
    }


def _order_flow_candidate_passes(metrics: dict[str, Any], search_config: dict[str, Any]) -> bool:
    position_proxy = metrics.get("position_aware", {})
    position_proxy = position_proxy if isinstance(position_proxy, dict) else {}
    min_trades = int(search_config.get("min_trades", 30))
    min_positions = int(search_config.get("min_positions", 15))
    min_profit_factor = float(search_config.get("min_profit_factor", 1.5))
    min_position_profit_factor = float(search_config.get("min_position_profit_factor", min_profit_factor))
    min_mean_return = float(search_config.get("min_mean_return", 0.0005))
    min_position_mean_return = float(search_config.get("min_position_mean_return", min_mean_return))
    if int(metrics.get("trade_count", 0)) < min_trades:
        return False
    if int(position_proxy.get("trade_count", 0)) < min_positions:
        return False
    profit_factor = metrics.get("profit_factor")
    if profit_factor is None or float(profit_factor) <= min_profit_factor:
        return False
    position_profit_factor = position_proxy.get("profit_factor")
    if position_profit_factor is None or float(position_profit_factor) <= min_position_profit_factor:
        return False
    mean_return = metrics.get("mean_return")
    if mean_return is None or float(mean_return) <= min_mean_return:
        return False
    position_mean = position_proxy.get("mean_return")
    return position_mean is not None and float(position_mean) > min_position_mean_return


def _order_flow_candidate_score(metrics: dict[str, Any]) -> tuple[float, float, int, int]:
    position_proxy = metrics.get("position_aware", {})
    position_proxy = position_proxy if isinstance(position_proxy, dict) else {}
    position_pf = float(position_proxy.get("profit_factor") or 0.0)
    position_mean = float(position_proxy.get("mean_return") or 0.0)
    position_count = int(position_proxy.get("trade_count", 0))
    trade_count = int(metrics.get("trade_count", 0))
    return position_pf, position_mean, position_count, trade_count


def _order_flow_search_constraints(search_config: dict[str, Any]) -> dict[str, Any]:
    min_profit_factor = float(search_config.get("min_profit_factor", 1.5))
    min_mean_return = float(search_config.get("min_mean_return", 0.0005))
    return {
        "min_trades": int(search_config.get("min_trades", 30)),
        "min_positions": int(search_config.get("min_positions", 15)),
        "min_profit_factor": min_profit_factor,
        "min_position_profit_factor": float(
            search_config.get("min_position_profit_factor", min_profit_factor),
        ),
        "min_mean_return": min_mean_return,
        "min_position_mean_return": float(
            search_config.get("min_position_mean_return", min_mean_return),
        ),
    }


def _order_flow_search_label_dataset(
    *,
    fit_features: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    exit_config: dict[str, float],
    vertical_bars: int,
    window: TrainTestWindow,
    args: argparse.Namespace,
    round_trip_cost: float,
) -> tuple[AfmlDataset | None, dict[str, Any] | None]:
    side = pd.Series(1, index=pd.DatetimeIndex(fit_features.index), dtype=int)
    close_train = close.loc[close.index <= window.train_end]
    if close_train.empty:
        return None, {"enabled": True, "status": "skipped_empty_train_close"}
    try:
        dataset = build_afml_meta_dataset(
            close_train,
            side,
            features=fit_features,
            pt_sl=exit_config,
            target=volatility.loc[volatility.index <= window.train_end],
            min_ret=args.min_ret,
            meta_label_min_ret=round_trip_cost,
            vertical_barrier_days=None,
            vertical_barrier_bars=vertical_bars,
            oldest_weight=args.oldest_weight,
            feature_fracdiff_d=args.feature_fracdiff_d,
            feature_fracdiff_threshold=args.feature_fracdiff_threshold,
            return_type="log",
        )
    except ValueError as exc:
        return None, {"enabled": True, "status": "skipped_label_error", "error": str(exc)}
    return dataset, None


def _order_flow_search_grid(search_config: dict[str, Any]) -> list[tuple[float, float, float, float, float]]:
    signed_grid = _float_grid(search_config, "signed_quantiles", [0.90, 0.92, 0.94])
    ofi_min_grid = _float_grid(search_config, "ofi_min_quantiles", [0.55, 0.60, 0.65])
    ofi_max_grid = _float_grid(search_config, "ofi_max_quantiles", [0.80, 0.85, 0.90])
    buy_grid = _float_grid(search_config, "buy_share_quantiles", [0.90, 0.92, 0.94])
    vpin_grid = _float_grid(search_config, "vpin_max_quantiles", [0.90, 0.95, 1.0])
    return list(product(signed_grid, ofi_min_grid, ofi_max_grid, buy_grid, vpin_grid))


def _order_flow_search_row(
    *,
    base_config: dict[str, Any],
    quantiles: tuple[float, float, float, float, float],
    fit_features: pd.DataFrame,
    meta_dataset: AfmlDataset,
    search_config: dict[str, Any],
    round_trip_cost: float,
) -> tuple[OrderFlowShockRuleSideModel, dict[str, Any]] | None:
    signed_q, ofi_min_q, ofi_max_q, buy_q, vpin_q = quantiles
    candidate_config = _order_flow_candidate_rule_config(
        base_config,
        signed_quantile=signed_q,
        ofi_min_quantile=ofi_min_q,
        ofi_max_quantile=ofi_max_q,
        buy_share_quantile=buy_q,
        vpin_max_quantile=vpin_q,
    )
    if candidate_config is None:
        return None
    model = OrderFlowShockRuleSideModel(candidate_config).fit(fit_features)
    metrics = _order_flow_candidate_metrics(
        model,
        fit_features=fit_features,
        labels=meta_dataset.labels,
        events=meta_dataset.events,
        round_trip_cost=round_trip_cost,
    )
    return model, {
        "signed_quantile": signed_q,
        "ofi_min_quantile": ofi_min_q,
        "ofi_max_quantile": ofi_max_q,
        "buy_share_quantile": buy_q,
        "vpin_max_quantile": vpin_q,
        "passes": _order_flow_candidate_passes(metrics, search_config),
        "metrics": metrics,
        "thresholds": model.diagnostics(),
    }


def _run_order_flow_search_grid(
    *,
    base_config: dict[str, Any],
    fit_features: pd.DataFrame,
    meta_dataset: AfmlDataset,
    search_config: dict[str, Any],
    round_trip_cost: float,
) -> tuple[OrderFlowShockRuleSideModel | None, dict[str, Any] | None, list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    selected_model: OrderFlowShockRuleSideModel | None = None
    selected_row: dict[str, Any] | None = None
    max_evaluations = int(search_config.get("max_evaluations", 250))
    for evaluated, quantiles in enumerate(_order_flow_search_grid(search_config), start=1):
        result = _order_flow_search_row(
            base_config=base_config,
            quantiles=quantiles,
            fit_features=fit_features,
            meta_dataset=meta_dataset,
            search_config=search_config,
            round_trip_cost=round_trip_cost,
        )
        if result is None:
            continue
        model, row = result
        rows.append(row)
        if row["passes"] and (
            selected_row is None
            or _order_flow_candidate_score(row["metrics"])
            > _order_flow_candidate_score(selected_row["metrics"])
        ):
            selected_model = model
            selected_row = row
        if evaluated >= max_evaluations:
            return selected_model, selected_row, rows, evaluated
    return selected_model, selected_row, rows, len(rows)


def search_order_flow_shock_rule(
    *,
    base_config: dict[str, Any],
    fit_features: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    exit_config: dict[str, float],
    vertical_bars: int,
    window: TrainTestWindow,
    args: argparse.Namespace,
    strategy_config: dict[str, Any],
    round_trip_cost: float,
) -> tuple[OrderFlowShockRuleSideModel | None, dict[str, Any]]:
    search_config = _order_flow_search_config(strategy_config)
    if not bool(search_config.get("enabled", False)):
        return None, {"enabled": False}
    if fit_features.empty:
        return None, {"enabled": True, "status": "skipped_empty_fit_features"}

    meta_dataset, error = _order_flow_search_label_dataset(
        fit_features=fit_features,
        close=close,
        volatility=volatility,
        exit_config=exit_config,
        vertical_bars=vertical_bars,
        window=window,
        args=args,
        round_trip_cost=round_trip_cost,
    )
    if error is not None:
        return None, error
    assert meta_dataset is not None
    selected_model, selected_row, rows, evaluated = _run_order_flow_search_grid(
        base_config=base_config,
        fit_features=fit_features,
        meta_dataset=meta_dataset,
        search_config=search_config,
        round_trip_cost=round_trip_cost,
    )

    top_rows = sorted(
        rows,
        key=lambda row: _order_flow_candidate_score(row["metrics"]),
        reverse=True,
    )[: int(search_config.get("report_top_n", 10))]
    diagnostics = {
        "enabled": True,
        "status": "selected" if selected_model is not None else "no_candidate_passed_constraints",
        "evaluated": evaluated,
        "label_count": len(meta_dataset.labels),
        "selected": selected_row,
        "top_candidates": top_rows,
        "constraints": _order_flow_search_constraints(search_config),
    }
    return selected_model, diagnostics


def fit_primary_rule_model_with_optional_search(
    *,
    rule_model: Any,
    rule_model_name: str,
    fit_features: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    exit_config: dict[str, float],
    vertical_bars: int,
    window: TrainTestWindow,
    event_config: dict[str, Any],
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    round_trip_cost: float,
) -> tuple[Any, dict[str, Any]]:
    search_diagnostics: dict[str, Any] = {"enabled": False}
    if rule_model_name == PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME:
        rule_config = primary_model_config(strategy_config).get("order_flow_shock", {})
        rule_config = rule_config if isinstance(rule_config, dict) else {}
        selected_rule_model, search_diagnostics = search_order_flow_shock_rule(
            base_config=rule_config,
            fit_features=fit_features,
            close=close,
            volatility=volatility,
            exit_config=exit_config,
            vertical_bars=vertical_bars,
            window=window,
            args=args,
            strategy_config=strategy_config,
            round_trip_cost=round_trip_cost,
        )
        if selected_rule_model is not None:
            rule_model = selected_rule_model
    if hasattr(rule_model, "fit"):
        rule_model.fit(fit_features)
    diagnostics = primary_rule_side_model_diagnostics(
        rule_model,
        mode=rule_model_name,
        event_config=event_config,
    )
    diagnostics["search"] = search_diagnostics
    return rule_model, diagnostics


def primary_model_enabled(strategy_config: dict[str, Any]) -> bool:
    config = primary_model_config(strategy_config)
    return bool(config.get("enabled", False))


def primary_oof_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    oof = config.get("oof", {})
    return oof if isinstance(oof, dict) else {}


def primary_abstain_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    abstain = config.get("abstain", {})
    return abstain if isinstance(abstain, dict) else {}


def primary_regime_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    regime = config.get("regime_gating", {})
    return regime if isinstance(regime, dict) else {}


def primary_regime_switching_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    regime = config.get("regime_switching", {})
    return regime if isinstance(regime, dict) else {}


def primary_router_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    router = config.get("router", {})
    return router if isinstance(router, dict) else {}


def primary_policy_config(strategy_config: dict[str, Any], policy: str) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    policies = config.get("policies", {})
    if not isinstance(policies, dict):
        return {}
    policy_config = policies.get(policy, {})
    return policy_config if isinstance(policy_config, dict) else {}


def primary_expert_weighting_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    router = primary_router_config(strategy_config)
    if router:
        return router
    weighting = config.get("expert_weighting", {})
    return weighting if isinstance(weighting, dict) else {}


def primary_information_gate_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    gate = config.get("information_gate", {})
    return gate if isinstance(gate, dict) else {}


def primary_candidate_filter_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    config = primary_model_config(strategy_config)
    filters = config.get("candidate_filters", {})
    return filters if isinstance(filters, dict) else {}


def _append_abstain_reason(existing: pd.Series, reason: str) -> pd.Series:
    text = existing.fillna("").astype(str)
    return text.where(text.eq(""), text + "+") + reason


def _allowed_candidate_filter_sides(config: dict[str, Any]) -> list[int]:
    raw_sides = config.get("sides", [])
    if isinstance(raw_sides, (str, int, float)):
        raw_sides = [raw_sides]
    side_aliases = {
        "long": 1,
        "buy": 1,
        "1": 1,
        "+1": 1,
        "short": -1,
        "sell": -1,
        "-1": -1,
    }
    allowed_sides: list[int] = []
    for value in raw_sides:
        if isinstance(value, str):
            side = side_aliases.get(value.strip().lower())
            if side is None:
                raise ValueError(f"Unsupported primary_model.candidate_filters side: {value!r}")
        else:
            side = int(value)
            if side not in {-1, 1}:
                raise ValueError(f"Unsupported primary_model.candidate_filters side: {value!r}")
        if side not in allowed_sides:
            allowed_sides.append(side)
    return allowed_sides


def apply_primary_candidate_filters(
    frame: pd.DataFrame,
    strategy_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = primary_candidate_filter_config(strategy_config)
    if not bool(config.get("enabled", False)):
        return frame, {"enabled": False}
    out = frame.copy()
    initial_active = int((out.get("primary_side", pd.Series(dtype=int)).astype(int) != 0).sum())
    allowed_states = config.get("volatility_states", [])
    if isinstance(allowed_states, str):
        allowed_states = [allowed_states]
    allowed_states = [str(state) for state in allowed_states if str(state)]
    allowed_sides = _allowed_candidate_filter_sides(config)
    blocked = pd.Series(False, index=out.index)
    filters_applied: list[str] = []
    if allowed_states:
        if "volatility_state" in out:
            blocked |= ~out["volatility_state"].fillna("unknown").astype(str).isin(allowed_states)
        else:
            blocked |= True
        filters_applied.append("volatility_states")
    if allowed_sides:
        if "primary_side" in out:
            blocked |= ~out["primary_side"].astype(int).isin(allowed_sides)
        else:
            blocked |= True
        filters_applied.append("sides")
    if bool(blocked.any()):
        active_blocked = blocked & (out["primary_side"].astype(int) != 0)
        out.loc[blocked, "primary_side"] = 0
        if "primary_prediction" in out:
            out.loc[blocked, "primary_prediction"] = 0
        if "primary_abstain_reason" not in out:
            out["primary_abstain_reason"] = ""
        out.loc[blocked, "primary_abstain_reason"] = _append_abstain_reason(
            out.loc[blocked, "primary_abstain_reason"],
            "candidate_filter",
        )
    final_active = int((out["primary_side"].astype(int) != 0).sum()) if "primary_side" in out else 0
    return out, {
        "enabled": True,
        "filters_applied": filters_applied,
        "allowed_volatility_states": allowed_states,
        "allowed_sides": allowed_sides,
        "initial_active_candidates": initial_active,
        "filtered_active_candidates": int(active_blocked.sum()) if bool(blocked.any()) else 0,
        "final_active_candidates": final_active,
    }


def make_multi_expert_primary_model(
    *,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    rule_model: Any,
    expert_weights: dict[str, float] | None = None,
    expert_scores: dict[str, float] | None = None,
    expert_metrics: dict[str, Any] | None = None,
    seed_offset: int = 0,
    progress_label: str | None = None,
    policy: str | None = None,
) -> AfmlMoeWeightedVotePrimarySideModel:
    config = primary_model_config(strategy_config)
    mode = str(config.get("mode", PRIMARY_MOE_MODEL_NAME))
    if mode not in {PRIMARY_MOE_MODEL_NAME, "regime_switching_dual_policy"}:
        raise ValueError(f"Unsupported primary_model.mode: {mode}")
    abstain = primary_abstain_config(strategy_config)
    regime = primary_regime_config(strategy_config)
    router = primary_router_config(strategy_config)
    experts_config = config.get("experts")
    if policy is not None:
        policy_config = primary_policy_config(strategy_config, policy)
        experts_config = policy_config.get("experts", experts_config)
    if isinstance(experts_config, dict):
        for name, values in experts_config.items():
            lowered = str(name).lower()
            if lowered in BANNED_PRIMARY_EXPERT_NAMES:
                raise ValueError(f"Primary expert {name!r} has been removed from AFML-MoE")
            if isinstance(values, dict):
                family = str(values.get("family", "")).lower()
                if family in BANNED_PRIMARY_EXPERT_NAMES:
                    raise ValueError(f"Primary expert {name!r} uses banned family {family!r}")
    base_seed = int(section(strategy_config, "sequential_bagging").get("random_seed", 2027))
    return AfmlMoeWeightedVotePrimarySideModel(
        experts=experts_config,
        router=router,
        confidence_threshold=float(
            config.get("confidence_threshold", abstain.get("confidence_threshold", 0.45)),
        ),
        disagreement_threshold=float(
            router.get(
                "max_disagreement",
                config.get("disagreement_threshold", abstain.get("disagreement_threshold", 0.45)),
            ),
        ),
        abstain_enabled=bool(abstain.get("enabled", config.get("abstain_enabled", True))),
        regime_gating_enabled=bool(regime.get("enabled", config.get("regime_gating_enabled", True))),
        expert_weights=expert_weights,
        expert_scores=expert_scores,
        expert_metrics=expert_metrics,
        random_state=base_seed + seed_offset,
        n_jobs=args.n_jobs,
        verbose=not args.no_progress,
        progress_interval=args.progress_interval,
        progress_label=progress_label,
        rule_model=rule_model,
    )


def fit_afml_side_model(
    model: Any,
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    *,
    progress_label: str | None = None,
    args: argparse.Namespace | None = None,
) -> Any:
    started = (
        progress_start(
            args,
            f"{progress_label}: fitting side model rows={len(train_index):,} "
            f"features={dataset.X.shape[1]:,} classes={class_counts(dataset.y.loc[train_index])}",
        )
        if progress_label
        else None
    )
    if getattr(model, "requires_dataset", False):
        model.fit_dataset(dataset, train_index)
    elif getattr(model, "requires_event_spans", False):
        model.fit(
            dataset.X.loc[train_index],
            dataset.y.loc[train_index],
            sample_weight=dataset.sample_weight.loc[train_index],
            t1=dataset.events.loc[train_index, "t1"],
            bar_index=dataset.close.index,
        )
    else:
        model.fit(
            dataset.X.loc[train_index],
            dataset.y.loc[train_index],
            sample_weight=dataset.sample_weight.loc[train_index],
        )
    if progress_label:
        progress_done(args, f"{progress_label}: fitting side model", started)
    return model


def build_primary_side_dataset(
    *,
    close: pd.Series,
    features: pd.DataFrame,
    volatility: pd.Series,
    exit_config: dict[str, float],
    vertical_bars: int,
    window: TrainTestWindow,
    args: argparse.Namespace,
) -> AfmlDataset:
    cusum_threshold = make_rolling_cusum_threshold(
        close,
        window=args.cusum_window,
        train_start=window.train_start,
        train_end=window.train_end,
        floor_quantile=args.cusum_floor_quantile,
    ) * float(args.cusum_threshold_mult)
    return build_afml_dataset(
        close,
        features=features,
        cusum_threshold=cusum_threshold,
        cusum_threshold_mult=1.0,
        pt_sl=exit_config,
        target=volatility,
        min_ret=args.min_ret,
        vertical_barrier_days=None,
        vertical_barrier_bars=vertical_bars,
        drop_neutral=True,
        rare_label_min_pct=args.rare_label_min_pct,
        oldest_weight=args.oldest_weight,
        feature_fracdiff_d=args.feature_fracdiff_d,
        feature_fracdiff_threshold=args.feature_fracdiff_threshold,
        return_type="log",
    )


def purged_train_index_for_validation(
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    validation_index: pd.DatetimeIndex,
    *,
    embargo_bars: int,
) -> pd.DatetimeIndex:
    all_index = pd.DatetimeIndex(dataset.y.index)
    starts, ends = dataset_event_span_arrays(dataset)
    positions = pd.Series(np.arange(len(all_index), dtype=np.int64), index=all_index)
    validation_index = pd.DatetimeIndex(validation_index).intersection(all_index)
    if validation_index.empty:
        return pd.DatetimeIndex([])
    validation_positions = positions.loc[validation_index].to_numpy(dtype=np.int64)
    validation_start = int(starts[validation_positions].min())
    validation_end = int(ends[validation_positions].max())

    candidate_index = pd.DatetimeIndex(train_index).difference(validation_index).intersection(all_index)
    if candidate_index.empty:
        return candidate_index
    candidate_positions = positions.loc[candidate_index].to_numpy(dtype=np.int64)
    embargo = int(max(0, embargo_bars))
    keep = (
        (ends[candidate_positions] < validation_start - embargo)
        | (starts[candidate_positions] > validation_end + embargo)
    )
    return pd.DatetimeIndex(candidate_index[keep])


def weighted_direction_accuracy(
    truth: pd.Series,
    prediction: pd.Series,
    sample_weight: pd.Series,
) -> float:
    truth = truth.astype(int)
    prediction = prediction.reindex(truth.index).fillna(0).astype(int)
    weights = sample_weight.reindex(truth.index).fillna(0.0).astype(float)
    correct = (truth == prediction).astype(float)
    if weights.sum() <= 0.0:
        return float(correct.mean()) if len(correct) else 0.0
    return float(np.average(correct.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def weighted_mean(values: pd.Series, sample_weight: pd.Series | None = None) -> float:
    values = values.dropna().astype(float)
    if values.empty:
        return 0.0
    if sample_weight is None:
        return float(values.mean())
    weights = sample_weight.reindex(values.index).fillna(0.0).clip(lower=0.0).astype(float)
    if weights.sum() <= 0.0:
        return float(values.mean())
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def negative_log_loss_score(
    truth: pd.Series,
    proba: pd.DataFrame,
    sample_weight: pd.Series | None = None,
) -> float:
    truth = truth.astype(int)
    index = truth.index.intersection(proba.index)
    if index.empty:
        return float("-inf")
    truth = truth.loc[index]
    proba = proba.loc[index]
    probabilities = pd.Series(0.0, index=index, dtype=float)
    for label in sorted(truth.unique()):
        if label in proba.columns:
            mask = truth == int(label)
            probabilities.loc[mask] = pd.to_numeric(proba.loc[mask, label], errors="coerce")
    probabilities = probabilities.clip(lower=1e-12, upper=1.0).fillna(1e-12)
    loss = -np.log(probabilities)
    weights = sample_weight.reindex(index).fillna(0.0).clip(lower=0.0) if sample_weight is not None else None
    return -weighted_mean(loss, weights)


def side_adjusted_proxy_returns(
    labels: pd.DataFrame,
    prediction: pd.Series,
) -> pd.Series:
    if "ret" not in labels:
        return pd.Series(dtype=float)
    index = labels.index.intersection(prediction.index)
    if index.empty:
        return pd.Series(dtype=float)
    raw_returns = labels.loc[index, "ret"].astype(float)
    side = prediction.reindex(index).fillna(0).astype(int)
    return (raw_returns * side).rename("side_adjusted_return")


def expert_information_metrics(
    *,
    truth: pd.Series,
    proba: pd.DataFrame,
    sample_weight: pd.Series,
    labels: pd.DataFrame,
    round_trip_cost: float = 0.0,
) -> dict[str, Any]:
    prediction = pd.Series(proba.idxmax(axis=1).astype(int), index=proba.index)
    index = truth.index.intersection(prediction.index)
    metrics = classification_metrics(truth.loc[index], prediction.loc[index], positive_label=1)
    proxy_returns = side_adjusted_proxy_returns(labels.loc[index], prediction.loc[index])
    if not proxy_returns.empty:
        proxy_returns = proxy_returns - float(round_trip_cost)
    neg_log_loss = negative_log_loss_score(
        truth.loc[index],
        proba.loc[index],
        sample_weight.loc[index],
    )
    wins = proxy_returns[proxy_returns > 0.0]
    losses = proxy_returns[proxy_returns < 0.0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(-losses.sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (math.inf if gross_profit > 0.0 else 0.0)
    return {
        "count": len(index),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "negative_log_loss": neg_log_loss,
        "proxy_mean_return": weighted_mean(proxy_returns, sample_weight.loc[index]),
        "proxy_win_rate": float((proxy_returns > 0.0).mean()) if not proxy_returns.empty else 0.0,
        "proxy_profit_factor": profit_factor,
        "proxy_gross_profit": gross_profit,
        "proxy_gross_loss": gross_loss,
        "proxy_win_count": int((proxy_returns > 0.0).sum()),
        "proxy_loss_count": int((proxy_returns < 0.0).sum()),
        "round_trip_cost": float(round_trip_cost),
    }


def expert_information_score(
    metrics: dict[str, Any],
    *,
    n_classes: int,
    config: dict[str, Any],
) -> float:
    random_score = 1.0 / float(max(2, n_classes))
    baseline_neg_log_loss = math.log(random_score)
    weights = config.get("weights", {}) if isinstance(config.get("weights"), dict) else {}
    proxy_scale = max(1e-12, float(config.get("proxy_return_scale", 0.001)))
    balanced_gain = max(0.0, float(metrics.get("balanced_accuracy", 0.0)) - random_score)
    f1_gain = max(0.0, float(metrics.get("macro_f1", 0.0)) - random_score)
    log_loss_gain = max(0.0, float(metrics.get("negative_log_loss", float("-inf"))) - baseline_neg_log_loss)
    proxy_gain = max(0.0, float(metrics.get("proxy_mean_return", 0.0)) / proxy_scale)
    score = (
        float(weights.get("balanced_accuracy", 0.40)) * balanced_gain
        + float(weights.get("macro_f1", 0.25)) * f1_gain
        + float(weights.get("neg_log_loss", 0.20)) * log_loss_gain
        + float(weights.get("proxy_return", 0.15)) * proxy_gain
    )
    min_balanced_accuracy = float(config.get("min_balanced_accuracy", random_score))
    min_macro_f1 = float(config.get("min_macro_f1", 0.0))
    min_proxy_return = float(config.get("min_proxy_mean_return", 0.0))
    min_profit_factor = float(config.get("min_profit_factor", 1.0))
    if float(metrics.get("balanced_accuracy", 0.0)) <= min_balanced_accuracy:
        return 0.0
    if float(metrics.get("macro_f1", 0.0)) < min_macro_f1:
        return 0.0
    if float(metrics.get("proxy_mean_return", 0.0)) <= min_proxy_return:
        return 0.0
    if float(metrics.get("proxy_profit_factor", 0.0)) < min_profit_factor:
        return 0.0
    return max(0.0, float(score))


def normalize_positive_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {name: max(0.0, float(value)) for name, value in weights.items()}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in positive.items()}


PRIMARY_POLICY_NAMES = ("trend_following", "mean_reversion")


def policy_rule_model(strategy_config: dict[str, Any], symbol: str, policy: str) -> Any:
    if policy == "trend_following":
        return CvdSlopeRuleSideModel(runtime_event_definition(strategy_config))
    if policy == "mean_reversion":
        policy_config = primary_policy_config(strategy_config, policy)
        config = policy_config.get("mean_reversion_rule", {})
        return MeanReversionRuleSideModel(config if isinstance(config, dict) else {})
    raise ValueError(f"Unsupported primary policy: {policy}")


def fit_primary_policy_model(
    *,
    policy: str,
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    symbol: str,
    seed_offset: int,
    progress_label: str,
) -> AfmlMoeWeightedVotePrimarySideModel:
    model = make_multi_expert_primary_model(
        strategy_config=strategy_config,
        args=args,
        rule_model=policy_rule_model(strategy_config, symbol, policy),
        seed_offset=seed_offset,
        progress_label=progress_label,
        policy=policy,
    )
    return fit_afml_side_model(
        model,
        dataset,
        train_index,
        progress_label=progress_label,
        args=args,
    )


def policy_net_returns(
    dataset: AfmlDataset,
    policy_frame: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    round_trip_cost: float,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(index).intersection(dataset.labels.index).intersection(policy_frame.index)
    if index.empty:
        return pd.DataFrame(columns=["side", "net_return"])
    side = policy_frame.loc[index, "primary_side"].fillna(0).astype(int)
    raw_return = dataset.labels.loc[index, "ret"].astype(float) * side
    return pd.DataFrame(
        {
            "side": side,
            "net_return": raw_return - float(round_trip_cost),
        },
        index=index,
    )


def policy_edge_is_eligible(
    summary: dict[str, Any],
    *,
    config: dict[str, Any],
) -> bool:
    min_trades = int(config.get("min_policy_trades", 30))
    min_profit_factor = float(config.get("min_policy_profit_factor", 1.0))
    min_mean_return = float(config.get("min_policy_mean_return", 0.0))
    min_win_count = int(config.get("min_policy_win_count", 1))
    min_loss_count = int(config.get("min_policy_loss_count", 1))
    profit_factor = summary.get("profit_factor")
    mean_return = summary.get("mean_return")
    return (
        int(summary.get("trade_count", 0)) >= min_trades
        and int(summary.get("win_count", 0)) >= min_win_count
        and int(summary.get("loss_count", 0)) >= min_loss_count
        and profit_factor is not None
        and float(profit_factor) > min_profit_factor
        and mean_return is not None
        and float(mean_return) > min_mean_return
    )


def policy_edge_score(summary: dict[str, Any], *, config: dict[str, Any]) -> float:
    profit_factor = min(float(summary.get("profit_factor") or 0.0), 10.0)
    mean_return = float(summary.get("mean_return") or 0.0)
    return profit_factor + mean_return / max(1e-12, float(config.get("proxy_return_scale", 0.001)))


def build_policy_edge_table(
    *,
    dataset: AfmlDataset,
    regime_frame: pd.DataFrame,
    policy_frames: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    config: dict[str, Any],
    validation_config: dict[str, Any],
    n_trials: int,
    round_trip_cost: float,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(index).intersection(regime_frame.index).intersection(dataset.labels.index)
    table: dict[str, Any] = {}
    rows: list[pd.DataFrame] = []
    for policy, frame in policy_frames.items():
        returns = policy_net_returns(
            dataset,
            frame,
            index,
            round_trip_cost=round_trip_cost,
        )
        if returns.empty:
            continue
        returns["policy"] = policy
        returns["primary_market_state"] = regime_frame.loc[returns.index, "primary_market_state"].astype(str)
        rows.append(returns)
    if not rows:
        return table
    combined = pd.concat(rows, axis=0)
    combined = combined.loc[combined["side"].astype(int) != 0].copy()
    if combined.empty:
        return table

    group_specs = [
        ("primary_market_state",),
        ("all",),
    ]
    for spec in group_specs:
        if spec == ("all",):
            grouped = [("all", combined)]
        else:
            grouped = list(combined.groupby("primary_market_state", sort=True))
        for state, state_group in grouped:
            state_key = str(state)
            table.setdefault(state_key, {})
            for (policy, side), group in state_group.groupby(["policy", "side"], sort=True):
                summary = return_stream_summary(
                    pd.to_numeric(group["net_return"], errors="coerce"),
                    validation_config=validation_config,
                    n_trials=n_trials,
                    return_basis="cost_adjusted_policy_label_ret",
                    round_trip_cost=float(round_trip_cost),
                )
                summary["policy"] = str(policy)
                summary["side"] = int(side)
                summary["primary_market_state"] = state_key
                summary["eligible"] = policy_edge_is_eligible(summary, config=config)
                summary["score"] = policy_edge_score(summary, config=config) if summary["eligible"] else 0.0
                table[state_key].setdefault(str(policy), {})[str(int(side))] = summary
    return table


def flatten_policy_edge_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, state_table in table.items():
        if not isinstance(state_table, dict):
            continue
        for policy, policy_table in state_table.items():
            if not isinstance(policy_table, dict):
                continue
            for side, summary in policy_table.items():
                if not isinstance(summary, dict):
                    continue
                row = dict(summary)
                row["primary_market_state"] = state
                row["policy"] = policy
                row["side"] = int(side)
                rows.append(row)
    return rows


def policy_edge_table_has_eligible(table: dict[str, Any]) -> bool:
    return any(bool(row.get("eligible", False)) for row in flatten_policy_edge_table(table))


def primary_frame_activity_summary(
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(index).intersection(frame.index)
    if index.empty:
        return {"rows": 0}
    sub = frame.loc[index]
    active = sub["primary_side"].astype(int) != 0
    summary: dict[str, Any] = {
        "rows": len(sub),
        "candidate_count": int(active.sum()),
        "abstain_count": int((~active).sum()),
        "abstain_rate": float((~active).mean()),
        "side_counts": class_counts(sub.loc[active, "primary_side"]) if active.any() else {},
    }
    for column in ("primary_confidence", "primary_uncertainty", "primary_disagreement"):
        if column in sub:
            summary[f"mean_{column.removeprefix('primary_')}"] = float(
                pd.to_numeric(sub[column], errors="coerce").mean(),
            )
    if "primary_abstain_reason" in sub:
        reasons = sub.loc[~active, "primary_abstain_reason"].fillna("").astype(str)
        summary["abstain_reasons"] = {
            reason: int(count)
            for reason, count in reasons.value_counts().sort_index().items()
            if reason
        }
    if "primary_driver_expert" in sub:
        drivers = sub.loc[active, "primary_driver_expert"].fillna("").astype(str)
        summary["driver_expert_counts"] = {
            driver: int(count)
            for driver, count in drivers.value_counts().sort_index().items()
            if driver
        }
    if "primary_policy" in sub:
        policies = sub.loc[active, "primary_policy"].fillna("").astype(str)
        summary["policy_counts"] = {
            policy: int(count)
            for policy, count in policies.value_counts().sort_index().items()
            if policy
        }
    if "primary_market_state" in sub:
        states = sub["primary_market_state"].fillna("").astype(str)
        summary["market_state_counts"] = {
            state: int(count)
            for state, count in states.value_counts().sort_index().items()
            if state
        }
    return summary


def primary_information_audit(
    *,
    dataset: AfmlDataset,
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    index = pd.DatetimeIndex(index).intersection(dataset.y.index).intersection(frame.index)
    if index.empty:
        return {"rows": 0}
    sub = frame.loc[index].copy()
    truth = dataset.y.loc[index]
    side = sub["primary_side"].fillna(0).astype(int)
    active = side != 0
    proxy_returns = side_adjusted_proxy_returns(dataset.labels.loc[index], side)
    active_returns = proxy_returns.loc[active]
    audit: dict[str, Any] = {
        "rows": len(index),
        "active_rows": int(active.sum()),
        "active_rate": float(active.mean()),
        "metrics_all": classification_metrics(truth, side, positive_label=1),
        "metrics_active": (
            classification_metrics(truth.loc[active], side.loc[active], positive_label=1)
            if active.any()
            else {}
        ),
        "proxy_mean_return": weighted_mean(active_returns, dataset.sample_weight.loc[active_returns.index]),
        "proxy_win_rate": float((active_returns > 0.0).mean()) if not active_returns.empty else 0.0,
    }
    if "primary_confidence" in sub:
        confidence = pd.to_numeric(sub["primary_confidence"], errors="coerce")
        decile_frame = pd.DataFrame(
            {
                "confidence": confidence,
                "proxy_return": proxy_returns,
                "correct": (truth == side).astype(float),
                "active": active.astype(int),
            },
            index=index,
        ).dropna(subset=["confidence"])
        if not decile_frame.empty:
            try:
                decile_frame["decile"] = pd.qcut(
                    decile_frame["confidence"].rank(method="first"),
                    q=min(10, len(decile_frame)),
                    labels=False,
                )
            except ValueError:
                decile_frame["decile"] = 0
            audit["confidence_decile_edge"] = [
                {
                    "decile": int(decile),
                    "count": len(group),
                    "active_count": int(group["active"].sum()),
                    "mean_confidence": float(group["confidence"].mean()),
                    "accuracy": float(group["correct"].mean()),
                    "mean_proxy_return": float(group.loc[group["active"] == 1, "proxy_return"].mean())
                    if int(group["active"].sum()) > 0
                    else 0.0,
                }
                for decile, group in decile_frame.groupby("decile", sort=True)
            ]
    if "primary_driver_expert" in sub and not active_returns.empty:
        driver = sub.loc[active_returns.index, "primary_driver_expert"].fillna("").astype(str)
        audit["driver_proxy_pnl"] = {
            name: {
                "count": int((driver == name).sum()),
                "mean_return": float(active_returns.loc[driver == name].mean()),
                "sum_return": float(active_returns.loc[driver == name].sum()),
            }
            for name in sorted(driver[driver != ""].unique())
        }
    if not active_returns.empty:
        active_side = side.loc[active_returns.index]
        audit["side_proxy_pnl"] = {
            str(int(name)): {
                "count": int((active_side == name).sum()),
                "mean_return": float(active_returns.loc[active_side == name].mean()),
                "sum_return": float(active_returns.loc[active_side == name].sum()),
            }
            for name in sorted(active_side.unique())
        }
    return audit


def primary_failure_reasons(  # noqa: C901
    diagnostics: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> list[str]:
    config = config if isinstance(config, dict) else {}
    audit = diagnostics.get("primary_information_audit", {})
    if not isinstance(audit, dict):
        return []
    oof_audit = audit.get("oof", {}) if isinstance(audit.get("oof"), dict) else {}
    test_audit = audit.get("test", {}) if isinstance(audit.get("test"), dict) else {}
    info = oof_audit or test_audit
    reasons: list[str] = []
    active_metrics = info.get("metrics_active", {}) if isinstance(info.get("metrics_active"), dict) else {}
    min_active_accuracy = float(config.get("min_active_accuracy", 0.5))
    min_proxy_mean_return = float(config.get("min_proxy_mean_return", 0.0))
    min_proxy_win_rate = float(config.get("min_proxy_win_rate", 0.5))
    if active_metrics and float(active_metrics.get("accuracy", 0.0)) <= min_active_accuracy:
        reasons.append("weak_active_edge")
    if float(info.get("proxy_mean_return", 0.0)) <= min_proxy_mean_return:
        reasons.append("weak_active_edge")
    if float(info.get("proxy_win_rate", 0.0)) <= min_proxy_win_rate:
        reasons.append("weak_active_edge")

    activity = diagnostics.get("activity", {})
    activity_all = activity.get("all", {}) if isinstance(activity, dict) and isinstance(activity.get("all"), dict) else {}
    max_mean_disagreement = float(config.get("max_mean_disagreement", 0.45))
    disagreement = activity_all.get("mean_disagreement")
    if disagreement is not None and float(disagreement) > max_mean_disagreement:
        reasons.append("excess_disagreement")

    min_driver_count = int(config.get("min_driver_count", 30))
    min_driver_mean_return = float(config.get("min_driver_mean_return", 0.0))
    driver_pnl = info.get("driver_proxy_pnl", {}) if isinstance(info.get("driver_proxy_pnl"), dict) else {}
    for values in driver_pnl.values():
        if (
            isinstance(values, dict)
            and int(values.get("count", 0)) >= min_driver_count
            and float(values.get("mean_return", 0.0)) <= min_driver_mean_return
        ):
            reasons.append("driver_negative_proxy_pnl")
            break

    min_side_count = int(config.get("min_side_count", 30))
    min_side_mean_return = float(config.get("min_side_mean_return", 0.0))
    side_pnl = info.get("side_proxy_pnl", {}) if isinstance(info.get("side_proxy_pnl"), dict) else {}
    for values in side_pnl.values():
        if (
            isinstance(values, dict)
            and int(values.get("count", 0)) >= min_side_count
            and float(values.get("mean_return", 0.0)) <= min_side_mean_return
        ):
            reasons.append("side_instability")
            break

    oof = diagnostics.get("oof", {}) if isinstance(diagnostics.get("oof"), dict) else {}
    if bool(oof.get("all_experts_rejected", False)):
        reasons.append("all_experts_below_information_threshold")
    if bool(oof.get("all_policies_rejected", False)):
        reasons.append("no_positive_policy_edge")
    rejected_experts = oof.get("rejected_unstable_experts", {})
    if isinstance(rejected_experts, dict) and rejected_experts:
        reasons.append("unstable_expert_folds")

    return sorted(set(reasons))


def primary_feature_importance_diagnostics(
    *,
    model: Any,
    dataset: AfmlDataset,
    test_index: pd.DatetimeIndex,
    strategy_config: dict[str, Any],
    output_dir: Path,
    symbol: str,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    config = primary_model_config(strategy_config).get("feature_importance", {})
    config = config if isinstance(config, dict) else {}
    if not bool(config.get("enabled", False)):
        return {"enabled": False}

    diagnostics: dict[str, Any] = {"enabled": True}
    stage_started = progress_start(args, f"{symbol}: primary feature importance diagnostics")
    tree_paths: dict[str, str | None] = {}
    policy_models: dict[str, Any]
    if hasattr(model, "trend_model") and hasattr(model, "reversion_model"):
        policy_models = {
            "trend_following": model.trend_model,
            "mean_reversion": model.reversion_model,
        }
    else:
        policy_models = {"global": model}
    for policy, policy_model in policy_models.items():
        expert_configs = getattr(policy_model, "expert_configs_", {})
        for name, expert in getattr(policy_model, "expert_models_", {}).items():
            config = expert_configs.get(name, {}) if isinstance(expert_configs, dict) else {}
            if str(config.get("family", "bagged_tree")).lower() != "bagged_tree":
                continue
            try:
                mdi_started = progress_start(args, f"{symbol}: primary MDI {name}")
                mdi = mdi_feature_importance(expert)
            except ValueError:
                continue
            finally:
                progress_done(args, f"{symbol}: primary MDI {name}", mdi_started)
            key = name if policy == "global" else f"{policy}_{name}"
            path = output_dir / f"{symbol}_primary_{key}_mdi_importance.csv"
            tree_paths[key] = write_importance_csv(mdi, path)
            diagnostics[f"{key}_top_mdi_features"] = mdi.head(10).to_dict(orient="records")
    diagnostics["mdi_csv"] = tree_paths

    if bool(config.get("mda_enabled", True)) and not test_index.empty:
        mda = mda_feature_importance(
            model,
            dataset.X.loc[test_index],
            dataset.y.loc[test_index],
            sample_weight=dataset.sample_weight.loc[test_index],
            n_repeats=int(config.get("mda_repeats", 2)),
            random_state=int(config.get("random_state", 42)),
            progress_label=f"{symbol}:primary_mda",
            progress_interval=int(getattr(args, "progress_interval", 10) if args is not None else 10),
        )
        diagnostics["mda_csv"] = write_importance_csv(
            mda,
            output_dir / f"{symbol}_primary_mda_importance.csv",
        )
        diagnostics["top_mda_features"] = mda.head(10).to_dict(orient="records")
    else:
        diagnostics["mda_csv"] = None

    diagnostics["sfi_enabled"] = bool(config.get("sfi_enabled", False))
    diagnostics["sfi_csv"] = None
    progress_done(args, f"{symbol}: primary feature importance diagnostics", stage_started)
    return diagnostics


def primary_oof_frame_and_weights(
    *,
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    symbol: str,
    round_trip_cost: float,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    config = primary_oof_config(strategy_config)
    if not bool(config.get("enabled", True)):
        return pd.DataFrame(), {}, {"enabled": False}

    ordered = pd.DatetimeIndex(train_index).intersection(dataset.X.index).sort_values()
    n_splits = max(2, int(config.get("n_splits", 3)))
    min_train_events = int(config.get("min_train_events", 50))
    min_validation_events = int(config.get("min_validation_events", 10))
    embargo_bars = int(config.get("embargo_bars", args.embargo_bars))
    folds = [
        pd.DatetimeIndex(values)
        for values in np.array_split(ordered.to_numpy(), n_splits)
        if len(values) >= min_validation_events
    ]
    stage_started = progress_start(
        args,
        f"{symbol}: primary OOF training folds={len(folds)}/{n_splits} "
        f"ordered_train_events={len(ordered):,} embargo_bars={embargo_bars}",
    )

    oof_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    expert_score_sum: dict[str, float] = {}
    expert_score_weight: dict[str, float] = {}
    expert_metric_rows: dict[str, list[dict[str, Any]]] = {}
    weighting_config = primary_expert_weighting_config(strategy_config)
    n_classes = max(2, int(dataset.y.nunique()))

    for fold_id, validation_index in enumerate(folds, start=1):
        fold_started = progress_start(
            args,
            f"{symbol}: primary OOF fold {fold_id}/{len(folds)} "
            f"validation_events={len(validation_index):,}",
        )
        inner_train = purged_train_index_for_validation(
            dataset,
            ordered,
            validation_index,
            embargo_bars=embargo_bars,
        )
        progress_print(
            args,
            f"{symbol}: primary OOF fold {fold_id}/{len(folds)} "
            f"purged_train_events={len(inner_train):,} "
            f"validation_start={validation_index.min()} validation_end={validation_index.max()}",
        )
        if len(inner_train) < min_train_events or dataset.y.loc[inner_train].nunique() < 2:
            fold_summaries.append(
                {
                    "fold": fold_id,
                    "status": "skipped_insufficient_train",
                    "train_events": len(inner_train),
                    "validation_events": len(validation_index),
                },
            )
            progress_done(args, f"{symbol}: primary OOF fold {fold_id}/{len(folds)} skipped", fold_started)
            continue

        model = make_multi_expert_primary_model(
            strategy_config=strategy_config,
            args=args,
            rule_model=CvdSlopeRuleSideModel(runtime_event_definition(strategy_config)),
            seed_offset=30_000 + fold_id,
            progress_label=f"{symbol}:primary_oof_{fold_id}",
        )
        model = fit_afml_side_model(
            model,
            dataset,
            inner_train,
            progress_label=f"{symbol}:primary_oof_{fold_id}",
            args=args,
        )
        predict_started = progress_start(args, f"{symbol}: primary OOF fold {fold_id}/{len(folds)} predict")
        fold_frame = model.primary_side_frame(dataset.X.loc[validation_index])
        oof_frames.append(fold_frame)
        progress_done(args, f"{symbol}: primary OOF fold {fold_id}/{len(folds)} predict", predict_started)

        metrics_started = progress_start(
            args,
            f"{symbol}: primary OOF fold {fold_id}/{len(folds)} expert metrics",
        )
        expert_frames = model.expert_probability_frames(dataset.X.loc[validation_index])
        for name, proba in expert_frames.items():
            prediction = pd.Series(proba.idxmax(axis=1).astype(int), index=proba.index)
            metrics = expert_information_metrics(
                truth=dataset.y.loc[prediction.index],
                proba=proba,
                sample_weight=dataset.sample_weight.loc[prediction.index],
                labels=dataset.labels.loc[prediction.index],
                round_trip_cost=round_trip_cost,
            )
            score = expert_information_score(
                metrics,
                n_classes=n_classes,
                config=weighting_config,
            )
            weight = float(dataset.sample_weight.loc[prediction.index].sum())
            expert_score_sum[name] = expert_score_sum.get(name, 0.0) + score * max(weight, 1.0)
            expert_score_weight[name] = expert_score_weight.get(name, 0.0) + max(weight, 1.0)
            metric_row = {"fold": fold_id, "score": score}
            metric_row.update(metrics)
            expert_metric_rows.setdefault(name, []).append(metric_row)
        progress_done(
            args,
            f"{symbol}: primary OOF fold {fold_id}/{len(folds)} expert metrics",
            metrics_started,
        )

        fold_summaries.append(
            {
                "fold": fold_id,
                "status": "ok",
                "train_events": len(inner_train),
                "validation_events": len(validation_index),
                "validation_start": validation_index.min(),
                "validation_end": validation_index.max(),
                "embargo_bars": embargo_bars,
            },
        )
        progress_done(args, f"{symbol}: primary OOF fold {fold_id}/{len(folds)}", fold_started)

    if not oof_frames:
        raise ValueError(
            "primary_model.oof could not produce any fold predictions; "
            "lower min_train_events/min_validation_events or disable primary_model.oof.enabled",
        )

    oof_frame = pd.concat(oof_frames, axis=0).sort_index()
    raw_scores = {
        name: expert_score_sum[name] / expert_score_weight[name]
        for name in expert_score_sum
        if expert_score_weight.get(name, 0.0) > 0.0
    }
    min_positive_folds = int(
        weighting_config.get("min_positive_folds", max(1, min(n_splits, len(folds)) // 2 + 1)),
    )
    min_fold_proxy_mean_return = float(
        weighting_config.get(
            "min_fold_proxy_mean_return",
            weighting_config.get("min_proxy_mean_return", 0.0),
        ),
    )
    min_fold_balanced_accuracy = float(
        weighting_config.get(
            "min_fold_balanced_accuracy",
            weighting_config.get("min_balanced_accuracy", 0.5),
        ),
    )
    min_fold_profit_factor = float(
        weighting_config.get(
            "min_fold_profit_factor",
            weighting_config.get("min_profit_factor", 1.0),
        ),
    )
    stable_scores: dict[str, float] = {}
    rejected_unstable_experts: dict[str, dict[str, Any]] = {}
    for name, score in raw_scores.items():
        rows = expert_metric_rows.get(name, [])
        positive_fold_count = sum(
            1
            for row in rows
            if float(row.get("score", 0.0)) > 0.0
            and float(row.get("proxy_mean_return", 0.0)) >= min_fold_proxy_mean_return
            and float(row.get("balanced_accuracy", 0.0)) >= min_fold_balanced_accuracy
            and float(row.get("proxy_profit_factor", 0.0)) >= min_fold_profit_factor
        )
        if positive_fold_count >= min_positive_folds:
            stable_scores[name] = score
        else:
            rejected_unstable_experts[name] = {
                "fold_count": len(rows),
                "positive_fold_count": positive_fold_count,
                "min_positive_folds": min_positive_folds,
                "min_fold_proxy_mean_return": min_fold_proxy_mean_return,
                "min_fold_balanced_accuracy": min_fold_balanced_accuracy,
                "min_fold_profit_factor": min_fold_profit_factor,
            }
    learned_weights = normalize_positive_weights(stable_scores)
    progress_print(
        args,
        f"{symbol}: primary OOF learned_weights={learned_weights} "
        f"rejected_experts={sorted(rejected_unstable_experts)}",
    )
    expert_validation_metrics = {
        name: {
            key: weighted_mean(
                pd.Series([float(row.get(key, 0.0)) for row in rows]),
                pd.Series([float(row.get("count", 1.0)) for row in rows]),
            )
            for key in (
                "score",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "negative_log_loss",
                "proxy_mean_return",
                "proxy_win_rate",
                "proxy_profit_factor",
            )
        }
        for name, rows in expert_metric_rows.items()
    }
    diagnostics = {
        "enabled": True,
        "n_splits_requested": n_splits,
        "folds": fold_summaries,
        "coverage_events": len(oof_frame),
        "coverage_rate": float(len(oof_frame) / len(ordered)) if len(ordered) else 0.0,
        "embargo_bars": embargo_bars,
        "expert_validation_score": raw_scores,
        "expert_stable_score": stable_scores,
        "expert_validation_metrics": expert_validation_metrics,
        "expert_validation_fold_metrics": expert_metric_rows,
        "rejected_unstable_experts": rejected_unstable_experts,
        "learned_expert_weights": learned_weights,
        "all_experts_rejected": not bool(learned_weights),
        "weighting_method": "purged_oof_cost_adjusted_utility",
        "weighting_config": weighting_config,
        "round_trip_cost": float(round_trip_cost),
    }
    progress_done(args, f"{symbol}: primary OOF training", stage_started)
    return oof_frame, learned_weights, diagnostics


def primary_regime_switching_oof_frame_and_gater(
    *,
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    symbol: str,
    round_trip_cost: float,
) -> tuple[pd.DataFrame, dict[str, Any], Any, dict[str, Any]]:
    config = primary_oof_config(strategy_config)
    if not bool(config.get("enabled", True)):
        return pd.DataFrame(), {}, None, {"enabled": False}

    switching_config = primary_regime_switching_config(strategy_config)
    validation_config = section(strategy_config, "validation")
    ordered = pd.DatetimeIndex(train_index).intersection(dataset.X.index).sort_values()
    n_splits = max(2, int(config.get("n_splits", 3)))
    min_train_events = int(config.get("min_train_events", 50))
    min_validation_events = int(config.get("min_validation_events", 10))
    embargo_bars = int(config.get("embargo_bars", args.embargo_bars))
    folds = [
        pd.DatetimeIndex(values)
        for values in np.array_split(ordered.to_numpy(), n_splits)
        if len(values) >= min_validation_events
    ]
    regime_model = fit_regime_state_model(
        dataset.X,
        ordered,
        config=switching_config.get("regime_scoring", switching_config),
    )

    oof_frames: list[pd.DataFrame] = []
    oof_policy_frames: dict[str, list[pd.DataFrame]] = {policy: [] for policy in PRIMARY_POLICY_NAMES}
    fold_summaries: list[dict[str, Any]] = []

    for fold_id, validation_index in enumerate(folds, start=1):
        inner_train = purged_train_index_for_validation(
            dataset,
            ordered,
            validation_index,
            embargo_bars=embargo_bars,
        )
        if len(inner_train) < min_train_events or dataset.y.loc[inner_train].nunique() < 2:
            fold_summaries.append(
                {
                    "fold": fold_id,
                    "status": "skipped_insufficient_train",
                    "train_events": len(inner_train),
                    "validation_events": len(validation_index),
                },
            )
            continue

        policy_models = {
            policy: fit_primary_policy_model(
                policy=policy,
                dataset=dataset,
                train_index=inner_train,
                strategy_config=strategy_config,
                args=args,
                symbol=symbol,
                seed_offset=50_000 + fold_id * 100 + policy_id,
                progress_label=f"{symbol}:primary_oof_{fold_id}:{policy}",
            )
            for policy_id, policy in enumerate(PRIMARY_POLICY_NAMES, start=1)
        }
        train_policy_frames = {
            policy: model.primary_side_frame(dataset.X.loc[inner_train])
            for policy, model in policy_models.items()
        }
        train_regime = regime_model.transform(dataset.X.loc[inner_train])
        fold_edge_table = build_policy_edge_table(
            dataset=dataset,
            regime_frame=train_regime,
            policy_frames=train_policy_frames,
            index=inner_train,
            config=switching_config,
            validation_config=validation_config,
            n_trials=1,
            round_trip_cost=round_trip_cost,
        )
        switch_model = RegimeSwitchingPrimarySideModel(
            trend_model=policy_models["trend_following"],
            reversion_model=policy_models["mean_reversion"],
            regime_model=regime_model,
            policy_edge_table=fold_edge_table,
            abstain_if_no_positive_policy=bool(
                switching_config.get("abstain_if_no_positive_policy", True),
            ),
        )
        fold_frame = switch_model.primary_side_frame(dataset.X.loc[validation_index])
        oof_frames.append(fold_frame)

        validation_policy_frames = {
            policy: model.primary_side_frame(dataset.X.loc[validation_index])
            for policy, model in policy_models.items()
        }
        for policy, frame in validation_policy_frames.items():
            oof_policy_frames[policy].append(frame)

        fold_summaries.append(
            {
                "fold": fold_id,
                "status": "ok",
                "train_events": len(inner_train),
                "validation_events": len(validation_index),
                "validation_start": validation_index.min(),
                "validation_end": validation_index.max(),
                "embargo_bars": embargo_bars,
                "eligible_policy_edges": sum(
                    1 for row in flatten_policy_edge_table(fold_edge_table) if row.get("eligible")
                ),
            },
        )

    if not oof_frames:
        raise ValueError(
            "primary_model.oof could not produce any fold predictions; "
            "lower min_train_events/min_validation_events or disable primary_model.oof.enabled",
        )

    oof_frame = pd.concat(oof_frames, axis=0).sort_index()
    combined_policy_frames = {
        policy: pd.concat(frames, axis=0).sort_index()
        for policy, frames in oof_policy_frames.items()
        if frames
    }
    oof_regime = regime_model.transform(dataset.X.loc[oof_frame.index])
    final_edge_table = build_policy_edge_table(
        dataset=dataset,
        regime_frame=oof_regime,
        policy_frames=combined_policy_frames,
        index=oof_frame.index,
        config=switching_config,
        validation_config=validation_config,
        n_trials=1,
        round_trip_cost=round_trip_cost,
    )
    diagnostics = {
        "enabled": True,
        "mode": "regime_switching_dual_policy",
        "n_splits_requested": n_splits,
        "folds": fold_summaries,
        "coverage_events": len(oof_frame),
        "coverage_rate": float(len(oof_frame) / len(ordered)) if len(ordered) else 0.0,
        "embargo_bars": embargo_bars,
        "regime_model": regime_model.diagnostics(),
        "policy_edge_table": final_edge_table,
        "policy_edge_rows": flatten_policy_edge_table(final_edge_table),
        "all_policies_rejected": not policy_edge_table_has_eligible(final_edge_table),
        "switching_config": switching_config,
    }
    return oof_frame, final_edge_table, regime_model, diagnostics


def train_primary_side_model(
    *,
    close: pd.Series,
    features: pd.DataFrame,
    volatility: pd.Series,
    exit_config: dict[str, float],
    vertical_bars: int,
    window: TrainTestWindow,
    event_config: dict[str, Any],
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    symbol: str,
    round_trip_cost: float = 0.0,
) -> PrimaryTrainingResult:
    rule_model, rule_model_name, rule_model_diagnostics = make_primary_rule_side_model(
        strategy_config=strategy_config,
        event_config=event_config,
    )
    if not primary_model_enabled(strategy_config):
        regime_config = primary_regime_switching_config(strategy_config)
        gating_config = primary_regime_config(strategy_config)
        volatility_low_quantile = float(
            first_not_none(
                gating_config.get("volatility_low_quantile"),
                regime_config.get("volatility_low_quantile"),
                0.33,
            ),
        )
        volatility_high_quantile = float(
            first_not_none(
                gating_config.get("volatility_high_quantile"),
                regime_config.get("volatility_high_quantile"),
                0.67,
            ),
        )
        fit_features = primary_rule_calibration_features(
            features=features,
            volatility=volatility,
            window=window,
            volatility_low_quantile=volatility_low_quantile,
            volatility_high_quantile=volatility_high_quantile,
            strategy_config=strategy_config,
        )
        walk_config = _order_flow_walk_forward_config(strategy_config)
        if rule_model_name == PRIMARY_ORDER_FLOW_SHOCK_RULE_NAME and bool(walk_config.get("enabled", False)):
            frame, walk_forward_diagnostics = primary_side_frame_from_walk_forward_order_flow(
                features=features,
                volatility=volatility,
                window=window,
                strategy_config=strategy_config,
                volatility_low_quantile=volatility_low_quantile,
                volatility_high_quantile=volatility_high_quantile,
            )
            rule_model_diagnostics = primary_rule_side_model_diagnostics(
                rule_model,
                mode=rule_model_name,
                event_config=event_config,
            )
            rule_model_diagnostics["search"] = {"enabled": False, "status": "skipped_walk_forward"}
            rule_model_diagnostics["walk_forward"] = walk_forward_diagnostics
        else:
            rule_model, rule_model_diagnostics = fit_primary_rule_model_with_optional_search(
                rule_model=rule_model,
                rule_model_name=rule_model_name,
                fit_features=fit_features,
                close=close,
                volatility=volatility,
                exit_config=exit_config,
                vertical_bars=vertical_bars,
                window=window,
                event_config=event_config,
                strategy_config=strategy_config,
                args=args,
                round_trip_cost=round_trip_cost,
            )
            rule_model_diagnostics["walk_forward"] = {"enabled": False}
            frame = primary_side_frame_from_rule(
                rule_model,
                features,
                volatility=volatility,
                train_start=window.train_start,
                train_end=window.train_end,
                low_quantile=volatility_low_quantile,
                high_quantile=volatility_high_quantile,
            )
        frame, candidate_filter_summary = apply_primary_candidate_filters(frame, strategy_config)
        candidate_side = frame["primary_side"][frame["primary_side"] != 0]
        diagnostics = {
            "model": rule_model_name,
            "enabled": False,
            "candidate_count": len(candidate_side),
            "side_counts": class_counts(candidate_side) if not candidate_side.empty else {},
            "volatility_state_counts": value_counts_dict(
                frame.get("volatility_state", pd.Series(dtype=object)),
            ),
            "candidate_filter": candidate_filter_summary,
            "event_definition": event_config,
            "rule_model": rule_model_diagnostics,
        }
        return PrimaryTrainingResult(
            model=rule_model,
            dataset=None,
            frame=frame,
            train_index=pd.DatetimeIndex([]),
            test_index=pd.DatetimeIndex([]),
            diagnostics=diagnostics,
        )

    primary_stage_started = progress_start(args, f"{symbol}: primary training pipeline")
    dataset_started = progress_start(args, f"{symbol}: primary dataset build")
    dataset = build_primary_side_dataset(
        close=close,
        features=features,
        volatility=volatility,
        exit_config=exit_config,
        vertical_bars=vertical_bars,
        window=window,
        args=args,
    )
    progress_done(args, f"{symbol}: primary dataset build", dataset_started)
    train_index, test_index = split_dataset_indices_for_window(dataset, window)
    progress_print(
        args,
        f"{symbol}: primary dataset labels={len(dataset.y):,} "
        f"features={dataset.X.shape[1]:,} train={len(train_index):,} "
        f"test={len(test_index):,} classes={class_counts(dataset.y)}",
    )
    mode = str(primary_model_config(strategy_config).get("mode", PRIMARY_MOE_MODEL_NAME))
    information_gate = primary_information_gate_config(strategy_config)
    final_model_skipped = False
    final_model_skip_reason: str | None = None
    if mode == "regime_switching_dual_policy":
        oof_frame, policy_edge_table, regime_model, oof_diagnostics = (
            primary_regime_switching_oof_frame_and_gater(
                dataset=dataset,
                train_index=train_index,
                strategy_config=strategy_config,
                args=args,
                symbol=symbol,
                round_trip_cost=round_trip_cost,
            )
        )
        early_reject = bool(information_gate.get("enabled", False)) and bool(
            oof_diagnostics.get("all_policies_rejected", False),
        )
        if early_reject:
            final_model_skipped = True
            final_model_skip_reason = "primary_information_gate_rejected_oof_all_policies"
            progress_print(
                args,
                f"{symbol}: skipping primary final fit because OOF rejected all policies",
            )
            model = rule_model
            frame = abstain_primary_side_frame(
                dataset.X.index,
                reason="primary_information_gate_rejected",
            )
        else:
            if regime_model is None:
                regime_model = fit_regime_state_model(
                    dataset.X,
                    train_index,
                    config=primary_regime_switching_config(strategy_config),
                )
            policy_models = {
                policy: fit_primary_policy_model(
                    policy=policy,
                    dataset=dataset,
                    train_index=train_index,
                    strategy_config=strategy_config,
                    args=args,
                    symbol=symbol,
                    seed_offset=70_000 + policy_id,
                    progress_label=f"{symbol}:primary_final:{policy}",
                )
                for policy_id, policy in enumerate(PRIMARY_POLICY_NAMES, start=1)
            }
            model = RegimeSwitchingPrimarySideModel(
                trend_model=policy_models["trend_following"],
                reversion_model=policy_models["mean_reversion"],
                regime_model=regime_model,
                policy_edge_table=policy_edge_table,
                abstain_if_no_positive_policy=bool(
                    primary_regime_switching_config(strategy_config).get(
                        "abstain_if_no_positive_policy",
                        True,
                    ),
                ),
            )
            final_predict_started = progress_start(args, f"{symbol}: primary final predict full dataset")
            frame = model.primary_side_frame(dataset.X)
            progress_done(args, f"{symbol}: primary final predict full dataset", final_predict_started)
    else:
        oof_frame, learned_weights, oof_diagnostics = primary_oof_frame_and_weights(
            dataset=dataset,
            train_index=train_index,
            strategy_config=strategy_config,
            args=args,
            symbol=symbol,
            round_trip_cost=round_trip_cost,
        )
        early_reject = bool(information_gate.get("enabled", False)) and bool(
            oof_diagnostics.get("all_experts_rejected", False),
        )
        if early_reject:
            final_model_skipped = True
            final_model_skip_reason = "primary_information_gate_rejected_oof_all_experts"
            progress_print(
                args,
                f"{symbol}: skipping primary final fit because OOF rejected all experts",
            )
            model = rule_model
            frame = abstain_primary_side_frame(
                dataset.X.index,
                reason="primary_information_gate_rejected",
            )
        else:
            final_weights = learned_weights or None
            final_scores = (
                oof_diagnostics.get("expert_stable_score", {})
                if isinstance(oof_diagnostics, dict)
                else {}
            )
            final_metrics = (
                oof_diagnostics.get("expert_validation_metrics", {})
                if isinstance(oof_diagnostics, dict)
                else {}
            )
            model = make_multi_expert_primary_model(
                strategy_config=strategy_config,
                args=args,
                rule_model=rule_model,
                expert_weights=final_weights,
                expert_scores=final_scores if isinstance(final_scores, dict) else None,
                expert_metrics=final_metrics if isinstance(final_metrics, dict) else None,
                seed_offset=40_000,
                progress_label=f"{symbol}:primary_final",
            )
            model = fit_afml_side_model(
                model,
                dataset,
                train_index,
                progress_label=f"{symbol}:primary_final",
                args=args,
            )
            final_predict_started = progress_start(args, f"{symbol}: primary final predict full dataset")
            frame = model.primary_side_frame(dataset.X)
            progress_done(args, f"{symbol}: primary final predict full dataset", final_predict_started)
    if not oof_frame.empty:
        frame.loc[oof_frame.index, oof_frame.columns] = oof_frame
        missing_oof_train = pd.DatetimeIndex(train_index).difference(oof_frame.index)
        if not missing_oof_train.empty:
            frame.loc[missing_oof_train, "primary_side"] = 0
            frame.loc[missing_oof_train, "primary_abstain_reason"] = "oof_unavailable"
    frame, candidate_filter_summary = apply_primary_candidate_filters(frame, strategy_config)
    information_gate = primary_information_gate_config(strategy_config)
    primary_rejected = bool(information_gate.get("enabled", False)) and bool(
        oof_diagnostics.get("all_experts_rejected", False)
        or oof_diagnostics.get("all_policies_rejected", False),
    )
    if primary_rejected:
        frame["primary_side"] = 0
        frame["primary_abstain_reason"] = "primary_information_gate_rejected"
    progress_print(
        args,
        f"{symbol}: primary information gate enabled={bool(information_gate.get('enabled', False))} "
        f"rejected={primary_rejected}",
    )

    oof_index = pd.DatetimeIndex(oof_frame.index).intersection(dataset.y.index)
    oof_metrics = (
        classification_metrics(dataset.y.loc[oof_index], frame.loc[oof_index, "primary_side"], positive_label=1)
        if not oof_index.empty
        else {}
    )
    test_metrics = classification_metrics(
        dataset.y.loc[test_index],
        frame.loc[test_index, "primary_side"],
        positive_label=1,
    )
    importance = primary_feature_importance_diagnostics(
        model=model,
        dataset=dataset,
        test_index=test_index,
        strategy_config=strategy_config,
        output_dir=output_dir,
        symbol=symbol,
        args=args,
    )
    final_model_diagnostics = (
        {
            "status": "skipped",
            "reason": final_model_skip_reason,
            "skipped_by_information_gate": True,
        }
        if final_model_skipped
        else model.diagnostics()
    )
    diagnostics = {
        "model": mode,
        "enabled": True,
        "label_counts": class_counts(dataset.y),
        "n_labels": len(dataset.y),
        "train_events": len(train_index),
        "test_events": len(test_index),
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "activity": {
            "all": primary_frame_activity_summary(frame, pd.DatetimeIndex(frame.index)),
            "train": primary_frame_activity_summary(frame, train_index),
            "test": primary_frame_activity_summary(frame, test_index),
        },
        "primary_information_audit": {
            "oof": primary_information_audit(dataset=dataset, frame=frame, index=oof_index),
            "test": primary_information_audit(dataset=dataset, frame=frame, index=test_index),
        },
        "information_gate": {
            "enabled": bool(information_gate.get("enabled", False)),
            "rejected": primary_rejected,
            "reason": "no_positive_policy_edge" if primary_rejected and mode == "regime_switching_dual_policy" else (
                "all_experts_below_information_threshold" if primary_rejected else None
            ),
        },
        "candidate_filter": candidate_filter_summary,
        "oof": oof_diagnostics,
        "final_model": final_model_diagnostics,
        "final_model_skipped": final_model_skipped,
        "feature_importance": importance,
        "regime_policy_audit": oof_diagnostics.get("policy_edge_rows", []),
        "event_definition": event_config,
        "sample_weight_summary": {
            "all": sample_weight_summary(dataset.sample_weight),
            "train": sample_weight_summary(dataset.sample_weight.loc[train_index]),
            "test": sample_weight_summary(dataset.sample_weight.loc[test_index]),
        },
    }
    diagnostics["primary_failure_reasons"] = primary_failure_reasons(
        diagnostics,
        config=information_gate,
    )
    diagnostics["information_gate"]["failure_reasons"] = diagnostics["primary_failure_reasons"]
    progress_done(args, f"{symbol}: primary training pipeline", primary_stage_started)
    return PrimaryTrainingResult(
        model=model,
        dataset=dataset,
        frame=frame,
        train_index=train_index,
        test_index=test_index,
        diagnostics=diagnostics,
    )


def fmt_metric(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def metrics_brief(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "n/a"
    return (
        f"accuracy={fmt_metric(metrics.get('accuracy'))} "
        f"precision={fmt_metric(metrics.get('positive_precision'))} "
        f"recall={fmt_metric(metrics.get('positive_recall'))} "
        f"f1={fmt_metric(metrics.get('positive_f1'))} "
        f"weighted_f1={fmt_metric(metrics.get('weighted_f1'))}"
    )


def print_primary_stage_summary(symbol: str, diagnostics: dict[str, Any]) -> None:
    activity = diagnostics.get("activity", {}) if isinstance(diagnostics.get("activity"), dict) else {}
    all_activity = activity.get("all", {}) if isinstance(activity.get("all"), dict) else {}
    test_activity = activity.get("test", {}) if isinstance(activity.get("test"), dict) else {}
    final_model = diagnostics.get("final_model", {}) if isinstance(diagnostics.get("final_model"), dict) else {}
    oof = diagnostics.get("oof", {}) if isinstance(diagnostics.get("oof"), dict) else {}
    print(
        f"{symbol}: PRIMARY SUMMARY "
        f"labels={diagnostics.get('label_counts', {})} "
        f"oof=[{metrics_brief(diagnostics.get('oof_metrics', {}))}] "
        f"test=[{metrics_brief(diagnostics.get('test_metrics', {}))}]",
        flush=True,
    )
    print(
        f"{symbol}: PRIMARY ACTIVITY "
        f"candidates={all_activity.get('candidate_count', 0):,} "
        f"abstain_rate={fmt_metric(all_activity.get('abstain_rate'))} "
        f"test_candidates={test_activity.get('candidate_count', 0):,} "
        f"mean_confidence={fmt_metric(all_activity.get('mean_confidence'))} "
        f"mean_disagreement={fmt_metric(all_activity.get('mean_disagreement'))}",
        flush=True,
    )
    if final_model or oof:
        print(
            f"{symbol}: PRIMARY EXPERTS "
            f"weights={final_model.get('expert_weights', {})} "
            f"drivers={all_activity.get('driver_expert_counts', {})} "
            f"oof_coverage={fmt_metric(oof.get('coverage_rate'))}",
            flush=True,
        )
    if diagnostics.get("model") == "regime_switching_dual_policy":
        print(
            f"{symbol}: PRIMARY REGIME "
            f"policies={all_activity.get('policy_counts', {})} "
            f"states={all_activity.get('market_state_counts', {})} "
            f"eligible_edges={sum(1 for row in diagnostics.get('regime_policy_audit', []) if row.get('eligible'))}",
            flush=True,
        )
    audit = diagnostics.get("primary_information_audit", {})
    test_audit = audit.get("test", {}) if isinstance(audit, dict) and isinstance(audit.get("test"), dict) else {}
    info_gate = diagnostics.get("information_gate", {})
    if test_audit:
        print(
            f"{symbol}: PRIMARY INFORMATION "
            f"active_rate={fmt_metric(test_audit.get('active_rate'))} "
            f"proxy_mean_ret={fmt_metric(test_audit.get('proxy_mean_return'))} "
            f"proxy_win_rate={fmt_metric(test_audit.get('proxy_win_rate'))} "
            f"gate_rejected={bool(info_gate.get('rejected', False))}",
            flush=True,
        )


def print_meta_stage_summary(
    symbol: str,
    *,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    accepted_count: int,
    signal_counts_: dict[str, int],
) -> None:
    print(
        f"{symbol}: META SUMMARY "
        f"train=[{metrics_brief(train_metrics)}] "
        f"test=[{metrics_brief(test_metrics)}] "
        f"accepted={accepted_count:,} signals={signal_counts_}",
        flush=True,
    )


def print_validation_stage_summary(symbol: str, validation: dict[str, Any], *, label: str) -> None:
    sharpe = validation.get("sharpe", {}) if isinstance(validation.get("sharpe"), dict) else {}
    failure = (
        validation.get("strategy_failure", {})
        if isinstance(validation.get("strategy_failure"), dict)
        else {}
    )
    cpcv = validation.get("cpcv", {}) if isinstance(validation.get("cpcv"), dict) else {}
    print(
        f"{symbol}: {label} AFML "
        f"sharpe={fmt_metric(sharpe.get('annualized_sharpe'))} "
        f"psr={fmt_metric(sharpe.get('probabilistic_sharpe_ratio'))} "
        f"dsr={fmt_metric(sharpe.get('deflated_sharpe_ratio'))} "
        f"failure_prob={fmt_metric(failure.get('failure_probability'))} "
        f"cpcv_splits={cpcv.get('n_splits', 0)} "
        f"cpcv_coverage={cpcv.get('test_coverage_min', 'n/a')}..{cpcv.get('test_coverage_max', 'n/a')}",
        flush=True,
    )


def print_sample_weight_stage_summary(symbol: str, weights: dict[str, Any], *, label: str) -> None:
    train = weights.get("train", {}) if isinstance(weights.get("train"), dict) else {}
    test = weights.get("test", {}) if isinstance(weights.get("test"), dict) else {}
    print(
        f"{symbol}: {label} SAMPLE WEIGHTS "
        f"train_count={train.get('count', 0):,} train_mean={fmt_metric(train.get('mean'))} "
        f"train_max={fmt_metric(train.get('max'))} "
        f"test_count={test.get('count', 0):,} test_mean={fmt_metric(test.get('mean'))} "
        f"test_max={fmt_metric(test.get('max'))}",
        flush=True,
    )


def signal_proxy_performance(
    *,
    meta_dataset: AfmlDataset,
    test_signals: pd.DataFrame,
    validation_config: dict[str, Any],
    n_trials: int,
    round_trip_cost: float = 0.0,
) -> dict[str, Any]:
    active_index = pd.DatetimeIndex(
        test_signals.index[test_signals["signal"].astype(int) != 0],
    ).intersection(meta_dataset.labels.index)
    raw_returns = meta_dataset.labels.loc[active_index, "ret"].dropna().astype(float)
    returns = (raw_returns - float(round_trip_cost)).rename("net_ret")
    event_ends = pd.Series(pd.NaT, index=active_index, dtype="datetime64[ns, UTC]")
    events = getattr(meta_dataset, "events", pd.DataFrame(index=active_index))
    if isinstance(events, pd.DataFrame) and "t1" in events:
        event_ends = pd.to_datetime(
            events.loc[active_index, "t1"],
            utc=True,
            errors="coerce",
        )
    position_proxy = position_aware_prediction_summary(
        pd.Series(1, index=returns.index, dtype=int),
        returns,
        event_ends=event_ends.reindex(returns.index),
    )
    if returns.empty:
        return {
            "return_basis": "cost_adjusted_label_ret",
            "round_trip_cost": float(round_trip_cost),
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "gross_profit_return": 0.0,
            "gross_loss_return": 0.0,
            "profit_factor": None,
            "mean_return": None,
            "median_return": None,
            "position_aware": position_proxy,
            "sharpe": sharpe_ratio_diagnostics(returns, n_trials=n_trials),
            "validation": afml_validation_diagnostics(
                returns,
                validation_config=validation_config,
                n_trials=n_trials,
            ),
        }

    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    raw_wins = raw_returns[raw_returns > 0.0]
    raw_losses = raw_returns[raw_returns < 0.0]
    raw_gross_profit = float(raw_wins.sum())
    raw_gross_loss = float(-raw_losses.sum())
    return {
        "return_basis": "cost_adjusted_label_ret",
        "round_trip_cost": float(round_trip_cost),
        "trade_count": len(returns),
        "win_count": len(wins),
        "loss_count": len(losses),
        "flat_count": int((returns == 0.0).sum()),
        "win_rate": float(len(wins) / len(returns)) if len(returns) else 0.0,
        "gross_profit_return": gross_profit,
        "gross_loss_return": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "raw_win_count": len(raw_wins),
        "raw_loss_count": len(raw_losses),
        "raw_gross_profit_return": raw_gross_profit,
        "raw_gross_loss_return": raw_gross_loss,
        "raw_profit_factor": raw_gross_profit / raw_gross_loss if raw_gross_loss > 0.0 else None,
        "raw_mean_return": float(raw_returns.mean()),
        "raw_median_return": float(raw_returns.median()),
        "position_aware": position_proxy,
        "sharpe": sharpe_ratio_diagnostics(returns, n_trials=n_trials),
        "validation": afml_validation_diagnostics(
            returns,
            validation_config=validation_config,
            n_trials=n_trials,
        ),
    }


def return_stream_summary(
    returns: pd.Series,
    *,
    validation_config: dict[str, Any],
    n_trials: int,
    return_basis: str = "cost_adjusted_label_ret",
    round_trip_cost: float | None = None,
) -> dict[str, Any]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "return_basis": return_basis,
            "round_trip_cost": round_trip_cost,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "flat_count": 0,
            "win_rate": 0.0,
            "gross_profit_return": 0.0,
            "gross_loss_return": 0.0,
            "profit_factor": None,
            "mean_return": None,
            "median_return": None,
            "sharpe": None,
            "psr": None,
            "dsr": None,
            "failure_probability": None,
        }

    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    sharpe = sharpe_ratio_diagnostics(returns, n_trials=n_trials)
    validation = afml_validation_diagnostics(
        returns,
        validation_config=validation_config,
        n_trials=n_trials,
    )
    failure = (
        validation.get("strategy_failure", {})
        if isinstance(validation.get("strategy_failure"), dict)
        else {}
    )
    return {
        "return_basis": return_basis,
        "round_trip_cost": round_trip_cost,
        "trade_count": len(returns),
        "win_count": len(wins),
        "loss_count": len(losses),
        "flat_count": int((returns == 0.0).sum()),
        "win_rate": float(len(wins) / len(returns)) if len(returns) else 0.0,
        "gross_profit_return": gross_profit,
        "gross_loss_return": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "sharpe": sharpe.get("annualized_sharpe"),
        "psr": sharpe.get("probabilistic_sharpe_ratio"),
        "dsr": sharpe.get("deflated_sharpe_ratio"),
        "failure_probability": failure.get("failure_probability"),
    }


def annotate_signal_outcomes(
    signals: pd.DataFrame,
    meta_dataset: AfmlDataset,
    *,
    round_trip_cost: float,
) -> pd.DataFrame:
    out = signals.copy()
    index = pd.DatetimeIndex(out.index)
    labels = meta_dataset.labels.reindex(index)
    events = meta_dataset.events.reindex(index)
    ret = labels["ret"] if "ret" in labels else pd.Series(np.nan, index=index)
    bin_ = labels["bin"] if "bin" in labels else pd.Series(pd.NA, index=index)
    t1 = events["t1"] if "t1" in events else pd.Series(pd.NaT, index=index)
    out["label_ret"] = pd.to_numeric(ret, errors="coerce")
    out["label_cost"] = float(round_trip_cost)
    out["label_net_ret"] = out["label_ret"] - float(round_trip_cost)
    out["meta_label"] = pd.to_numeric(bin_, errors="coerce").astype("Int64")
    out["cost_adjusted_meta_label"] = pd.Series(
        np.where(out["label_net_ret"].notna(), (out["label_net_ret"] > 0.0).astype(int), pd.NA),
        index=out.index,
    ).astype("Int64")
    out["event_end_time"] = pd.to_datetime(t1, utc=True, errors="coerce").astype(str)
    if "barrier" in labels:
        barrier = labels["barrier"]
    elif "barrier" in events:
        barrier = events["barrier"]
    else:
        barrier = pd.Series(pd.NA, index=index)
    out["barrier_outcome"] = barrier.astype("string")
    return out


def _utc_session(hour: int) -> str:
    if hour < 8:
        return "00-07_utc"
    if hour < 16:
        return "08-15_utc"
    return "16-23_utc"


def _add_decile_column(frame: pd.DataFrame, source_column: str, output_column: str) -> None:
    frame[output_column] = "missing"
    if source_column not in frame:
        return
    values = pd.to_numeric(frame[source_column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return
    if len(valid) < 2:
        frame.loc[valid.index, output_column] = "d0"
        return
    q = min(10, len(valid))
    try:
        labels = [f"d{i}" for i in range(q)]
        deciles = pd.qcut(valid.rank(method="first"), q=q, labels=labels)
    except ValueError:
        frame.loc[valid.index, output_column] = "d0"
        return
    frame.loc[valid.index, output_column] = deciles.astype(str)


def _json_group_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "missing"
        return f"{numeric:g}"
    text = str(value)
    return text if text else "missing"


def grouped_diagnostic_return_report(
    frame: pd.DataFrame,
    *,
    group_column: str,
    return_column: str,
    validation_config: dict[str, Any],
    n_trials: int,
    return_basis: str,
    round_trip_cost: float | None,
) -> list[dict[str, Any]]:
    if group_column not in frame or return_column not in frame:
        return []
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_column, dropna=False, sort=True):
        row = return_stream_summary(
            pd.to_numeric(group[return_column], errors="coerce"),
            validation_config=validation_config,
            n_trials=n_trials,
            return_basis=return_basis,
            round_trip_cost=round_trip_cost,
        )
        row["group"] = _json_group_value(value)
        rows.append(row)
    return rows


def diagnostic_failure_report(
    test_signals: pd.DataFrame,
    *,
    validation_config: dict[str, Any],
    n_trials: int,
    round_trip_cost: float = 0.0,
) -> dict[str, Any]:
    if "diagnostic_signal" not in test_signals:
        return {"status": "missing_diagnostic_signal"}
    if "label_ret" not in test_signals:
        return {"status": "missing_label_returns"}

    frame = test_signals.copy()
    diagnostic_signal = frame["diagnostic_signal"].fillna(0).astype(int)
    active = diagnostic_signal != 0
    if not bool(active.any()):
        return {
            "status": "no_diagnostic_trades",
            "diagnostic_signal_counts": class_counts(diagnostic_signal),
        }

    frame = frame.loc[active].copy()
    if "label_net_ret" not in frame:
        frame["label_net_ret"] = pd.to_numeric(frame["label_ret"], errors="coerce") - float(round_trip_cost)
    frame["event_hour_utc"] = pd.DatetimeIndex(frame.index).hour
    frame["event_session_utc"] = [_utc_session(int(hour)) for hour in frame["event_hour_utc"]]
    _add_decile_column(frame, "meta_probability", "meta_probability_decile")
    _add_decile_column(frame, "primary_confidence", "primary_confidence_decile")
    _add_decile_column(frame, "primary_disagreement", "primary_disagreement_decile")
    _add_decile_column(frame, "trgt", "trgt_decile")

    group_specs = {
        "by_primary_driver_expert": "primary_driver_expert",
        "by_primary_policy": "primary_policy",
        "by_primary_market_state": "primary_market_state",
        "by_primary_side": "primary_side",
        "by_meta_probability_decile": "meta_probability_decile",
        "by_primary_confidence_decile": "primary_confidence_decile",
        "by_primary_disagreement_decile": "primary_disagreement_decile",
        "by_trgt_decile": "trgt_decile",
        "by_event_hour_utc": "event_hour_utc",
        "by_event_session_utc": "event_session_utc",
    }
    report: dict[str, Any] = {
        "status": "ok",
        "return_basis": "cost_adjusted_label_ret",
        "round_trip_cost": float(round_trip_cost),
        "overall": return_stream_summary(
            pd.to_numeric(frame["label_net_ret"], errors="coerce"),
            validation_config=validation_config,
            n_trials=n_trials,
            return_basis="cost_adjusted_label_ret",
            round_trip_cost=float(round_trip_cost),
        ),
        "raw_overall": return_stream_summary(
            pd.to_numeric(frame["label_ret"], errors="coerce"),
            validation_config=validation_config,
            n_trials=n_trials,
            return_basis="raw_label_ret",
            round_trip_cost=None,
        ),
        "diagnostic_signal_counts": class_counts(diagnostic_signal),
    }
    for output_key, column in group_specs.items():
        report[output_key] = grouped_diagnostic_return_report(
            frame,
            group_column=column,
            return_column="label_net_ret",
            validation_config=validation_config,
            n_trials=n_trials,
            return_basis="cost_adjusted_label_ret",
            round_trip_cost=float(round_trip_cost),
        )
        report[f"raw_{output_key}"] = grouped_diagnostic_return_report(
            frame,
            group_column=column,
            return_column="label_ret",
            validation_config=validation_config,
            n_trials=n_trials,
            return_basis="raw_label_ret",
            round_trip_cost=None,
        )
    return report


def _worst_groups(rows: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("mean_return") or 0.0),
            -int(row.get("trade_count", 0)),
        ),
    )[:limit]


def print_diagnostic_failure_report(symbol: str, report: dict[str, Any]) -> None:
    if report.get("status") != "ok":
        print(f"{symbol}: DIAGNOSTIC FAILURE REPORT status={report.get('status')}", flush=True)
        return
    overall = report.get("overall", {}) if isinstance(report.get("overall"), dict) else {}
    drivers = _worst_groups(report.get("by_primary_driver_expert", []))
    policies = _worst_groups(report.get("by_primary_policy", []))
    states = _worst_groups(report.get("by_primary_market_state", []))
    sides = _worst_groups(report.get("by_primary_side", []))
    print(
        f"{symbol}: DIAGNOSTIC FAILURE OVERALL "
        f"basis={report.get('return_basis')} "
        f"cost={fmt_metric(report.get('round_trip_cost'))} "
        f"trades={overall.get('trade_count', 0):,} "
        f"win_rate={fmt_metric(overall.get('win_rate'))} "
        f"profit_factor={fmt_metric(overall.get('profit_factor'))} "
        f"mean_ret={fmt_metric(overall.get('mean_return'))} "
        f"sharpe={fmt_metric(overall.get('sharpe'))} "
        f"dsr={fmt_metric(overall.get('dsr'))} "
        f"failure_prob={fmt_metric(overall.get('failure_probability'))}",
        flush=True,
    )
    print(
        f"{symbol}: DIAGNOSTIC FAILURE WORST "
        f"drivers={drivers} policies={policies} states={states} sides={sides}",
        flush=True,
    )


def print_signal_proxy_summary(symbol: str, proxy: dict[str, Any], *, label: str = "SIGNAL PROXY") -> None:
    sharpe = proxy.get("sharpe", {}) if isinstance(proxy.get("sharpe"), dict) else {}
    validation = proxy.get("validation", {}) if isinstance(proxy.get("validation"), dict) else {}
    failure = (
        validation.get("strategy_failure", {})
        if isinstance(validation.get("strategy_failure"), dict)
        else {}
    )
    print(
        f"{symbol}: {label} "
        f"basis={proxy.get('return_basis', 'raw_label_ret')} "
        f"cost={fmt_metric(proxy.get('round_trip_cost'))} "
        f"trades={proxy.get('trade_count', 0):,} "
        f"wins={proxy.get('win_count', 0):,} losses={proxy.get('loss_count', 0):,} "
        f"gross_profit={fmt_metric(proxy.get('gross_profit_return'))} "
        f"gross_loss={fmt_metric(proxy.get('gross_loss_return'))} "
        f"profit_factor={fmt_metric(proxy.get('profit_factor'))} "
        f"mean_ret={fmt_metric(proxy.get('mean_return'))} "
        f"sharpe={fmt_metric(sharpe.get('annualized_sharpe'))} "
        f"psr={fmt_metric(sharpe.get('probabilistic_sharpe_ratio'))} "
        f"dsr={fmt_metric(sharpe.get('deflated_sharpe_ratio'))} "
        f"failure_prob={fmt_metric(failure.get('failure_probability'))}",
        flush=True,
    )


def add_diagnostic_bet_size_columns(
    signals: pd.DataFrame,
    *,
    step_size: float | None,
    min_abs_size: float,
    max_abs_size: float,
    probability_status: str = "uncalibrated_probability",
) -> pd.DataFrame:
    if "diagnostic_signal" not in signals.columns:
        return signals
    diagnostic_sized = make_bet_size_frame(
        signals,
        probability_col="meta_probability",
        probability_status=probability_status,
        signal_col="diagnostic_signal",
        min_abs_size=min_abs_size,
        max_abs_size=max_abs_size,
        step_size=step_size,
    )
    out = signals.copy()
    out["diagnostic_bet_size"] = diagnostic_sized["bet_size"]
    out["diagnostic_bet_size_abs"] = diagnostic_sized["bet_size_abs"]
    out["diagnostic_bet_size_probability_status"] = diagnostic_sized["bet_size_probability_status"]
    return out


def apply_afml_gate_to_signal_frame(signals: pd.DataFrame, afml_gate: dict[str, Any]) -> pd.DataFrame:
    out = signals.copy()
    approved = bool(afml_gate.get("approved_for_backtest", False))
    out["gate_status"] = str(afml_gate.get("approval_status", "unknown"))
    out["approved_trading"] = approved
    if approved:
        out["signal_source"] = np.where(out["signal"].astype(int) != 0, "approved_signal", "none")
        return out

    out["signal"] = 0
    if "meta_prediction" in out:
        out["meta_prediction"] = 0
    for column in ("bet_size", "bet_size_abs", "bet_size_raw", "bet_size_raw_abs"):
        if column in out:
            out[column] = 0.0
    if "approved_threshold" in out:
        out["approved_threshold"] = np.nan
    if "approved_upper_threshold" in out:
        out["approved_upper_threshold"] = np.nan
    if "diagnostic_signal" in out:
        out["signal_source"] = np.where(
            out["diagnostic_signal"].astype(int) != 0,
            "diagnostic_signal",
            "none",
        )
    else:
        out["signal_source"] = "none"
    return out


def meta_gate_collapse_summary(
    prediction: pd.Series,
    *,
    max_accepted_rate: float,
) -> dict[str, Any]:
    prediction = prediction.dropna().astype(int)
    if prediction.empty:
        return {
            "collapsed": True,
            "reason": "empty_prediction",
            "accepted_rate": 0.0,
            "counts": {},
        }
    counts = class_counts(prediction)
    accepted_rate = float((prediction == 1).mean())
    single_class = bool((prediction == prediction.iloc[0]).all())
    allow_full_acceptance = float(max_accepted_rate) >= 1.0
    collapsed = (single_class and not allow_full_acceptance) or accepted_rate > max_accepted_rate
    reason = None
    if single_class and not allow_full_acceptance:
        reason = "single_class_prediction"
    elif accepted_rate > max_accepted_rate:
        reason = "accepted_rate_above_limit"
    return {
        "collapsed": bool(collapsed),
        "reason": reason,
        "accepted_rate": accepted_rate,
        "max_accepted_rate": float(max_accepted_rate),
        "counts": counts,
    }


def validation_value(summary: dict[str, Any], *path: str) -> Any:
    current: Any = summary
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def afml_acceptance_gate_summary(  # noqa: C901
    *,
    validation_config: dict[str, Any],
    meta_test_metrics: dict[str, Any],
    primary_before_meta_metrics: dict[str, Any],
    meta_collapse: dict[str, Any],
    signal_proxy: dict[str, Any],
    active_signal_count: int,
    threshold_selection: dict[str, Any] | None,
    primary_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = validation_config.get("acceptance_gate", {})
    config = config if isinstance(config, dict) else {}
    if not bool(config.get("enabled", True)):
        return {
            "enabled": False,
            "approved_for_backtest": True,
            "approval_status": "gate_disabled",
            "blockers": [],
            "warnings": [],
        }
    blockers: list[str] = []
    warnings: list[str] = []
    if bool(meta_collapse.get("collapsed", False)):
        blockers.append("collapsed_meta_gate")
    min_precision_lift = float(config.get("min_precision_lift", 0.0))
    min_f1_lift = float(config.get("min_f1_lift", 0.0))
    precision_lift = float(meta_test_metrics.get("positive_precision", 0.0)) - float(
        primary_before_meta_metrics.get("positive_precision", 0.0),
    )
    f1_lift = float(meta_test_metrics.get("positive_f1", 0.0)) - float(
        primary_before_meta_metrics.get("positive_f1", 0.0),
    )
    if precision_lift <= min_precision_lift:
        blockers.append("meta_precision_not_above_primary_baseline")
    block_f1_lift = bool(config.get("block_f1_lift", True))
    if f1_lift <= min_f1_lift:
        if block_f1_lift:
            blockers.append("meta_f1_not_above_primary_baseline")
        else:
            warnings.append("meta_f1_not_above_primary_baseline")
    profit_factor = signal_proxy.get("profit_factor")
    min_profit_factor = float(config.get("min_proxy_profit_factor", 1.0))
    if profit_factor is None or float(profit_factor) <= min_profit_factor:
        blockers.append("signal_proxy_profit_factor_below_threshold")
    position_proxy_config = config.get("position_proxy", {})
    if isinstance(position_proxy_config, dict) and bool(position_proxy_config.get("enabled", False)):
        position_proxy = signal_proxy.get("position_aware", {})
        position_proxy = position_proxy if isinstance(position_proxy, dict) else {}
        min_positions = int(position_proxy_config.get("min_positions", 1))
        if int(position_proxy.get("trade_count", 0)) < min_positions:
            blockers.append("position_proxy_count_below_minimum")
        min_position_win_rate = position_proxy_config.get("min_win_rate")
        if min_position_win_rate is not None:
            position_win_rate = position_proxy.get("win_rate")
            if position_win_rate is None or float(position_win_rate) < float(min_position_win_rate):
                blockers.append("position_proxy_win_rate_below_threshold")
        min_position_profit_factor = float(
            position_proxy_config.get("min_profit_factor", min_profit_factor),
        )
        position_profit_factor = position_proxy.get("profit_factor")
        if position_profit_factor is None or float(position_profit_factor) <= min_position_profit_factor:
            blockers.append("position_proxy_profit_factor_below_threshold")
        min_position_mean = float(position_proxy_config.get("min_mean_return", 0.0))
        position_mean = position_proxy.get("mean_return")
        if position_mean is None or float(position_mean) <= min_position_mean:
            blockers.append("position_proxy_mean_return_below_threshold")
    min_active_signals = int(config.get("min_active_signals", 30))
    if int(active_signal_count) < min_active_signals:
        blockers.append("active_signal_count_below_minimum")
    signal_validation = signal_proxy.get("validation", {}) if isinstance(signal_proxy.get("validation"), dict) else {}
    dsr = validation_value(signal_validation, "sharpe", "deflated_sharpe_ratio")
    failure_probability = validation_value(signal_validation, "strategy_failure", "failure_probability")
    if bool(config.get("block_dsr_zero", True)) and (dsr is None or float(dsr) <= 0.0):
        blockers.append("deflated_sharpe_is_zero")
    max_failure_probability = float(config.get("max_failure_probability", 0.95))
    if failure_probability is None or float(failure_probability) >= max_failure_probability:
        blockers.append("strategy_failure_probability_too_high")
    if threshold_selection and bool(threshold_selection.get("enabled", False)):
        decision = (
            threshold_selection.get("threshold_decision", {})
            if isinstance(threshold_selection.get("threshold_decision"), dict)
            else {}
        )
        if threshold_selection.get("approved_threshold") is None:
            blockers.append("no_eligible_meta_threshold")
        if not bool(threshold_selection.get("selected_passes_constraints", False)):
            blockers.append("threshold_selection_failed_constraints")
        if decision.get("status") == "diagnostic_only":
            blockers.append("no_eligible_meta_threshold")
    primary_diagnostics = primary_diagnostics if isinstance(primary_diagnostics, dict) else {}
    primary_reasons = primary_diagnostics.get("primary_failure_reasons", [])
    primary_reasons = primary_reasons if isinstance(primary_reasons, list) else []
    if primary_reasons:
        if any(
            reason
            in {
                "weak_active_edge",
                "excess_disagreement",
                "all_experts_below_information_threshold",
            }
            for reason in primary_reasons
        ):
            blockers.append("weak_primary_information")
        if any(
            reason
            in {
                "driver_negative_proxy_pnl",
                "side_instability",
                "unstable_expert_folds",
            }
            for reason in primary_reasons
        ):
            blockers.append("unstable_driver_edge")
    blockers = sorted(set(blockers))
    strict_psr = float(config.get("strict_psr", 0.95))
    strict_dsr = float(config.get("strict_dsr", 0.95))
    strict_failure = float(config.get("strict_failure_probability", 0.05))
    psr = validation_value(signal_validation, "sharpe", "probabilistic_sharpe_ratio")
    if not blockers:
        strict_pass = (
            psr is not None
            and dsr is not None
            and failure_probability is not None
            and float(psr) >= strict_psr
            and float(dsr) >= strict_dsr
            and float(failure_probability) <= strict_failure
        )
        approval_status = "approved_strict" if strict_pass else "approved_diagnostic"
        if not strict_pass:
            warnings.append("strict_research_thresholds_not_met")
    else:
        approval_status = "rejected_by_afml_gate"
    return {
        "enabled": True,
        "action": str(config.get("action", "skip_real_backtest")),
        "approved_for_backtest": not blockers,
        "approval_status": approval_status,
        "blockers": blockers,
        "primary_failure_reasons": primary_reasons,
        "warnings": warnings,
        "precision_lift": precision_lift,
        "f1_lift": f1_lift,
        "min_precision_lift": min_precision_lift,
        "min_f1_lift": min_f1_lift,
        "block_f1_lift": block_f1_lift,
        "min_proxy_profit_factor": min_profit_factor,
        "max_failure_probability": max_failure_probability,
        "min_active_signals": min_active_signals,
        "strict_thresholds": {
            "psr": strict_psr,
            "dsr": strict_dsr,
            "failure_probability": strict_failure,
        },
    }


def print_afml_acceptance_gate_summary(symbol: str, gate: dict[str, Any]) -> None:
    print(
        f"{symbol}: AFML ACCEPTANCE "
        f"status={gate.get('approval_status')} "
        f"approved_for_backtest={gate.get('approved_for_backtest')} "
        f"blockers={gate.get('blockers', [])} "
        f"warnings={gate.get('warnings', [])}",
        flush=True,
    )


def research_diagnosis_summary(
    *,
    primary_diagnostics: dict[str, Any],
    threshold_selection: dict[str, Any] | None,
    signal_proxy: dict[str, Any],
    diagnostic_signal_proxy: dict[str, Any],
    afml_gate: dict[str, Any],
    active_signal_count: int,
    diagnostic_active_signal_count: int,
) -> dict[str, Any]:
    blockers = afml_gate.get("blockers", []) if isinstance(afml_gate.get("blockers"), list) else []
    primary_reasons = (
        primary_diagnostics.get("primary_failure_reasons", [])
        if isinstance(primary_diagnostics, dict)
        else []
    )
    threshold_selection = threshold_selection if isinstance(threshold_selection, dict) else {}
    threshold_decision = (
        threshold_selection.get("threshold_decision", {})
        if isinstance(threshold_selection.get("threshold_decision"), dict)
        else {}
    )
    failed_stages: list[str] = []
    if primary_reasons:
        failed_stages.append("primary")
    if threshold_decision.get("approved_threshold") is None and bool(threshold_selection.get("enabled", False)):
        failed_stages.append("meta_threshold")
    if active_signal_count <= 0:
        failed_stages.append("approved_signal_generation")
    if signal_proxy.get("profit_factor") is None or float(signal_proxy.get("profit_factor") or 0.0) <= 1.0:
        failed_stages.append("signal_proxy")
    if "execution" in blockers:
        failed_stages.append("execution")
    return {
        "status": afml_gate.get("approval_status"),
        "approved_for_backtest": bool(afml_gate.get("approved_for_backtest", False)),
        "failed_stages": sorted(set(failed_stages)),
        "blockers": blockers,
        "primary_failure_reasons": primary_reasons,
        "threshold_decision": threshold_decision,
        "approved_active_signals": int(active_signal_count),
        "diagnostic_active_signals": int(diagnostic_active_signal_count),
        "approved_signal_proxy": {
            "trade_count": signal_proxy.get("trade_count"),
            "profit_factor": signal_proxy.get("profit_factor"),
            "mean_return": signal_proxy.get("mean_return"),
            "position_aware": signal_proxy.get("position_aware"),
        },
        "diagnostic_signal_proxy": {
            "trade_count": diagnostic_signal_proxy.get("trade_count"),
            "profit_factor": diagnostic_signal_proxy.get("profit_factor"),
            "mean_return": diagnostic_signal_proxy.get("mean_return"),
            "position_aware": diagnostic_signal_proxy.get("position_aware"),
        },
    }


def print_backtest_stage_summary(summary: dict[str, Any]) -> None:
    stats = summary.get("position_statistics", {})
    validation = summary.get("validation", {})
    turnover = summary.get("turnover", {})
    sharpe = validation.get("sharpe", {}) if isinstance(validation.get("sharpe"), dict) else {}
    failure = (
        validation.get("strategy_failure", {})
        if isinstance(validation.get("strategy_failure"), dict)
        else {}
    )
    print(
        "REAL BACKTEST METRICS: "
        f"pnl={summary.get('realized_pnl_usdt')} "
        f"return_pct={summary.get('return_pct')} "
        f"gross_profit={stats.get('gross_profit_usdt')} "
        f"gross_loss={stats.get('gross_loss_usdt')} "
        f"profit_factor={fmt_metric(stats.get('profit_factor'))} "
        f"win_rate={fmt_metric(stats.get('win_rate'))} "
        f"max_drawdown={stats.get('max_drawdown_usdt')} "
        f"turnover={fmt_metric(turnover.get('turnover'))} "
        f"sharpe={fmt_metric(sharpe.get('annualized_sharpe'))} "
        f"psr={fmt_metric(sharpe.get('probabilistic_sharpe_ratio'))} "
        f"dsr={fmt_metric(sharpe.get('deflated_sharpe_ratio'))} "
        f"failure_prob={fmt_metric(failure.get('failure_probability'))}",
        flush=True,
    )


def split_dataset_indices_for_window(
    dataset: Any,
    window: TrainTestWindow,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    index = pd.DatetimeIndex(dataset.y.index)
    label_end = pd.Series(pd.to_datetime(dataset.events.loc[index, "t1"], utc=True), index=index)
    train_mask = (
        (index >= window.train_start)
        & (index <= window.train_end)
        & label_end.notna().to_numpy(dtype=bool)
        & (label_end <= window.train_end).to_numpy(dtype=bool)
    )
    test_mask = (
        (index >= window.test_start)
        & (index <= window.test_end)
        & label_end.notna().to_numpy(dtype=bool)
        & (label_end <= window.test_end).to_numpy(dtype=bool)
    )
    train_index = pd.DatetimeIndex(index[train_mask])
    test_index = pd.DatetimeIndex(index[test_mask])
    if train_index.empty or test_index.empty:
        raise ValueError("Explicit train/test split leaves one side empty")
    return train_index, test_index


def split_train_validation_indices_for_pruning(
    dataset: Any,
    train_index: pd.DatetimeIndex,
    *,
    validation_fraction: float,
    embargo_bars: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("mda_pruning_validation_fraction must be in (0, 1)")
    ordered = pd.DatetimeIndex(train_index).intersection(dataset.X.index).sort_values()
    if len(ordered) < 4:
        raise ValueError("MDA pruning requires at least four training events")

    split_pos = int(np.floor(len(ordered) * (1.0 - validation_fraction)))
    split_pos = max(1, min(split_pos, len(ordered) - 1))
    validation_index = pd.DatetimeIndex(ordered[split_pos:])
    inner_train_index = pd.DatetimeIndex(ordered[:split_pos])
    validation_start = validation_index.min()

    if int(embargo_bars) > 0:
        bar_index = pd.DatetimeIndex(dataset.close.index).sort_values()
        if not bar_index.empty:
            validation_start_pos = int(bar_index.searchsorted(validation_start, side="left"))
            embargo_start_pos = max(0, validation_start_pos - int(embargo_bars))
            embargo_cutoff = bar_index[embargo_start_pos]
            inner_train_index = pd.DatetimeIndex(inner_train_index[inner_train_index <= embargo_cutoff])

    if "t1" in dataset.events.columns and not inner_train_index.empty:
        event_end = pd.to_datetime(dataset.events.loc[inner_train_index, "t1"], errors="coerce")
        purged_index = event_end.index[event_end.isna() | (event_end <= validation_start)]
        inner_train_index = pd.DatetimeIndex(purged_index)

    if inner_train_index.empty or validation_index.empty:
        raise ValueError("MDA pruning train/validation split is empty after purge and embargo")
    return inner_train_index, validation_index


def fit_meta_model(
    model: Any,
    dataset: Any,
    train_index: pd.DatetimeIndex,
    *,
    progress_label: str | None = None,
    args: argparse.Namespace | None = None,
) -> Any:
    started = (
        progress_start(
            args,
            f"{progress_label}: fitting meta model rows={len(train_index):,} "
            f"features={dataset.X.shape[1]:,} classes={class_counts(dataset.y.loc[train_index])}",
        )
        if progress_label
        else None
    )
    if getattr(model, "requires_dataset", False):
        model.fit_dataset(dataset, train_index)
    elif getattr(model, "requires_event_spans", False):
        model.fit(
            dataset.X.loc[train_index],
            dataset.y.loc[train_index],
            sample_weight=dataset.sample_weight.loc[train_index],
            t1=dataset.events.loc[train_index, "t1"],
            bar_index=dataset.close.index,
        )
    else:
        model.fit(
            dataset.X.loc[train_index],
            dataset.y.loc[train_index],
            sample_weight=dataset.sample_weight.loc[train_index],
        )
    if progress_label:
        progress_done(args, f"{progress_label}: fitting meta model", started)
    return model


def replace_dataset_features(dataset: AfmlDataset, features: pd.DataFrame) -> AfmlDataset:
    return AfmlDataset(
        close=dataset.close,
        features=features.reindex(dataset.features.index),
        events=dataset.events,
        labels=dataset.labels,
        sample_weight=dataset.sample_weight,
        volatility=dataset.volatility,
        t_events=dataset.t_events,
        cusum_threshold=dataset.cusum_threshold,
    )


def feature_selection_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    meta_config = section(strategy_config, "meta_model")
    config = meta_config.get("feature_selection", {})
    return config if isinstance(config, dict) else {}


def is_primary_context_feature(column: str) -> bool:
    return column in PRIMARY_CONTEXT_COLUMNS or column.startswith(PRIMARY_CONTEXT_PREFIXES)


def protected_primary_feature_columns(columns: list[str], strategy_config: dict[str, Any]) -> list[str]:
    config = feature_selection_config(strategy_config)
    protected = config.get("protected_primary_features", True)
    if not protected:
        return []
    if isinstance(protected, list):
        requested = {str(column) for column in protected}
        return [column for column in columns if column in requested]
    return [column for column in columns if is_primary_context_feature(column)]


def selected_with_protected_features(selected: list[str], protected: list[str]) -> list[str]:
    out: list[str] = []
    for column in [*selected, *protected]:
        if column not in out:
            out.append(column)
    return out


def pca_feature_selection_enabled(strategy_config: dict[str, Any], args: argparse.Namespace) -> bool:
    config = feature_selection_config(strategy_config)
    return bool(config.get("enabled", args.pca_components is not None))


def selected_feature_count(
    *,
    available: int,
    strategy_config: dict[str, Any],
) -> int:
    config = feature_selection_config(strategy_config)
    top_n = config.get("top_n")
    if top_n is None:
        return available
    return max(1, min(int(top_n), available))


def mda_pruning_enabled(strategy_config: dict[str, Any]) -> bool:
    config = feature_selection_config(strategy_config)
    return bool(config.get("mda_prune_negative", True))


def final_mda_enabled(strategy_config: dict[str, Any]) -> bool:
    config = feature_selection_config(strategy_config)
    return bool(config.get("final_mda_enabled", config.get("mda_enabled", True)))


def final_sfi_enabled(strategy_config: dict[str, Any]) -> bool:
    config = feature_selection_config(strategy_config)
    return bool(config.get("final_sfi_enabled", config.get("sfi_enabled", True)))


def write_importance_csv(frame: pd.DataFrame, path: Path) -> str | None:
    if frame.empty:
        return None
    frame.to_csv(path, index=False)
    return str(path)


def threshold_selection_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    meta_config = section(strategy_config, "meta_model")
    config = meta_config.get("threshold_selection", {})
    return config if isinstance(config, dict) else {}


def probability_calibration_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    validation = section(strategy_config, "validation")
    config = validation.get("probability_calibration", {})
    if not isinstance(config, dict):
        return {"enabled": False, "method": "sigmoid"}
    method = str(config.get("method", "sigmoid")).strip().lower()
    if method not in {"sigmoid", "isotonic"}:
        method = "sigmoid"
    return {
        "enabled": bool(config.get("enabled", False)),
        "method": method,
    }


def threshold_grid(config: dict[str, Any], fallback_threshold: float) -> list[float]:
    if isinstance(config.get("thresholds"), list):
        values = [float(value) for value in config["thresholds"]]
    else:
        start = float(config.get("min_threshold", 0.50))
        stop = float(config.get("max_threshold", 0.90))
        step = float(config.get("step", 0.05))
        if step <= 0.0:
            raise ValueError("meta_model.threshold_selection.step must be positive")
        values = list(np.arange(start, stop + step / 2.0, step))
    values.append(float(fallback_threshold))
    return sorted({round(min(1.0, max(0.0, value)), 6) for value in values})


def threshold_upper_grid(config: dict[str, Any]) -> list[float | None]:
    band_config = config.get("probability_band", {})
    if not isinstance(band_config, dict) or not bool(band_config.get("enabled", False)):
        return [None]
    raw_values = band_config.get("upper_thresholds", [])
    if not isinstance(raw_values, list):
        raw_values = []
    values: list[float | None] = [None]
    for raw_value in raw_values:
        if raw_value is None:
            values.append(None)
            continue
        value = round(min(1.0, max(0.0, float(raw_value))), 6)
        if value >= 1.0:
            values.append(None)
        else:
            values.append(value)
    return sorted(set(values), key=lambda value: 1.0 if value is None else value)


def binary_prediction_summary(
    truth: pd.Series,
    prediction: pd.Series,
    returns: pd.Series,
    *,
    threshold: float,
    upper_threshold: float | None = None,
    event_ends: pd.Series | None = None,
) -> dict[str, Any]:
    prediction = prediction.reindex(truth.index).fillna(0).astype(int)
    metrics = classification_metrics(truth, prediction, positive_label=1)
    active_returns = returns.reindex(truth.index[prediction == 1]).dropna().astype(float)
    wins = active_returns[active_returns > 0.0]
    losses = active_returns[active_returns < 0.0]
    gross_profit = float(active_returns[active_returns > 0.0].sum())
    gross_loss = float(-active_returns[active_returns < 0.0].sum())
    row = {
        "threshold": float(threshold),
        "upper_threshold": None if upper_threshold is None else float(upper_threshold),
        "accepted_count": int((prediction == 1).sum()),
        "accepted_rate": float((prediction == 1).mean()) if len(prediction) else 0.0,
        "win_count": len(wins),
        "loss_count": len(losses),
        "flat_count": int((active_returns == 0.0).sum()),
        "precision": metrics["positive_precision"],
        "recall": metrics["positive_recall"],
        "f1": metrics["positive_f1"],
        "accuracy": metrics["accuracy"],
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "mean_return": float(active_returns.mean()) if not active_returns.empty else None,
        "gross_profit_return": gross_profit,
        "gross_loss_return": gross_loss,
    }
    if event_ends is not None:
        row["execution_proxy"] = position_aware_prediction_summary(
            prediction,
            returns,
            event_ends=event_ends,
        )
    return row


def position_aware_prediction_summary(
    prediction: pd.Series,
    returns: pd.Series,
    *,
    event_ends: pd.Series,
) -> dict[str, Any]:
    """
    Collapse overlapping candidate events into a single-position hold-to-barrier proxy.
    """
    index = pd.DatetimeIndex(prediction.index).sort_values()
    prediction = prediction.reindex(index).fillna(0).astype(int)
    returns = returns.reindex(index).fillna(0.0).astype(float)
    event_ends = pd.to_datetime(event_ends.reindex(index), utc=True, errors="coerce")

    selected: list[pd.Timestamp] = []
    open_until: pd.Timestamp | None = None
    for timestamp in index:
        if int(prediction.loc[timestamp]) != 1:
            continue
        if open_until is not None and timestamp <= open_until:
            continue
        selected.append(timestamp)
        end = event_ends.loc[timestamp]
        if pd.isna(end):
            end = timestamp
        open_until = max(pd.Timestamp(end), pd.Timestamp(timestamp))

    selected_returns = returns.reindex(selected).dropna().astype(float)
    wins = selected_returns[selected_returns > 0.0]
    losses = selected_returns[selected_returns < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trade_count": len(selected_returns),
        "win_count": len(wins),
        "loss_count": len(losses),
        "flat_count": int((selected_returns == 0.0).sum()),
        "win_rate": float(len(wins) / len(selected_returns)) if len(selected_returns) else 0.0,
        "gross_profit_return": gross_profit,
        "gross_loss_return": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "mean_return": float(selected_returns.mean()) if not selected_returns.empty else None,
        "median_return": float(selected_returns.median()) if not selected_returns.empty else None,
    }


def threshold_failed_constraints(
    row: dict[str, Any],
    *,
    min_trades: int,
    max_accepted_rate: float,
    min_precision_lift: float,
    min_profit_factor: float,
    min_win_count: int,
    min_loss_count: int,
    min_mean_return: float,
    execution_config: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    execution_config = execution_config if isinstance(execution_config, dict) else {}
    execution_primary = bool(execution_config.get("enabled", False)) and bool(
        execution_config.get("use_as_primary_constraint", False),
    )
    if int(row["accepted_count"]) < min_trades:
        failures.append("accepted_count_below_min_trades")
    if float(row["accepted_rate"]) > max_accepted_rate:
        failures.append("accepted_rate_above_limit")
    if float(row["precision_lift"]) < min_precision_lift:
        failures.append("precision_lift_below_minimum")
    if int(row.get("win_count", 0)) < min_win_count:
        failures.append("win_count_below_minimum")
    if int(row.get("loss_count", 0)) < min_loss_count:
        failures.append("loss_count_below_minimum")
    if not execution_primary:
        profit_factor = row.get("profit_factor")
        if profit_factor is None or float(profit_factor) <= min_profit_factor:
            failures.append("profit_factor_below_minimum")
        mean_return = row.get("mean_return")
        if mean_return is None or float(mean_return) <= min_mean_return:
            failures.append("mean_return_below_minimum")
    failures.extend(
        threshold_execution_failed_constraints(
            row,
            execution_config=execution_config,
            min_profit_factor=min_profit_factor,
            min_mean_return=min_mean_return,
        ),
    )
    return failures


def threshold_execution_failed_constraints(
    row: dict[str, Any],
    *,
    execution_config: dict[str, Any] | None,
    min_profit_factor: float,
    min_mean_return: float,
) -> list[str]:
    failures: list[str] = []
    if isinstance(execution_config, dict) and bool(execution_config.get("enabled", False)):
        execution = row.get("execution_proxy", {})
        execution = execution if isinstance(execution, dict) else {}
        min_positions = int(execution_config.get("min_positions", 1))
        if int(execution.get("trade_count", 0)) < min_positions:
            failures.append("execution_position_count_below_minimum")
        min_execution_win_rate = execution_config.get("min_win_rate")
        if min_execution_win_rate is not None:
            execution_win_rate = execution.get("win_rate")
            if execution_win_rate is None or float(execution_win_rate) < float(min_execution_win_rate):
                failures.append("execution_win_rate_below_minimum")
        min_execution_pf = float(execution_config.get("min_profit_factor", min_profit_factor))
        execution_pf = execution.get("profit_factor")
        if execution_pf is None or float(execution_pf) <= min_execution_pf:
            failures.append("execution_profit_factor_below_minimum")
        min_execution_mean = float(execution_config.get("min_mean_return", min_mean_return))
        execution_mean = execution.get("mean_return")
        if execution_mean is None or float(execution_mean) <= min_execution_mean:
            failures.append("execution_mean_return_below_minimum")
    return failures


def threshold_filter_candidates(
    filter_features: pd.DataFrame | None,
    config: dict[str, Any],
    index: pd.Index,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "filter_name": "none",
            "excluded_filter_columns": [],
            "mask": pd.Series(True, index=index),
        },
    ]
    filter_config = config.get("candidate_filters", {})
    if not isinstance(filter_config, dict) or not bool(filter_config.get("enabled", False)):
        return candidates
    if filter_features is None or filter_features.empty:
        return candidates

    requested_columns = filter_config.get("columns", [])
    if not isinstance(requested_columns, list):
        requested_columns = []
    available_columns = [
        str(column)
        for column in requested_columns
        if str(column) in filter_features.columns
    ]
    if not available_columns:
        return candidates

    features = filter_features.reindex(index).fillna(0.0)
    max_exclusions = int(filter_config.get("max_exclusions", 1))
    max_exclusions = max(1, min(max_exclusions, len(available_columns)))
    column_sets: list[tuple[str, ...]] = []
    for size in range(1, max_exclusions + 1):
        column_sets.extend(combinations(available_columns, size))
    for columns in column_sets:
        excluded = pd.Series(False, index=index)
        for column in columns:
            excluded |= pd.to_numeric(features[column], errors="coerce").fillna(0.0) >= 0.5
        mask = ~excluded
        if bool(mask.any()) and bool(excluded.any()):
            filter_name = "+".join(columns)
            candidates.append(
                {
                    "filter_name": f"exclude:{filter_name}",
                    "excluded_filter_columns": list(columns),
                    "excluded_count": int(excluded.sum()),
                    "mask": mask.astype(bool),
                },
            )
    return candidates


def select_meta_probability_threshold(
    *,
    truth: pd.Series,
    proba_1: pd.Series,
    returns: pd.Series,
    fallback_threshold: float,
    config: dict[str, Any],
    filter_features: pd.DataFrame | None = None,
    event_ends: pd.Series | None = None,
) -> dict[str, Any]:
    truth = truth.astype(int)
    proba_1 = proba_1.reindex(truth.index).fillna(0.0).astype(float)
    returns = returns.reindex(truth.index).fillna(0.0).astype(float)
    base_rate = float((truth == 1).mean()) if len(truth) else 0.0
    min_trades = int(config.get("min_trades", 30))
    max_accepted_rate = float(config.get("max_accepted_rate", 0.60))
    min_precision_lift = float(config.get("min_precision_lift", 0.0))
    min_profit_factor = float(config.get("min_profit_factor", 1.0))
    min_win_count = int(config.get("min_win_count", 1))
    min_loss_count = int(config.get("min_loss_count", 1))
    min_mean_return = float(config.get("min_mean_return", 0.0))
    execution_config = config.get("execution_proxy", {})
    execution_config = execution_config if isinstance(execution_config, dict) else {}
    filter_candidates = threshold_filter_candidates(filter_features, config, truth.index)
    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid(config, fallback_threshold):
        for upper_threshold in threshold_upper_grid(config):
            if upper_threshold is not None and float(upper_threshold) < float(threshold):
                continue
            base_prediction = proba_1 >= threshold
            if upper_threshold is not None:
                base_prediction &= proba_1 <= float(upper_threshold)
            base_prediction = base_prediction.astype(int)
            for filter_candidate in filter_candidates:
                prediction = base_prediction.where(filter_candidate["mask"], 0)
                row = binary_prediction_summary(
                    truth,
                    prediction,
                    returns,
                    threshold=threshold,
                    upper_threshold=upper_threshold,
                    event_ends=event_ends,
                )
                profit_factor = row["profit_factor"]
                row["filter_name"] = filter_candidate["filter_name"]
                row["excluded_filter_columns"] = filter_candidate["excluded_filter_columns"]
                if "excluded_count" in filter_candidate:
                    row["excluded_count"] = filter_candidate["excluded_count"]
                row["precision_lift"] = float(row["precision"] - base_rate)
                row["failed_constraints"] = threshold_failed_constraints(
                    row,
                    min_trades=min_trades,
                    max_accepted_rate=max_accepted_rate,
                    min_precision_lift=min_precision_lift,
                    min_profit_factor=min_profit_factor,
                    min_win_count=min_win_count,
                    min_loss_count=min_loss_count,
                    min_mean_return=min_mean_return,
                    execution_config=execution_config,
                )
                row["passes_constraints"] = not row["failed_constraints"]
                pf_score = 0.0 if profit_factor is None else min(float(profit_factor), 10.0)
                if bool(execution_config.get("enabled", False)):
                    execution = row.get("execution_proxy", {})
                    execution = execution if isinstance(execution, dict) else {}
                    execution_pf = execution.get("profit_factor")
                    pf_score = 0.0 if execution_pf is None else min(float(execution_pf), 10.0)
                    trade_reference = float(execution_config.get("trade_count_reference", min_trades))
                    trade_reference = max(1.0, trade_reference)
                    trade_weight = float(execution_config.get("trade_count_weight", 0.0))
                    trade_score = trade_weight * min(
                        2.0,
                        float(execution.get("trade_count", 0)) / trade_reference,
                    )
                else:
                    trade_score = 0.0
                filter_penalty = 0.0 if row["filter_name"] == "none" else 0.01
                band_penalty = 0.0 if row["upper_threshold"] is None else 0.005
                row["objective_score"] = (
                    pf_score
                    + float(row["f1"])
                    + trade_score
                    - filter_penalty
                    - band_penalty
                )
                rows.append(row)
    eligible = [row for row in rows if bool(row["passes_constraints"])]
    diagnostic_selected = max(
        rows,
        key=lambda row: (float(row["objective_score"]), float(row["f1"]), float(row["precision"])),
    )
    approved_selected = (
        max(eligible, key=lambda row: (float(row["objective_score"]), float(row["f1"]), float(row["precision"])))
        if eligible
        else None
    )
    approved_threshold = None if approved_selected is None else float(approved_selected["threshold"])
    diagnostic_threshold = float(diagnostic_selected["threshold"])
    approved_upper_threshold = (
        None
        if approved_selected is None or approved_selected.get("upper_threshold") is None
        else float(approved_selected["upper_threshold"])
    )
    diagnostic_upper_threshold = (
        None
        if diagnostic_selected.get("upper_threshold") is None
        else float(diagnostic_selected["upper_threshold"])
    )
    return {
        "enabled": True,
        "status": "ok" if eligible else "no_threshold_passed_constraints",
        "base_positive_rate": base_rate,
        "selected_threshold": diagnostic_threshold if approved_selected is None else approved_threshold,
        "selected_upper_threshold": diagnostic_upper_threshold
        if approved_selected is None
        else approved_upper_threshold,
        "selected_passes_constraints": approved_selected is not None,
        "approved_threshold": approved_threshold,
        "approved_upper_threshold": approved_upper_threshold,
        "diagnostic_threshold": diagnostic_threshold,
        "diagnostic_upper_threshold": diagnostic_upper_threshold,
        "threshold_decision": {
            "status": "approved" if approved_selected is not None else "diagnostic_only",
            "approved_threshold": approved_threshold,
            "approved_upper_threshold": approved_upper_threshold,
            "diagnostic_threshold": diagnostic_threshold,
            "diagnostic_upper_threshold": diagnostic_upper_threshold,
            "failed_constraints": []
            if approved_selected is not None
            else sorted({failure for row in rows for failure in row["failed_constraints"]}),
        },
        "constraints": {
            "min_trades": min_trades,
            "max_accepted_rate": max_accepted_rate,
            "min_precision_lift": min_precision_lift,
            "min_profit_factor": min_profit_factor,
            "min_win_count": min_win_count,
            "min_loss_count": min_loss_count,
            "min_mean_return": min_mean_return,
        },
        "candidate_filter_config": config.get("candidate_filters", {}),
        "probability_band_config": config.get("probability_band", {}),
        "execution_proxy_config": execution_config,
        "selected": approved_selected if approved_selected is not None else diagnostic_selected,
        "approved": approved_selected,
        "diagnostic": diagnostic_selected,
        "candidates": rows,
    }


def fit_rule_meta_model_with_feature_selection(  # noqa: C901
    *,
    meta_dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    symbol: str,
    meta_probability_threshold: float,
    meta_label_min_ret: float,
) -> MetaTrainingResult:
    stage_started = progress_start(
        args,
        f"{symbol}: meta training pipeline train={len(train_index):,} "
        f"test={len(test_index):,} raw_features={meta_dataset.X.shape[1]:,}",
    )
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "method": "original_feature_space",
    }
    final_dataset = meta_dataset
    final_feature_columns = list(meta_dataset.X.columns)
    all_feature_columns = list(meta_dataset.X.columns)
    protected_columns = protected_primary_feature_columns(all_feature_columns, strategy_config)

    if pca_feature_selection_enabled(strategy_config, args):
        pca_components = args.pca_components
        if pca_components is None:
            pca_components = section(strategy_config, "meta_model").get("pca_components", 0.95)
        pca_components = float(pca_components)
        selectable_columns = [
            column
            for column in all_feature_columns
            if column not in set(protected_columns)
        ]
        if not selectable_columns:
            raise ValueError("PCA feature selection requires at least one non-protected feature")
        selectable_dataset = replace_dataset_features(meta_dataset, meta_dataset.features[selectable_columns])
        pca_started = progress_start(
            args,
            f"{symbol}: PCA transform fit raw_features={len(selectable_columns):,}",
        )
        pca_dataset, pca_transformer = _pca_dataset(
            selectable_dataset,
            train_index,
            pca_components=pca_components,
            pca_random_state=args.pca_random_state,
        )
        progress_done(args, f"{symbol}: PCA transform fit", pca_started)
        if pca_transformer is None:
            raise ValueError("PCA feature selection requires a fitted transformer")
        print(
            f"{symbol}: PCA feature selection fit on train fold "
            f"raw_features={len(selectable_columns)} components={pca_dataset.X.shape[1]}",
            flush=True,
        )
        pca_model = model_from_config(
            role="meta-pca",
            strategy_config=strategy_config,
            args=args,
            seed_offset=20_000,
        )
        pca_model = fit_meta_model(
            pca_model,
            pca_dataset,
            train_index,
            progress_label=f"{symbol}:meta_pca",
            args=args,
        )
        pca_importance_started = progress_start(args, f"{symbol}: PCA MDI backprojection")
        pca_importance = pca_mdi_feature_importance(
            pca_model,
            pca_transformer,
            selectable_dataset.X.columns,
        )
        progress_done(args, f"{symbol}: PCA MDI backprojection", pca_importance_started)
        top_n = selected_feature_count(
            available=len(selectable_columns),
            strategy_config=strategy_config,
        )
        selected_raw = pca_importance.head(top_n)["feature"].astype(str).tolist()
        final_feature_columns = selected_with_protected_features(selected_raw, protected_columns)
        final_dataset = replace_dataset_features(meta_dataset, meta_dataset.features[final_feature_columns])
        pca_train_pred = threshold_binary_prediction(
            pca_model,
            pca_dataset.X.loc[train_index],
            threshold=meta_probability_threshold,
        )
        pca_test_pred = threshold_binary_prediction(
            pca_model,
            pca_dataset.X.loc[test_index],
            threshold=meta_probability_threshold,
        )
        pca_train_metrics = classification_metrics(
            pca_dataset.y.loc[train_index],
            pca_train_pred,
            positive_label=1,
        )
        pca_test_metrics = classification_metrics(
            pca_dataset.y.loc[test_index],
            pca_test_pred,
            positive_label=1,
        )
        pca_importance_path = output_dir / f"{symbol}_meta_pca_backprojected_importance.csv"
        diagnostics = {
            "enabled": True,
            "method": "train_fold_standardize_pca_backproject_select_original_then_mda_negative_prune",
            "pca_components": pca_components,
            "raw_feature_count": len(all_feature_columns),
            "selectable_feature_count": len(selectable_columns),
            "pca_component_count": int(pca_dataset.X.shape[1]),
            "selected_raw_feature_count": len(selected_raw),
            "primary_context_feature_count": len(protected_columns),
            "protected_primary_features": protected_columns,
            "selected_feature_count": len(final_feature_columns),
            "selected_features": final_feature_columns,
            "pca_importance_csv": write_importance_csv(pca_importance, pca_importance_path),
            "pca_train_metrics": pca_train_metrics,
            "pca_test_metrics": pca_test_metrics,
        }

        if mda_pruning_enabled(strategy_config):
            config = feature_selection_config(strategy_config)
            pruning_validation_fraction = float(config.get("mda_pruning_validation_fraction", 0.25))
            pruning_split_started = progress_start(args, f"{symbol}: meta MDA pruning split")
            pruning_train_index, pruning_validation_index = split_train_validation_indices_for_pruning(
                final_dataset,
                train_index,
                validation_fraction=pruning_validation_fraction,
                embargo_bars=int(args.embargo_bars),
            )
            progress_done(args, f"{symbol}: meta MDA pruning split", pruning_split_started)
            provisional_model = model_from_config(
                role="meta-mda-prune",
                strategy_config=strategy_config,
                args=args,
                seed_offset=10_000,
            )
            provisional_model = fit_meta_model(
                provisional_model,
                final_dataset,
                pruning_train_index,
                progress_label=f"{symbol}:meta_mda_prune",
                args=args,
            )
            pruning_mda = mda_feature_importance(
                provisional_model,
                final_dataset.X.loc[pruning_validation_index],
                final_dataset.y.loc[pruning_validation_index],
                sample_weight=final_dataset.sample_weight.loc[pruning_validation_index],
                n_repeats=int(config.get("mda_repeats", 3)),
                random_state=int(args.pca_random_state),
                progress_label=f"{symbol}:meta_mda_pruning",
                progress_interval=args.progress_interval,
            )
            retained_features = pruning_mda.loc[pruning_mda["importance"] >= 0.0, "feature"].astype(str).tolist()
            retained_features = selected_with_protected_features(retained_features, protected_columns)
            dropped_features = [
                feature
                for feature in final_feature_columns
                if feature not in set(retained_features) and feature not in set(protected_columns)
            ]
            if not retained_features:
                raise ValueError("MDA pruning removed every selected feature; all pruning MDA scores are negative")
            final_feature_columns = [
                feature
                for feature in final_feature_columns
                if feature in set(retained_features)
            ]
            final_dataset = replace_dataset_features(meta_dataset, meta_dataset.features[final_feature_columns])
            pruning_path = output_dir / f"{symbol}_meta_mda_pruning_importance.csv"
            diagnostics["mda_pruning_enabled"] = True
            diagnostics["mda_pruning_csv"] = write_importance_csv(pruning_mda, pruning_path)
            diagnostics["mda_pruning_candidate_feature_count"] = len(selected_raw)
            diagnostics["mda_pruning_retained_feature_count"] = len(final_feature_columns)
            diagnostics["mda_pruning_dropped_feature_count"] = len(dropped_features)
            diagnostics["mda_pruning_dropped_features"] = dropped_features
            diagnostics["mda_pruning_protected_feature_count"] = len(protected_columns)
            diagnostics["mda_pruning_validation_fraction"] = pruning_validation_fraction
            diagnostics["mda_pruning_train_events"] = len(pruning_train_index)
            diagnostics["mda_pruning_validation_events"] = len(pruning_validation_index)
            diagnostics["selected_feature_count"] = len(final_feature_columns)
            diagnostics["selected_features"] = final_feature_columns
        else:
            diagnostics["mda_pruning_enabled"] = False
    else:
        diagnostics["primary_context_feature_count"] = len(protected_columns)
        diagnostics["protected_primary_features"] = protected_columns

    selected_threshold = float(meta_probability_threshold)
    approved_threshold: float | None = selected_threshold
    diagnostic_threshold = float(meta_probability_threshold)
    approved_upper_threshold: float | None = None
    diagnostic_upper_threshold: float | None = None
    threshold_config = threshold_selection_config(strategy_config)
    calibration_config = probability_calibration_config(strategy_config)
    calibration_probability: pd.Series | None = None
    calibration_truth: pd.Series | None = None
    calibration_summary: dict[str, Any] = {
        "enabled": bool(calibration_config.get("enabled", False)),
        "method": calibration_config.get("method", "sigmoid"),
        "status": "configured_off"
        if not bool(calibration_config.get("enabled", False))
        else "not_fitted",
    }
    if bool(threshold_config.get("enabled", False)):
        try:
            validation_fraction = float(threshold_config.get("validation_fraction", 0.25))
            threshold_split_started = progress_start(args, f"{symbol}: meta threshold split")
            threshold_train_index, threshold_validation_index = split_train_validation_indices_for_pruning(
                final_dataset,
                train_index,
                validation_fraction=validation_fraction,
                embargo_bars=int(threshold_config.get("embargo_bars", args.embargo_bars)),
            )
            progress_done(args, f"{symbol}: meta threshold split", threshold_split_started)
            threshold_model = model_from_config(
                role="meta-threshold",
                strategy_config=strategy_config,
                args=args,
                seed_offset=40_000,
            )
            threshold_model = fit_meta_model(
                threshold_model,
                final_dataset,
                threshold_train_index,
                progress_label=f"{symbol}:meta_threshold",
                args=args,
            )
            threshold_predict_started = progress_start(args, f"{symbol}: meta threshold predict/select")
            threshold_proba = threshold_model.predict_proba(final_dataset.X.loc[threshold_validation_index])
            if 1 not in threshold_proba.columns:
                raise ValueError("threshold model did not produce class 1 probabilities")
            threshold_proba_1 = threshold_proba[1]
            if bool(calibration_config.get("enabled", False)):
                try:
                    calibration_truth = final_dataset.y.loc[threshold_validation_index].astype(int)
                    calibrated_threshold_proba = calibrate_binary_probabilities(
                        threshold_proba_1,
                        calibration_truth,
                        target_probability=threshold_proba_1,
                        method=str(calibration_config.get("method", "sigmoid")),
                    )
                    calibration_probability = threshold_proba_1.astype(float)
                    threshold_proba_1 = calibrated_threshold_proba
                    calibration_summary = {
                        "enabled": True,
                        "method": calibration_config.get("method", "sigmoid"),
                        "status": "calibrated",
                        "calibration_events": len(calibration_truth),
                        "calibration_positive_rate": float((calibration_truth == 1).mean()),
                        "calibration_fold": "threshold_validation",
                    }
                except Exception as exc:
                    calibration_probability = None
                    calibration_truth = None
                    calibration_summary = {
                        "enabled": True,
                        "method": calibration_config.get("method", "sigmoid"),
                        "status": "failed",
                        "error": str(exc),
                    }
            threshold_summary = select_meta_probability_threshold(
                truth=final_dataset.y.loc[threshold_validation_index],
                proba_1=threshold_proba_1,
                returns=final_dataset.labels.loc[threshold_validation_index, "ret"] - float(meta_label_min_ret),
                fallback_threshold=float(meta_probability_threshold),
                config=threshold_config,
                filter_features=add_threshold_filter_features(
                    final_dataset.X.loc[threshold_validation_index],
                    primary_side=final_dataset.events.loc[threshold_validation_index, "side"],
                ),
                event_ends=final_dataset.events.loc[threshold_validation_index, "t1"],
            )
            progress_done(args, f"{symbol}: meta threshold predict/select", threshold_predict_started)
            threshold_summary["return_basis"] = "cost_adjusted_label_ret"
            threshold_summary["round_trip_cost"] = float(meta_label_min_ret)
            threshold_summary["train_events"] = len(threshold_train_index)
            threshold_summary["validation_events"] = len(threshold_validation_index)
            threshold_summary["validation_fraction"] = validation_fraction
            approved_threshold = (
                None
                if threshold_summary.get("approved_threshold") is None
                else float(threshold_summary["approved_threshold"])
            )
            approved_upper_threshold = (
                None
                if threshold_summary.get("approved_upper_threshold") is None
                else float(threshold_summary["approved_upper_threshold"])
            )
            diagnostic_threshold = float(threshold_summary.get("diagnostic_threshold", meta_probability_threshold))
            diagnostic_upper_threshold = (
                None
                if threshold_summary.get("diagnostic_upper_threshold") is None
                else float(threshold_summary["diagnostic_upper_threshold"])
            )
            selected_threshold = approved_threshold if approved_threshold is not None else diagnostic_threshold
        except Exception as exc:
            threshold_summary = {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
                "selected_threshold": selected_threshold,
                "selected_upper_threshold": None,
                "selected_passes_constraints": False,
                "approved_threshold": None,
                "approved_upper_threshold": None,
                "diagnostic_threshold": diagnostic_threshold,
                "diagnostic_upper_threshold": diagnostic_upper_threshold,
                "threshold_decision": {
                    "status": "diagnostic_only",
                    "approved_threshold": None,
                    "approved_upper_threshold": None,
                    "diagnostic_threshold": diagnostic_threshold,
                    "diagnostic_upper_threshold": diagnostic_upper_threshold,
                    "failed_constraints": ["threshold_selection_failed"],
                },
            }
            approved_threshold = None
            approved_upper_threshold = None
        diagnostics["threshold_selection"] = threshold_summary
    else:
        diagnostics["threshold_selection"] = {
            "enabled": False,
            "selected_threshold": selected_threshold,
            "selected_upper_threshold": None,
            "selected_passes_constraints": None,
            "approved_threshold": selected_threshold,
            "approved_upper_threshold": None,
            "diagnostic_threshold": selected_threshold,
            "diagnostic_upper_threshold": None,
            "threshold_decision": {
                "status": "approved",
                "approved_threshold": selected_threshold,
                "approved_upper_threshold": None,
                "diagnostic_threshold": selected_threshold,
                "diagnostic_upper_threshold": None,
                "failed_constraints": [],
            },
        }

    meta_model = model_from_config(
        role="meta",
        strategy_config=strategy_config,
        args=args,
        seed_offset=10_000,
    )
    meta_model = fit_meta_model(
        meta_model,
        final_dataset,
        train_index,
        progress_label=f"{symbol}:meta_final",
        args=args,
    )
    if hasattr(meta_model, "diagnostics"):
        diagnostics["model_diagnostics"] = meta_model.diagnostics()
    scoring_model: Any = meta_model
    if (
        bool(calibration_config.get("enabled", False))
        and calibration_probability is not None
        and calibration_truth is not None
        and calibration_summary.get("status") == "calibrated"
    ):
        scoring_model = CalibratedBinaryProbabilityModel(
            meta_model,
            calibration_probability=calibration_probability,
            calibration_truth=calibration_truth,
            method=str(calibration_config.get("method", "sigmoid")),
        )
    diagnostics["probability_calibration"] = calibration_summary
    if approved_threshold is None:
        train_pred = pd.Series(0, index=train_index, name="prediction")
        test_pred = pd.Series(0, index=test_index, name="prediction")
    else:
        train_pred = threshold_binary_prediction(
            scoring_model,
            final_dataset.X.loc[train_index],
            threshold=approved_threshold,
            upper_threshold=approved_upper_threshold,
        )
        test_pred = threshold_binary_prediction(
            scoring_model,
            final_dataset.X.loc[test_index],
            threshold=approved_threshold,
            upper_threshold=approved_upper_threshold,
        )
    diagnostics["threshold_decision"] = diagnostics["threshold_selection"].get("threshold_decision", {})
    diagnostics["approved_probability_threshold"] = approved_threshold
    diagnostics["approved_probability_upper_threshold"] = approved_upper_threshold
    diagnostics["diagnostic_probability_threshold"] = diagnostic_threshold
    diagnostics["diagnostic_probability_upper_threshold"] = diagnostic_upper_threshold
    approved_row = diagnostics["threshold_selection"].get("approved")
    diagnostic_row = diagnostics["threshold_selection"].get("diagnostic")
    diagnostics["approved_excluded_filter_columns"] = (
        list(approved_row.get("excluded_filter_columns", []))
        if isinstance(approved_row, dict)
        else []
    )
    diagnostics["diagnostic_excluded_filter_columns"] = (
        list(diagnostic_row.get("excluded_filter_columns", []))
        if isinstance(diagnostic_row, dict)
        else []
    )

    if diagnostics.get("enabled"):
        config = feature_selection_config(strategy_config)
        importance_seed = int(args.pca_random_state)
        mdi_started = progress_start(args, f"{symbol}: meta final MDI")
        mdi = mdi_feature_importance(meta_model)
        progress_done(args, f"{symbol}: meta final MDI", mdi_started)
        diagnostics["final_mdi_csv"] = write_importance_csv(
            mdi,
            output_dir / f"{symbol}_meta_selected_mdi_importance.csv",
        )
        diagnostics["final_top_mdi_features"] = mdi.head(10).to_dict(orient="records")
        diagnostics["final_mda_enabled"] = final_mda_enabled(strategy_config)
        diagnostics["final_sfi_enabled"] = final_sfi_enabled(strategy_config)
        diagnostics["final_mda_csv"] = None
        diagnostics["final_sfi_csv"] = None
        diagnostics["final_top_mda_features"] = []
        diagnostics["final_top_sfi_features"] = []
        if diagnostics["final_mda_enabled"]:
            mda = mda_feature_importance(
                meta_model,
                final_dataset.X.loc[test_index],
                final_dataset.y.loc[test_index],
                sample_weight=final_dataset.sample_weight.loc[test_index],
                n_repeats=int(config.get("mda_repeats", 3)),
                random_state=importance_seed,
                progress_label=f"{symbol}:meta_final_mda",
                progress_interval=args.progress_interval,
            )
            diagnostics["final_mda_csv"] = write_importance_csv(
                mda,
                output_dir / f"{symbol}_meta_selected_mda_importance.csv",
            )
            diagnostics["final_top_mda_features"] = mda.head(10).to_dict(orient="records")
        else:
            progress_print(args, f"{symbol}: meta final MDA skipped by feature_selection config")
        if diagnostics["final_sfi_enabled"]:
            sfi_n_estimators = int(config.get("sfi_n_estimators", 10))
            sfi = sfi_feature_importance(
                final_dataset,
                train_index=train_index,
                test_index=test_index,
                model_factory=lambda: model_from_config(
                    role="meta-sfi",
                    strategy_config=strategy_config,
                    args=args,
                    seed_offset=30_000,
                    n_estimators_override=sfi_n_estimators,
                    announce=False,
                ),
                progress_label=f"{symbol}:meta_final_sfi",
                progress_interval=args.progress_interval,
            )
            diagnostics["final_sfi_csv"] = write_importance_csv(
                sfi,
                output_dir / f"{symbol}_meta_selected_sfi_importance.csv",
            )
            diagnostics["final_top_sfi_features"] = sfi.head(10).to_dict(orient="records")
        else:
            progress_print(args, f"{symbol}: meta final SFI skipped by feature_selection config")

    progress_done(args, f"{symbol}: meta training pipeline", stage_started)
    return MetaTrainingResult(
        model=scoring_model,
        train_pred=train_pred,
        test_pred=test_pred,
        feature_columns=final_feature_columns,
        diagnostics=diagnostics,
        probability_threshold=selected_threshold,
    )


def threshold_feature_mask(
    features: pd.DataFrame,
    excluded_filter_columns: list[str] | None,
) -> pd.Series:
    mask = pd.Series(True, index=features.index)
    for column in excluded_filter_columns or []:
        if column not in features.columns:
            continue
        excluded = pd.to_numeric(features[column], errors="coerce").fillna(0.0) >= 0.5
        mask &= ~excluded
    return mask


def make_rule_meta_signal_frame(
    primary_frame: pd.DataFrame,
    meta_model: Any,
    meta_features: pd.DataFrame,
    *,
    meta_probability_threshold: float,
    active_index: pd.DatetimeIndex,
    feature_columns: list[str] | None = None,
    approved_trading: bool = True,
    diagnostic_probability_threshold: float | None = None,
    approved_probability_upper_threshold: float | None = None,
    diagnostic_probability_upper_threshold: float | None = None,
    approved_excluded_filter_columns: list[str] | None = None,
    diagnostic_excluded_filter_columns: list[str] | None = None,
) -> pd.DataFrame:
    diagnostic_probability_threshold = (
        float(meta_probability_threshold)
        if diagnostic_probability_threshold is None
        else float(diagnostic_probability_threshold)
    )
    approved_threshold_value = float(meta_probability_threshold) if approved_trading else np.nan
    approved_upper_value = (
        float(approved_probability_upper_threshold)
        if approved_trading and approved_probability_upper_threshold is not None
        else np.nan
    )
    diagnostic_upper_value = (
        float(diagnostic_probability_upper_threshold)
        if diagnostic_probability_upper_threshold is not None
        else np.nan
    )
    frame = pd.DataFrame(
        {
            "signal": 0,
            "prediction": primary_frame["primary_prediction"],
            "confidence": 0.0,
            "primary_side": primary_frame["primary_side"],
            "meta_probability": 0.0,
            "meta_prediction": 0,
            "approved_threshold": approved_threshold_value,
            "approved_upper_threshold": approved_upper_value,
            "diagnostic_threshold": diagnostic_probability_threshold,
            "diagnostic_upper_threshold": diagnostic_upper_value,
            "approved_trading": bool(approved_trading),
            "signal_source": "approved_signal" if approved_trading else "diagnostic_signal",
            "gate_status": "pending_afml_gate",
            "diagnostic_signal": 0,
            "diagnostic_meta_prediction": 0,
        },
        index=primary_frame.index,
    )
    attribution_columns = [
        column
        for column in primary_frame.columns
        if column.startswith(("primary_driver_", "primary_expert_", "primary_moe_"))
        or column
        in {
            "primary_confidence",
            "primary_uncertainty",
            "primary_disagreement",
            "primary_abstain_reason",
            "primary_policy",
            "primary_market_state",
            "primary_prob_-1",
            "primary_prob_1",
            "primary_prob_margin",
            "trend_side",
            "trend_confidence",
            "trend_prob_-1",
            "trend_prob_1",
            "reversion_side",
            "reversion_confidence",
            "reversion_prob_-1",
            "reversion_prob_1",
            "trend_score",
            "reversion_score",
            "selected_policy_oof_edge",
            "selected_policy_profit_factor",
            "selected_policy_trade_count",
            "volatility_state",
            "flow_state",
            "entropy_state",
        }
    ]
    if attribution_columns:
        frame = frame.join(primary_frame[attribution_columns], how="left")

    if feature_columns is not None:
        meta_features = meta_features.reindex(columns=feature_columns)
    meta_features = meta_features.replace([np.inf, -np.inf], np.nan).dropna()
    candidate_index = primary_frame.index[primary_frame["primary_side"] != 0]
    if len(candidate_index) > 0:
        candidate_features = meta_features.loc[meta_features.index.intersection(candidate_index)]
        if not candidate_features.empty:
            meta_proba = meta_model.predict_proba(candidate_features)
            if 1 in meta_proba.columns:
                trade_probability = meta_proba[1]
            else:
                trade_probability = pd.Series(0.0, index=meta_proba.index)
            filter_features = add_threshold_filter_features(
                candidate_features,
                primary_side=primary_frame.loc[candidate_features.index, "primary_side"],
            )
            approved_filter = threshold_feature_mask(
                filter_features,
                approved_excluded_filter_columns,
            )
            diagnostic_filter = threshold_feature_mask(
                filter_features,
                diagnostic_excluded_filter_columns,
            )
            meta_prediction = (
                (
                    (trade_probability >= meta_probability_threshold)
                    & (
                        True
                        if approved_probability_upper_threshold is None
                        else trade_probability <= float(approved_probability_upper_threshold)
                    )
                    & approved_filter
                ).astype(int)
                if approved_trading
                else pd.Series(0, index=trade_probability.index, dtype=int)
            )
            diagnostic_meta_prediction = (
                (trade_probability >= diagnostic_probability_threshold)
                & (
                    True
                    if diagnostic_probability_upper_threshold is None
                    else trade_probability <= float(diagnostic_probability_upper_threshold)
                )
                & diagnostic_filter
            ).astype(int)
            frame.loc[candidate_features.index, "meta_probability"] = trade_probability
            frame.loc[candidate_features.index, "meta_prediction"] = meta_prediction
            frame.loc[candidate_features.index, "diagnostic_meta_prediction"] = diagnostic_meta_prediction
            frame.loc[candidate_features.index, "confidence"] = trade_probability
            frame.loc[candidate_features.index, "signal"] = (
                primary_frame.loc[candidate_features.index, "primary_side"] * meta_prediction
            ).astype(int)
            frame.loc[candidate_features.index, "diagnostic_signal"] = (
                primary_frame.loc[candidate_features.index, "primary_side"] * diagnostic_meta_prediction
            ).astype(int)

    inactive = ~frame.index.isin(active_index)
    frame.loc[inactive, "signal"] = 0
    frame.loc[inactive, "diagnostic_signal"] = 0
    frame.loc[inactive, "meta_prediction"] = 0
    frame.loc[inactive, "diagnostic_meta_prediction"] = 0
    return frame


def parse_model_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null"}:
            return None
        if lowered in {"sqrt", "log2", "avg_uniqueness", "avgu"}:
            return "avg_uniqueness" if lowered == "avgu" else lowered
        try:
            number = float(lowered)
        except ValueError:
            return value
        if number.is_integer():
            return int(number)
        return number
    return value


def resolve_max_samples(cli_value: Any, config_value: Any) -> float | int | str | None:
    raw = cli_value if cli_value is not None else config_value
    parsed = parse_model_scalar(raw)
    if parsed is None:
        return "avg_uniqueness"
    if parsed == "avgU":
        return "avg_uniqueness"
    if isinstance(parsed, float) and parsed >= 1.0:
        return "avg_uniqueness"
    if isinstance(parsed, int) and parsed == 1:
        return "avg_uniqueness"
    return parsed


def resolve_class_weight(cli_value: Any, config_value: Any) -> str | dict[int, float] | None:
    raw = cli_value if cli_value is not None else config_value
    if raw is None:
        return "balanced_subsample"
    if isinstance(raw, str) and raw.strip().lower() in {"none", "null"}:
        return None
    return raw


def model_from_config(
    *,
    role: str,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    seed_offset: int,
    n_estimators_override: int | None = None,
    announce: bool = True,
) -> Any:
    meta_config = section(strategy_config, "meta_model")
    bagging_config = section(strategy_config, "sequential_bagging")
    base_seed = int(bagging_config.get("random_seed", 2027))
    family = str(meta_config.get("family", META_SEQUENTIAL_BOOTSTRAP_RF)).strip().lower()
    n_estimators = int(n_estimators_override or args.n_estimators or meta_config.get("n_estimators", 100))
    max_features = parse_model_scalar(args.max_features or meta_config.get("max_features", 1))
    max_samples = resolve_max_samples(args.max_samples, meta_config.get("max_samples"))
    min_samples_leaf = int(args.min_samples_leaf or meta_config.get("min_samples_leaf", 1))
    max_depth = args.max_depth if args.max_depth is not None else meta_config.get("max_depth")
    class_weight = resolve_class_weight(args.class_weight, meta_config.get("class_weight"))

    if announce:
        print(
            f"  {role} model: family={family} estimators={n_estimators} max_features={max_features} "
            f"max_samples={max_samples} min_samples_leaf={min_samples_leaf} "
            f"min_weight_fraction_leaf={args.min_weight_fraction_leaf:g} "
            f"class_weight={class_weight}",
            flush=True,
        )
    base_kwargs = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": args.min_weight_fraction_leaf,
        "class_weight": class_weight,
        "random_state": base_seed + seed_offset,
        "n_jobs": args.n_jobs,
        "verbose": not args.no_progress,
        "progress_interval": args.progress_interval,
        "progress_label": role,
    }
    if family in {META_SEQUENTIAL_BOOTSTRAP_RF, "sequential_bootstrap_random_forest"}:
        return SequentialBootstrapBaggingClassifier(**base_kwargs)
    if family == META_PROFIT_WEIGHTED_SEQUENTIAL_BOOTSTRAP_RF:
        profit_config = meta_config.get("profit_weighting", {})
        profit_config = profit_config if isinstance(profit_config, dict) else {}
        return ProfitWeightedSequentialBootstrapClassifier(
            **base_kwargs,
            return_floor=float(profit_config.get("return_floor", 0.0)),
            return_weight_scale=float(profit_config.get("return_weight_scale", 0.001)),
            max_return_weight=float(profit_config.get("max_return_weight", 6.0)),
            positive_return_multiplier=float(profit_config.get("positive_return_multiplier", 1.0)),
            negative_return_multiplier=float(profit_config.get("negative_return_multiplier", 1.25)),
        )
    raise ValueError(
        f"Unsupported meta_model.family: {family!r}. "
        f"Use {META_SEQUENTIAL_BOOTSTRAP_RF!r} or {META_PROFIT_WEIGHTED_SEQUENTIAL_BOOTSTRAP_RF!r}.",
    )


def frame_with_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if "ts_event_ns" in frame.columns:
        index = pd.to_datetime(frame["ts_event_ns"], unit="ns", utc=True, errors="coerce")
    elif "ts_event" in frame.columns:
        index = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    else:
        raise ValueError("Input bars must contain 'ts_event_ns' or 'ts_event'.")

    out = frame.copy()
    out.index = pd.DatetimeIndex(index)
    out = out.loc[out.index.notna()].sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    return out


def filter_frame_by_time(
    frame: pd.DataFrame,
    *,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    start_ts = parse_time_bound(start)
    end_ts = parse_time_bound(end, is_end=True)
    out = frame
    if start_ts is not None:
        out = out.loc[out.index >= start_ts]
    if end_ts is not None:
        out = out.loc[out.index <= end_ts]
    if out.empty:
        raise ValueError("No bars remain after applying the requested time bounds.")
    return out


def make_train_test_window(  # noqa: C901
    index: pd.DatetimeIndex,
    *,
    train_start: Any,
    train_end: Any,
    test_start: Any,
    test_end: Any,
    test_days: float,
    train_days: float,
    embargo_bars: int,
) -> TrainTestWindow:
    if embargo_bars < 0:
        raise ValueError("embargo_bars cannot be negative")

    explicit_train = train_start is not None or train_end is not None
    explicit_test = test_start is not None or test_end is not None
    if explicit_train or explicit_test:
        requested_train_start = parse_time_bound(train_start) or index[0]
        requested_train_end = parse_time_bound(train_end, is_end=True)
        requested_test_start = parse_time_bound(test_start)
        requested_test_end = parse_time_bound(test_end, is_end=True) or index[-1]
        if requested_test_start is None:
            if test_days <= 0.0:
                raise ValueError("test_days must be positive when test_start is not set")
            requested_test_start = requested_test_end - pd.Timedelta(days=float(test_days))
        if requested_train_end is None:
            train_end_pos = int(index.searchsorted(requested_test_start, side="left")) - embargo_bars - 1
            if train_end_pos < 1:
                raise ValueError("Embargo leaves no training bars before the test window.")
            requested_train_end = index[train_end_pos]

        train_start_pos = int(index.searchsorted(requested_train_start, side="left"))
        train_end_pos = int(index.searchsorted(requested_train_end, side="right")) - 1
        test_start_pos = int(index.searchsorted(requested_test_start, side="left"))
        test_end_pos = int(index.searchsorted(requested_test_end, side="right")) - 1
        if train_start_pos >= len(index) or train_end_pos <= train_start_pos:
            raise ValueError("The requested training window does not contain enough bars.")
        if test_start_pos >= len(index) or test_end_pos <= test_start_pos:
            raise ValueError("The requested test window does not contain enough bars.")
        if index[train_end_pos] >= index[test_start_pos]:
            raise ValueError("train_end must be before test_start")
        return TrainTestWindow(
            train_start=index[train_start_pos],
            train_end=index[train_end_pos],
            test_start=index[test_start_pos],
            test_end=index[test_end_pos],
            embargo_bars=embargo_bars,
        )

    if test_days <= 0.0:
        raise ValueError("test_days must be positive")
    requested_test_end = index[-1]
    requested_test_start = requested_test_end - pd.Timedelta(days=float(test_days))
    test_start_pos = int(index.searchsorted(requested_test_start, side="left"))
    test_end_pos = int(index.searchsorted(requested_test_end, side="right")) - 1
    if test_start_pos >= len(index) or test_end_pos <= test_start_pos:
        raise ValueError("The requested test window does not contain enough bars.")

    train_end_pos = test_start_pos - embargo_bars - 1
    if train_end_pos < 1:
        raise ValueError("Embargo leaves no training bars before the test window.")
    train_end = index[train_end_pos]

    if train_days > 0.0:
        requested_train_start = train_end - pd.Timedelta(days=float(train_days))
        train_start_pos = int(index.searchsorted(requested_train_start, side="left"))
    else:
        train_start_pos = 0
    if train_start_pos >= train_end_pos:
        raise ValueError("Training lookback leaves no training bars.")

    return TrainTestWindow(
        train_start=index[train_start_pos],
        train_end=train_end,
        test_start=index[test_start_pos],
        test_end=index[test_end_pos],
        embargo_bars=embargo_bars,
    )


def _exit_config_value(config: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(config, dict):
        return None
    exit_definition = config.get("exit_definition", {})
    if isinstance(exit_definition, dict) and key in exit_definition:
        return exit_definition[key]
    return config.get(key)


def _resolve_exit_multiplier(
    *,
    runtime_config: dict[str, Any],
    key: str,
    default: float,
) -> float:
    value = _exit_config_value(runtime_config, key)
    if value is not None:
        return float(value)
    return float(default)


def directional_barrier_settings(
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, float], int]:
    barrier_config = section(strategy_config, "barrier_optimization")
    runtime_config = section(strategy_config, "runtime_strategy")
    long_pt = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        key="long_profit_taking_mult",
        default=1.0,
    )
    long_sl = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        key="long_stop_loss_mult",
        default=1.0,
    )
    short_pt = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        key="short_profit_taking_mult",
        default=1.0,
    )
    short_sl = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        key="short_stop_loss_mult",
        default=1.0,
    )
    exit_config = {
        "long_profit_taking_mult": long_pt,
        "long_stop_loss_mult": long_sl,
        "short_profit_taking_mult": short_pt,
        "short_stop_loss_mult": short_sl,
    }
    vertical_bars = int(
        args.vertical_barrier_bars
        or runtime_config.get("vertical_barrier_bars")
        or barrier_config.get("vertical_barrier_bars", 80)
    )
    if any(value <= 0.0 for value in exit_config.values()):
        raise ValueError("pt/sl multipliers must be positive for training labels.")
    return exit_config, vertical_bars


def volatility_target_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kind": VOLATILITY_TARGET_KIND,
        "span": int(args.volatility_ewma_span),
        "lookback_days": int(args.volatility_lookback_days),
        "floor_quantile": float(args.volatility_floor_quantile),
        "target_return_type": "simple_daily_return",
        "barrier_return_type": "log",
    }


def volatility_target_stage_summary(
    volatility: pd.Series,
    *,
    args: argparse.Namespace,
    train_start: Any,
    train_end: Any,
) -> dict[str, Any]:
    valid = volatility.dropna().astype(float)
    train_values = valid.loc[(valid.index >= pd.Timestamp(train_start)) & (valid.index <= pd.Timestamp(train_end))]
    return {
        **volatility_target_metadata(args),
        "valid_count": int(valid.shape[0]),
        "train_count": int(train_values.shape[0]),
        "train_mean": float(train_values.mean()) if not train_values.empty else None,
        "train_median": float(train_values.median()) if not train_values.empty else None,
        "train_min": float(train_values.min()) if not train_values.empty else None,
        "train_max": float(train_values.max()) if not train_values.empty else None,
    }


def print_volatility_target_summary(
    symbol: str,
    summary: dict[str, Any],
) -> None:
    print(
        f"{symbol}: VOLATILITY TARGET kind={summary['kind']} span={summary['span']} "
        f"lookback_days={summary['lookback_days']} floor_q={summary['floor_quantile']} "
        f"valid={summary['valid_count']:,} train={summary['train_count']:,} "
        f"train_median={summary['train_median']} train_mean={summary['train_mean']}",
        flush=True,
    )


def runtime_event_definition(strategy_config: dict[str, Any]) -> dict[str, Any]:
    return dict(section(strategy_config, "event_definition"))


def apply_directional_exit_columns(signals: pd.DataFrame, exit_config: dict[str, float]) -> pd.DataFrame:
    out = signals.copy()
    side = out["signal"].astype(int) if "signal" in out else pd.Series(0, index=out.index)
    long_pt = float(exit_config["long_profit_taking_mult"])
    long_sl = float(exit_config["long_stop_loss_mult"])
    short_pt = float(exit_config["short_profit_taking_mult"])
    short_sl = float(exit_config["short_stop_loss_mult"])
    out["profit_taking_mult"] = long_pt
    out["stop_loss_mult"] = long_sl
    out.loc[side > 0, "profit_taking_mult"] = long_pt
    out.loc[side > 0, "stop_loss_mult"] = long_sl
    out.loc[side < 0, "profit_taking_mult"] = short_pt
    out.loc[side < 0, "stop_loss_mult"] = short_sl
    return out


def build_feature_matrix(
    *,
    frame: pd.DataFrame,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    price_column: str,
    volatility: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    fair_value_config = section(strategy_config, "fair_value")
    event_config = section(strategy_config, "event_definition")

    pipeline_features = make_pipeline_features(
        frame,
        volatility=volatility,
        microstructure_window=args.microstructure_window,
        information_window=args.information_window,
        calendar_windows=args.calendar_windows,
        calendar_min_periods=args.calendar_min_periods,
        tail_quantile=args.tail_quantile,
        robust_z_clip=args.robust_z_clip,
        fracdiff_d=args.feature_fracdiff_d,
        fracdiff_threshold=args.feature_fracdiff_threshold,
    )
    cvd_features, _, _ = event_state_features(
        frame,
        price_column=price_column,
        fair_value_config=fair_value_config,
        event_config=event_config,
    )
    cvd_features.index = frame.index
    cvd_features = cvd_features.add_prefix("cvdslope_")

    features = pd.concat([pipeline_features, cvd_features], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all").ffill()
    return apply_feature_pattern_exclusions(features, strategy_config)


def decimal_places(text: str) -> int:
    value = text.strip()
    if not value:
        return 0
    if "e" in value.lower():
        value = format(Decimal(value), "f")
    if "." not in value:
        return 0
    return len(value.rstrip("0").partition(".")[2])


def infer_price_size_precision(path: Path, sample_rows: int = 5000) -> tuple[int, int]:
    price_precision = 0
    size_precision = 0
    with path.open("r", encoding="utf-8", newline="") as raw:
        reader = csv.DictReader(raw)
        for idx, row in enumerate(reader):
            if idx >= sample_rows:
                break
            for column in ("open", "high", "low", "close"):
                price_precision = max(price_precision, decimal_places(row[column]))
            size_precision = max(size_precision, decimal_places(row["volume"]))
    if price_precision <= 0:
        price_precision = 1
    if size_precision <= 0:
        size_precision = 1
    return price_precision, size_precision


def split_linear_symbol(symbol: str) -> tuple[str, str]:
    raw_symbol = normalize_symbol(symbol)
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if raw_symbol.endswith(quote) and len(raw_symbol) > len(quote):
            return raw_symbol[: -len(quote)], quote
    raise ValueError(f"Cannot infer base/quote for {symbol!r}.")


def quantity_increment(precision: int) -> str:
    return "1" if precision <= 0 else f"0.{'0' * (precision - 1)}1"


def price_increment(precision: int) -> str:
    return "1" if precision <= 0 else f"0.{'0' * (precision - 1)}1"


def make_instrument(symbol: str, csv_path: Path) -> CryptoPerpetual:
    raw_symbol = normalize_symbol(symbol)
    base_code, quote_code = split_linear_symbol(raw_symbol)
    price_precision, size_precision = infer_price_size_precision(csv_path)
    quote_currency = Currency.from_str(quote_code)
    max_price = "1000000000" if price_precision == 0 else f"1000000000.{'0' * price_precision}"
    return CryptoPerpetual(
        instrument_id=InstrumentId(
            symbol=Symbol(f"{raw_symbol}-PERP"),
            venue=BINANCE,
        ),
        raw_symbol=Symbol(raw_symbol),
        base_currency=Currency.from_str(base_code),
        quote_currency=quote_currency,
        settlement_currency=quote_currency,
        is_inverse=False,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price.from_str(price_increment(price_precision)),
        size_increment=Quantity.from_str(quantity_increment(size_precision)),
        ts_event=0,
        ts_init=0,
        max_quantity=None,
        min_quantity=Quantity.from_str(quantity_increment(size_precision)),
        max_notional=None,
        min_notional=Money(0, quote_currency),
        max_price=Price.from_str(max_price),
        min_price=Price.from_str(price_increment(price_precision)),
        margin_init=Decimal("0.0500"),
        margin_maint=Decimal("0.0250"),
        maker_fee=Decimal("0.000200"),
        taker_fee=Decimal("0.000500"),
    )


def make_bar_type(instrument: CryptoPerpetual) -> BarType:
    return BarType(
        instrument.id,
        BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
    )


def bars_from_frame(
    frame: pd.DataFrame,
    instrument: CryptoPerpetual,
    bar_type: BarType,
) -> list[Bar]:
    bars: list[Bar] = []
    if "ts_event_ns" not in frame.columns:
        raise ValueError("Backtest frame must contain ts_event_ns.")
    for row in frame.itertuples(index=False):
        ts_event = int(row.ts_event_ns)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(float(row.open)),
                high=instrument.make_price(float(row.high)),
                low=instrument.make_price(float(row.low)),
                close=instrument.make_price(float(row.close)),
                volume=instrument.make_qty(float(row.volume)),
                ts_event=ts_event,
                ts_init=ts_event,
            ),
        )
    return bars


def trade_size_for_notional(
    instrument: CryptoPerpetual,
    frame: pd.DataFrame,
    trade_notional: Decimal,
) -> Decimal:
    median_close = Decimal(str(float(frame["close"].median())))
    if median_close <= 0:
        raise ValueError(f"Invalid median close for {instrument.id}: {median_close}")
    raw_size = trade_notional / median_close
    qty = instrument.make_qty(raw_size)
    if qty.raw == 0:
        qty = instrument.min_quantity
    return Decimal(str(qty))


def min_position_change_for_instrument(
    instrument: CryptoPerpetual,
    configured: Any,
) -> Decimal:
    if configured is None:
        return Decimal(str(instrument.size_increment))
    if isinstance(configured, str) and configured.strip().lower() in {
        "size_increment",
        "min_quantity",
    }:
        if configured.strip().lower() == "min_quantity" and instrument.min_quantity is not None:
            return Decimal(str(instrument.min_quantity))
        return Decimal(str(instrument.size_increment))
    return Decimal(str(configured))


def money_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal(0)
    return Money.from_str(str(value)).as_decimal()


def positions_pnl_by_symbol(positions_report: pd.DataFrame) -> dict[str, str]:
    if positions_report.empty or "instrument_id" not in positions_report or "realized_pnl" not in positions_report:
        return {}
    totals: dict[str, Decimal] = {}
    for instrument_id, pnl in zip(
        positions_report["instrument_id"],
        positions_report["realized_pnl"],
        strict=True,
    ):
        symbol = str(instrument_id).split("-PERP", maxsplit=1)[0]
        totals[symbol] = totals.get(symbol, Decimal(0)) + money_decimal(pnl)
    return {symbol: str(value) for symbol, value in sorted(totals.items())}


def fills_turnover_statistics(fills_report: pd.DataFrame, starting_balance: Decimal) -> dict[str, Any]:
    if fills_report.empty:
        return {
            "status": "no_fills",
            "gross_notional_usdt": None,
            "turnover": None,
        }
    price_column = next((column for column in ("last_px", "price", "avg_px") if column in fills_report), None)
    quantity_column = next((column for column in ("last_qty", "quantity", "filled_qty") if column in fills_report), None)
    if price_column is None or quantity_column is None:
        return {
            "status": "missing_price_or_quantity_columns",
            "gross_notional_usdt": None,
            "turnover": None,
            "available_columns": list(fills_report.columns),
        }
    price = pd.to_numeric(fills_report[price_column], errors="coerce")
    quantity = pd.to_numeric(fills_report[quantity_column], errors="coerce")
    notional = (price * quantity).abs().dropna()
    gross_notional = float(notional.sum()) if not notional.empty else 0.0
    balance = float(starting_balance)
    return {
        "status": "ok",
        "price_column": price_column,
        "quantity_column": quantity_column,
        "fill_count_used": len(notional),
        "gross_notional_usdt": str(Decimal(str(gross_notional))),
        "turnover": gross_notional / balance if balance > 0.0 else None,
    }


def execution_cost_assumptions(strategy_config: dict[str, Any]) -> dict[str, Any]:
    cost_config = section(strategy_config, "execution_costs")
    return {
        "round_trip_cost": round_trip_cost_from_execution_config(cost_config),
        "slippage_per_side_rate": cost_config.get("slippage_per_side_rate"),
        "maker_fee_rate": cost_config.get("maker_fee_rate"),
        "taker_fee_rate": cost_config.get("taker_fee_rate"),
        "fee_liquidity": cost_config.get("fee_liquidity"),
        "entry_order_type": section(strategy_config, "backtest").get("entry_order_type", "market"),
    }


def position_return_series(positions_report: pd.DataFrame, starting_balance: Decimal) -> pd.Series:
    if positions_report.empty or "realized_pnl" not in positions_report:
        return pd.Series(dtype=float)
    balance = float(starting_balance)
    if balance <= 0.0:
        return pd.Series(dtype=float)
    pnl = np.asarray([float(money_decimal(value)) for value in positions_report["realized_pnl"]], dtype=float)
    return pd.Series(pnl / balance)


def _timestamp_ns_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for column in candidates:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().all():
            return numeric.astype("int64")
        timestamps = pd.to_datetime(frame[column], errors="coerce", utc=True, format="mixed")
        if timestamps.notna().all():
            return timestamps.astype("int64")
    return None


def position_event_span_arrays(
    positions_report: pd.DataFrame,
    bars: list[Bar],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if positions_report.empty or not bars:
        return None, None
    bar_ns = np.asarray([int(bar.ts_event) for bar in bars], dtype=np.int64)
    if bar_ns.size == 0:
        return None, None
    starts_ns = _timestamp_ns_column(positions_report, ("ts_init", "ts_opened"))
    ends_ns = _timestamp_ns_column(positions_report, ("ts_last", "ts_closed"))
    if starts_ns is None or ends_ns is None:
        return None, None
    starts = np.searchsorted(bar_ns, starts_ns.to_numpy(dtype=np.int64), side="left").astype(np.int64)
    ends = np.searchsorted(bar_ns, ends_ns.to_numpy(dtype=np.int64), side="right").astype(np.int64)
    ends = np.maximum(ends, starts)
    return starts, ends


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = values - running_max
    return float(np.min(drawdown))


def position_statistics(positions_report: pd.DataFrame) -> dict[str, Any]:
    if positions_report.empty or "realized_pnl" not in positions_report:
        return {
            "n_positions": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "max_drawdown_usdt": "0",
        }
    pnl = np.array([float(money_decimal(value)) for value in positions_report["realized_pnl"]])
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    equity = np.cumsum(pnl)
    return {
        "n_positions": len(pnl),
        "winning_positions": len(wins),
        "losing_positions": len(losses),
        "flat_positions": int(np.sum(pnl == 0.0)),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "mean_position_pnl_usdt": str(Decimal(str(float(np.mean(pnl)))) if len(pnl) else Decimal(0)),
        "median_position_pnl_usdt": str(Decimal(str(float(np.median(pnl)))) if len(pnl) else Decimal(0)),
        "gross_profit_usdt": str(Decimal(str(gross_profit))),
        "gross_loss_usdt": str(Decimal(str(gross_loss))),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "max_drawdown_usdt": str(Decimal(str(max_drawdown(equity)))),
    }


def run_symbol(
    *,
    symbol: str,
    input_csv: Path,
    output_dir: Path,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    symbol = normalize_symbol(symbol)
    workflow_config = section(strategy_config, "real_data_workflow")
    price_column = args.price_column or str(workflow_config.get("price_column", "close"))
    meta_config = section(strategy_config, "meta_model")
    embargo_bars = int(args.embargo_bars or meta_config.get("embargo_bars", 80))
    event_config = runtime_event_definition(strategy_config)
    runtime_strategy_config = strategy_config
    exit_config, vertical_bars = directional_barrier_settings(strategy_config, args)

    print(f"{symbol}: loading {input_csv}", flush=True)
    raw_frame = read_real_bars(input_csv, price_column=price_column)
    frame = frame_with_datetime_index(raw_frame)
    frame = filter_frame_by_time(frame, start=args.start, end=args.end)
    window = make_train_test_window(
        frame.index,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        test_days=args.test_days,
        train_days=args.train_days,
        embargo_bars=embargo_bars,
    )
    print(
        f"{symbol}: bars={len(frame):,} train={window.train_start} -> {window.train_end} "
        f"test={window.test_start} -> {window.test_end} embargo_bars={embargo_bars}",
        flush=True,
    )

    close = frame[price_column].dropna().astype(float)
    volatility_started = progress_start(args, f"{symbol}: volatility target build")
    volatility = make_ewma_volatility_target(
        close,
        span=args.volatility_ewma_span,
        lookback_days=args.volatility_lookback_days,
        train_start=window.train_start,
        train_end=window.train_end,
        floor_quantile=args.volatility_floor_quantile,
    )
    progress_done(args, f"{symbol}: volatility target build", volatility_started)
    volatility_summary = volatility_target_stage_summary(
        volatility,
        args=args,
        train_start=window.train_start,
        train_end=window.train_end,
    )
    print_volatility_target_summary(symbol, volatility_summary)

    print(f"{symbol}: building AFML/CVD feature matrix", flush=True)
    feature_started = progress_start(args, f"{symbol}: feature matrix build")
    features = build_feature_matrix(
        frame=frame,
        config=config,
        strategy_config=runtime_strategy_config,
        price_column=price_column,
        volatility=volatility,
        args=args,
    )
    progress_done(args, f"{symbol}: feature matrix build", feature_started)
    print(f"{symbol}: features={features.shape[1]:,}", flush=True)

    cost_config = section(strategy_config, "execution_costs")
    meta_label_min_ret = (
        float(args.meta_label_min_ret)
        if args.meta_label_min_ret is not None
        else round_trip_cost_from_execution_config(cost_config)
    )
    meta_probability_threshold = (
        float(args.meta_probability_threshold)
        if args.meta_probability_threshold is not None
        else float(meta_config.get("class_probability_threshold", 0.5))
    )

    primary_mode = (
        str(primary_model_config(strategy_config).get("mode", PRIMARY_MOE_MODEL_NAME))
        if primary_model_enabled(strategy_config)
        else primary_rule_mode(strategy_config)
    )
    print(
        f"{symbol}: fitting primary side model mode={primary_mode}",
        flush=True,
    )
    primary_training = train_primary_side_model(
        close=close,
        features=features,
        volatility=volatility,
        exit_config=exit_config,
        vertical_bars=vertical_bars,
        window=window,
        event_config=event_config,
        strategy_config=strategy_config,
        args=args,
        output_dir=output_dir,
        symbol=symbol,
        round_trip_cost=meta_label_min_ret,
    )
    primary_frame = primary_training.frame
    primary_side = primary_frame["primary_side"]
    candidate_side = primary_side[primary_side != 0]
    print(
        f"{symbol}: primary candidates={len(candidate_side):,} "
        f"sides={class_counts(candidate_side) if not candidate_side.empty else {}} "
        f"abstain_rate={primary_training.diagnostics.get('activity', {}).get('all', {}).get('abstain_rate')}",
        flush=True,
    )
    print_primary_stage_summary(symbol, primary_training.diagnostics)
    if candidate_side.empty:
        empty_index = pd.DatetimeIndex(
            primary_frame.index[
                (primary_frame.index >= window.test_start)
                & (primary_frame.index <= window.test_end)
            ],
        )
        empty_signals = pd.DataFrame(
            {
                "symbol": symbol,
                "signal": 0,
                "prediction": 0,
                "confidence": 0.0,
                "primary_side": 0,
                "primary_policy": "none",
                "primary_market_state": "unknown",
                "meta_probability": 0.0,
                "meta_prediction": 0,
                "trend_side": 0,
                "trend_confidence": 0.0,
                "reversion_side": 0,
                "reversion_confidence": 0.0,
                "trend_score": 0.0,
                "reversion_score": 0.0,
                "selected_policy_oof_edge": 0.0,
                "selected_policy_profit_factor": np.nan,
                "selected_policy_trade_count": 0,
                "volatility_state": "unknown",
                "flow_state": "unknown",
                "entropy_state": "unknown",
                "approved_threshold": np.nan,
                "diagnostic_threshold": 0.0,
                "approved_trading": False,
                "signal_source": "none",
                "gate_status": "rejected_by_afml_gate",
                "diagnostic_signal": 0,
                "diagnostic_meta_prediction": 0,
                "bet_size": 0.0,
                "bet_size_abs": 0.0,
                "bet_size_probability_status": "no_primary_candidate",
                "diagnostic_bet_size": 0.0,
                "diagnostic_bet_size_abs": 0.0,
                "diagnostic_bet_size_probability_status": "no_primary_candidate",
                "label_ret": np.nan,
                "label_cost": float(meta_label_min_ret),
                "label_net_ret": np.nan,
                "meta_label": pd.NA,
                "cost_adjusted_meta_label": pd.NA,
                "event_end_time": pd.NaT,
                "barrier_outcome": pd.NA,
            },
            index=empty_index,
        )
        signal_path = output_dir / f"{symbol}_signals_test.csv"
        write_signal_csv(empty_signals, signal_path)
        gate = {
            "enabled": True,
            "action": "skip_real_backtest",
            "approved_for_backtest": False,
            "approval_status": "rejected_by_afml_gate",
            "blockers": ["no_primary_candidates"],
            "warnings": [],
        }
        research_diagnosis = {
            "status": "rejected_by_afml_gate",
            "approved_for_backtest": False,
            "failed_stages": ["primary", "approved_signal_generation"],
            "blockers": ["no_primary_candidates"],
            "primary_failure_reasons": ["no_primary_candidates"],
            "threshold_decision": {},
            "approved_active_signals": 0,
            "diagnostic_active_signals": 0,
        }
        diagnostic_failure = diagnostic_failure_report(
            empty_signals,
            validation_config=section(strategy_config, "validation"),
            n_trials=1,
            round_trip_cost=meta_label_min_ret,
        )
        print_afml_acceptance_gate_summary(symbol, gate)
        print_diagnostic_failure_report(symbol, diagnostic_failure)
        print(f"{symbol}: wrote empty test signals -> {signal_path}", flush=True)
        return {
            "symbol": symbol,
            "input_csv": input_csv,
            "signal_csv": signal_path,
            "price_column": price_column,
            "n_bars": len(frame),
            "n_features": int(features.shape[1]),
            "window": window_to_dict(window),
            "exit_definition": exit_config,
            "vertical_barrier_bars": vertical_bars,
            "volatility_target": volatility_summary,
            "meta_label_min_ret": meta_label_min_ret,
            "primary": primary_training.diagnostics,
            "meta": {"status": "skipped_no_primary_candidates"},
            "signals": {
                "test_rows": len(empty_signals),
                "signal_counts": signal_counts(empty_signals),
                "active_signals": 0,
                "diagnostic_signal_counts": class_counts(empty_signals["diagnostic_signal"]),
            "diagnostic_active_signals": 0,
            "approved_trading": False,
            "signal_source": "none",
            "driver_expert_counts": {},
            "diagnostic_driver_expert_counts": {},
            "policy_counts": {},
            "diagnostic_policy_counts": {},
            "market_state_counts": {},
            "diagnostic_market_state_counts": {},
        },
            "diagnostic_failure_report": diagnostic_failure,
            "afml_acceptance_gate": gate,
            "research_diagnosis": research_diagnosis,
            "model_controls": {
                "primary_model": primary_mode,
                "meta_model": "skipped_no_primary_candidates",
                "probability_calibration": {"enabled": False},
            },
        }
    if primary_training.dataset is None:
        meta_features = make_rule_meta_features(features, primary_frame, strategy_config)
    else:
        meta_features = make_primary_meta_features(features, primary_frame)
        meta_features = apply_feature_pattern_exclusions(meta_features, strategy_config)
    print(
        f"{symbol}: labeling primary candidates for meta model "
        f"long_pt={exit_config['long_profit_taking_mult']:g} "
        f"long_sl={exit_config['long_stop_loss_mult']:g} "
        f"short_pt={exit_config['short_profit_taking_mult']:g} "
        f"short_sl={exit_config['short_stop_loss_mult']:g} "
        f"vertical_bars={vertical_bars}",
        flush=True,
    )
    meta_dataset_started = progress_start(args, f"{symbol}: meta dataset build")
    meta_dataset = build_afml_meta_dataset(
        close,
        primary_side,
        features=meta_features,
        pt_sl=exit_config,
        target=volatility,
        min_ret=args.min_ret,
        meta_label_min_ret=meta_label_min_ret,
        vertical_barrier_days=None,
        vertical_barrier_bars=vertical_bars,
        oldest_weight=args.oldest_weight,
        feature_fracdiff_d=args.feature_fracdiff_d,
        feature_fracdiff_threshold=args.feature_fracdiff_threshold,
        return_type="log",
    )
    progress_done(args, f"{symbol}: meta dataset build", meta_dataset_started)
    train_index, test_index = split_dataset_indices_for_window(meta_dataset, window)
    print(
        f"{symbol}: fitting meta model labels={len(meta_dataset.y):,} "
        f"classes={class_counts(meta_dataset.y)} "
        f"train={len(train_index):,} test={len(test_index):,}",
        flush=True,
    )
    meta_training = fit_rule_meta_model_with_feature_selection(
        meta_dataset=meta_dataset,
        train_index=train_index,
        test_index=test_index,
        strategy_config=strategy_config,
        args=args,
        output_dir=output_dir,
        symbol=symbol,
        meta_probability_threshold=meta_probability_threshold,
        meta_label_min_ret=meta_label_min_ret,
    )
    meta_train_metrics = classification_metrics(
        meta_dataset.y.loc[train_index],
        meta_training.train_pred,
        positive_label=1,
    )
    meta_test_metrics = classification_metrics(
        meta_dataset.y.loc[test_index],
        meta_training.test_pred,
        positive_label=1,
    )
    validation_n_trials = (
        int(section(strategy_config, "primary_optuna").get("n_trials", 1))
        if bool(section(strategy_config, "primary_optuna").get("enabled", False))
        else 1
    )
    validation_started = progress_start(args, f"{symbol}: meta validation diagnostics")
    validation = dataset_validation_diagnostics(
        meta_dataset,
        test_index,
        strategy_config=strategy_config,
        n_trials=validation_n_trials,
        round_trip_cost=meta_label_min_ret,
    )
    progress_done(args, f"{symbol}: meta validation diagnostics", validation_started)
    weights = {
        "all": sample_weight_summary(meta_dataset.sample_weight),
        "train": sample_weight_summary(meta_dataset.sample_weight.loc[train_index]),
        "test": sample_weight_summary(meta_dataset.sample_weight.loc[test_index]),
    }
    importance_completeness = feature_importance_completeness(meta_training.diagnostics)
    approved_probability_threshold = meta_training.diagnostics.get("approved_probability_threshold")
    diagnostic_probability_threshold = float(
        meta_training.diagnostics.get("diagnostic_probability_threshold", meta_training.probability_threshold),
    )
    approved_probability_upper_threshold = meta_training.diagnostics.get(
        "approved_probability_upper_threshold",
    )
    diagnostic_probability_upper_threshold = meta_training.diagnostics.get(
        "diagnostic_probability_upper_threshold",
    )
    approved_trading = approved_probability_threshold is not None
    approved_probability_threshold_value = (
        float(approved_probability_threshold)
        if approved_probability_threshold is not None
        else diagnostic_probability_threshold
    )

    signal_started = progress_start(args, f"{symbol}: signal generation")
    raw_signals = make_rule_meta_signal_frame(
        primary_frame,
        meta_training.model,
        meta_features,
        meta_probability_threshold=approved_probability_threshold_value,
        active_index=test_index,
        feature_columns=meta_training.feature_columns,
        approved_trading=approved_trading,
        diagnostic_probability_threshold=diagnostic_probability_threshold,
        approved_probability_upper_threshold=approved_probability_upper_threshold,
        diagnostic_probability_upper_threshold=diagnostic_probability_upper_threshold,
        approved_excluded_filter_columns=meta_training.diagnostics.get(
            "approved_excluded_filter_columns",
        ),
        diagnostic_excluded_filter_columns=meta_training.diagnostics.get(
            "diagnostic_excluded_filter_columns",
        ),
    )
    probability_status = (
        "calibrated_probability"
        if meta_training.diagnostics.get("probability_calibration", {}).get("status") == "calibrated"
        else "uncalibrated_probability"
    )
    signals = make_bet_size_frame(
        raw_signals,
        probability_col="meta_probability",
        probability_status=probability_status,
        min_abs_size=args.bet_size_min_abs,
        max_abs_size=args.bet_size_max_abs,
        step_size=args.bet_size_step,
    )
    signals = add_diagnostic_bet_size_columns(
        signals,
        step_size=args.bet_size_step,
        min_abs_size=args.bet_size_min_abs,
        max_abs_size=args.bet_size_max_abs,
        probability_status=probability_status,
    )
    progress_done(args, f"{symbol}: signal generation", signal_started)
    test_signals = signals.loc[signals.index.intersection(test_index)].copy()
    test_signals.insert(0, "symbol", symbol)
    test_signals["trgt"] = volatility.reindex(test_signals.index).ffill().bfill()
    test_signals = apply_directional_exit_columns(test_signals, exit_config)
    test_signals = annotate_signal_outcomes(
        test_signals,
        meta_dataset,
        round_trip_cost=meta_label_min_ret,
    )
    signal_path = output_dir / f"{symbol}_signals_test.csv"
    test_signal_counts = signal_counts(test_signals)
    active_signal_count = int((test_signals["signal"].astype(int) != 0).sum())
    diagnostic_active_signal_count = (
        int((test_signals["diagnostic_signal"].astype(int) != 0).sum())
        if "diagnostic_signal" in test_signals
        else 0
    )
    validation_config = section(strategy_config, "validation")
    proxy_started = progress_start(args, f"{symbol}: signal proxy performance")
    signal_proxy = signal_proxy_performance(
        meta_dataset=meta_dataset,
        test_signals=test_signals,
        validation_config=validation_config,
        n_trials=validation_n_trials,
        round_trip_cost=meta_label_min_ret,
    )
    diagnostic_proxy_signals = test_signals.copy()
    if "diagnostic_signal" in diagnostic_proxy_signals:
        diagnostic_proxy_signals["signal"] = diagnostic_proxy_signals["diagnostic_signal"]
    diagnostic_signal_proxy = signal_proxy_performance(
        meta_dataset=meta_dataset,
        test_signals=diagnostic_proxy_signals,
        validation_config=validation_config,
        n_trials=validation_n_trials,
        round_trip_cost=meta_label_min_ret,
    )
    progress_done(args, f"{symbol}: signal proxy performance", proxy_started)
    primary_before_meta_pred = pd.Series(1, index=test_index, name="prediction")
    primary_before_meta_metrics = classification_metrics(
        meta_dataset.y.loc[test_index],
        primary_before_meta_pred,
        positive_label=1,
    )
    acceptance_config = validation_config.get("acceptance_gate", {})
    acceptance_config = acceptance_config if isinstance(acceptance_config, dict) else {}
    max_accepted_rate = float(
        acceptance_config.get(
            "max_accepted_rate",
            threshold_selection_config(strategy_config).get("max_accepted_rate", 0.60),
        ),
    )
    raw_meta_collapse = meta_gate_collapse_summary(
        meta_training.test_pred,
        max_accepted_rate=max_accepted_rate,
    )
    meta_collapse = meta_gate_collapse_summary(
        test_signals["meta_prediction"],
        max_accepted_rate=max_accepted_rate,
    )
    afml_gate = afml_acceptance_gate_summary(
        validation_config=validation_config,
        meta_test_metrics=meta_test_metrics,
        primary_before_meta_metrics=primary_before_meta_metrics,
        meta_collapse=meta_collapse,
        signal_proxy=signal_proxy,
        active_signal_count=active_signal_count,
        threshold_selection=meta_training.diagnostics.get("threshold_selection"),
        primary_diagnostics=primary_training.diagnostics,
    )
    research_diagnosis = research_diagnosis_summary(
        primary_diagnostics=primary_training.diagnostics,
        threshold_selection=meta_training.diagnostics.get("threshold_selection"),
        signal_proxy=signal_proxy,
        diagnostic_signal_proxy=diagnostic_signal_proxy,
        afml_gate=afml_gate,
        active_signal_count=active_signal_count,
        diagnostic_active_signal_count=diagnostic_active_signal_count,
    )
    candidate_signal_counts = test_signal_counts
    candidate_active_signal_count = active_signal_count
    test_signals = apply_afml_gate_to_signal_frame(test_signals, afml_gate)
    test_signal_counts = signal_counts(test_signals)
    active_signal_count = int((test_signals["signal"].astype(int) != 0).sum())
    diagnostic_failure = diagnostic_failure_report(
        test_signals,
        validation_config=validation_config,
        n_trials=validation_n_trials,
        round_trip_cost=meta_label_min_ret,
    )
    write_signal_csv(test_signals, signal_path)

    print_meta_stage_summary(
        symbol,
        train_metrics=meta_train_metrics,
        test_metrics=meta_test_metrics,
        accepted_count=active_signal_count,
        signal_counts_=test_signal_counts,
    )
    print_validation_stage_summary(symbol, validation, label="META TEST")
    print_sample_weight_stage_summary(symbol, weights, label="META")
    print_signal_proxy_summary(symbol, signal_proxy)
    print_signal_proxy_summary(symbol, diagnostic_signal_proxy, label="DIAGNOSTIC SIGNAL PROXY")
    print_afml_acceptance_gate_summary(symbol, afml_gate)
    print_diagnostic_failure_report(symbol, diagnostic_failure)

    print(
        f"{symbol}: done meta_test_precision={meta_test_metrics['positive_precision']:.4f} "
        f"meta_test_recall={meta_test_metrics['positive_recall']:.4f} "
        f"meta_test_f1={meta_test_metrics['positive_f1']:.4f} "
        f"test_signals={test_signal_counts}",
        flush=True,
    )
    print(f"{symbol}: wrote test signals -> {signal_path}", flush=True)

    return {
        "symbol": symbol,
        "input_csv": input_csv,
        "signal_csv": signal_path,
        "price_column": price_column,
        "n_bars": len(frame),
        "n_features": int(features.shape[1]),
        "window": window_to_dict(window),
        "exit_definition": exit_config,
        "vertical_barrier_bars": vertical_bars,
        "volatility_target": volatility_summary,
        "meta_label_min_ret": meta_label_min_ret,
        "primary": primary_training.diagnostics,
        "meta": {
            "n_labels": len(meta_dataset.y),
            "label_counts": class_counts(meta_dataset.y),
            "train_events": len(train_index),
            "test_events": len(test_index),
            "train_precision": meta_train_metrics["positive_precision"],
            "train_recall": meta_train_metrics["positive_recall"],
            "test_precision": meta_test_metrics["positive_precision"],
            "test_recall": meta_test_metrics["positive_recall"],
            "train_metrics": meta_train_metrics,
            "test_metrics": meta_test_metrics,
            "primary_before_meta_test_metrics": primary_before_meta_metrics,
            "probability_threshold": meta_training.probability_threshold,
            "approved_probability_threshold": approved_probability_threshold,
            "diagnostic_probability_threshold": diagnostic_probability_threshold,
            "configured_probability_threshold": meta_probability_threshold,
            "threshold_decision": meta_training.diagnostics.get("threshold_decision"),
            "threshold_selection": meta_training.diagnostics.get("threshold_selection"),
            "collapse": meta_collapse,
            "raw_collapse": raw_meta_collapse,
            "feature_selection": meta_training.diagnostics,
            "feature_importance_completeness": importance_completeness,
            "sample_weight_summary": weights,
            "validation": validation,
            "signal_proxy_performance": signal_proxy,
            "diagnostic_signal_proxy_performance": diagnostic_signal_proxy,
            "diagnostic_failure_report": diagnostic_failure,
        },
        "signals": {
            "test_rows": len(test_signals),
            "signal_counts": test_signal_counts,
            "active_signals": active_signal_count,
            "approved_active_signals": active_signal_count,
            "candidate_signal_counts_before_gate": candidate_signal_counts,
            "candidate_active_signals_before_gate": candidate_active_signal_count,
            "diagnostic_signal_counts": class_counts(test_signals["diagnostic_signal"])
            if "diagnostic_signal" in test_signals
            else {},
            "diagnostic_active_signals": diagnostic_active_signal_count,
            "approved_trading": bool(afml_gate.get("approved_for_backtest", False)),
            "signal_source": "approved_signal"
            if bool(afml_gate.get("approved_for_backtest", False))
            else "diagnostic_signal",
            "driver_expert_counts": (
                {
                    driver: int(count)
                    for driver, count in test_signals.loc[
                        test_signals["signal"].astype(int) != 0,
                        "primary_driver_expert",
                    ]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                    if driver
                }
                if "primary_driver_expert" in test_signals
                else {}
            ),
            "diagnostic_driver_expert_counts": (
                {
                    driver: int(count)
                    for driver, count in test_signals.loc[
                        test_signals["diagnostic_signal"].astype(int) != 0,
                        "primary_driver_expert",
                    ]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                    if driver
                }
                if "diagnostic_signal" in test_signals and "primary_driver_expert" in test_signals
                else {}
            ),
            "policy_counts": (
                {
                    policy: int(count)
                    for policy, count in test_signals.loc[
                        test_signals["signal"].astype(int) != 0,
                        "primary_policy",
                    ]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                    if policy
                }
                if "primary_policy" in test_signals
                else {}
            ),
            "diagnostic_policy_counts": (
                {
                    policy: int(count)
                    for policy, count in test_signals.loc[
                        test_signals["diagnostic_signal"].astype(int) != 0,
                        "primary_policy",
                    ]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                    if policy
                }
                if "diagnostic_signal" in test_signals and "primary_policy" in test_signals
                else {}
            ),
            "market_state_counts": (
                {
                    state: int(count)
                    for state, count in test_signals.loc[
                        test_signals["signal"].astype(int) != 0,
                        "primary_market_state",
                    ]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                    if state
                }
                if "primary_market_state" in test_signals
                else {}
            ),
            "diagnostic_market_state_counts": (
                {
                    state: int(count)
                    for state, count in test_signals.loc[
                        test_signals["diagnostic_signal"].astype(int) != 0,
                        "primary_market_state",
                    ]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                    if state
                }
                if "diagnostic_signal" in test_signals and "primary_market_state" in test_signals
                else {}
            ),
        },
        "diagnostic_failure_report": diagnostic_failure,
        "afml_acceptance_gate": afml_gate,
        "research_diagnosis": research_diagnosis,
        "model_controls": {
            "primary_model": primary_mode,
            "meta_model": str(meta_config.get("family", META_SEQUENTIAL_BOOTSTRAP_RF)),
            "max_features": parse_model_scalar(args.max_features or meta_config.get("max_features", 1)),
            "max_samples": resolve_max_samples(args.max_samples, meta_config.get("max_samples")),
            "min_weight_fraction_leaf": args.min_weight_fraction_leaf,
            "class_weight": resolve_class_weight(args.class_weight, meta_config.get("class_weight")),
            "bet_size": {
                "min_abs_size": args.bet_size_min_abs,
                "max_abs_size": args.bet_size_max_abs,
                "step_size": args.bet_size_step,
            },
            "sequential_bootstrap": {
                "primary": bool(primary_training.dataset is not None),
                "meta": True,
            },
            "sample_weight": "return_attribution_x_time_decay_or_uniqueness",
            "probability_calibration": meta_training.diagnostics.get("probability_calibration", {}),
        },
    }


def run_real_backtest(
    summaries: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("Cannot run backtest without trained symbol summaries.")
    allow_rejected_backtest = bool(args.allow_rejected_backtest) or diagnostic_backtest_enabled_on_rejection(
        strategy_config,
    )

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("CVDSLOPE-001"),
            logging=LoggingConfig(
                log_level=args.log_level,
                bypass_logging=args.log_level == "CRITICAL",
            ),
        ),
    )
    starting_balance = Money(args.starting_balance, USDT)
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[starting_balance],
        default_leverage=args.default_leverage,
        book_type=BookType.L1_MBP,
        bar_execution=True,
        trade_execution=True,
    )

    strategy_config_path = str(strategy_config.get("_strategy_config_path", "afml_strategies/cvdslope_nooptuna.json"))
    backtest_config = section(strategy_config, "backtest")
    flatten_on_zero = bool(backtest_config.get("flatten_on_zero", False))
    entry_order_type = str(backtest_config.get("entry_order_type", "market"))
    entry_limit_post_only = bool(backtest_config.get("entry_limit_post_only", False))
    entry_limit_offset_bps = float(backtest_config.get("entry_limit_offset_bps", 0.0))
    all_bars: list[Bar] = []
    symbol_rows: list[dict[str, Any]] = []
    for summary in summaries:
        symbol = str(summary["symbol"])
        input_csv = resolve_repo_path(summary["input_csv"])
        signal_csv, used_unapproved_diagnostic, diagnostic_reason = backtest_signal_csv_for_summary(
            summary,
            output_dir=output_dir,
            allow_rejected_backtest=allow_rejected_backtest,
        )
        if not summary_approved_for_backtest(summary) and not used_unapproved_diagnostic:
            print(
                f"REAL BACKTEST SYMBOL SKIPPED: symbol={symbol} reason={diagnostic_reason}",
                flush=True,
            )
            symbol_rows.append(
                {
                    "symbol": symbol,
                    "skipped": True,
                    "skip_reason": diagnostic_reason,
                    "signal_csv": signal_csv,
                    "used_unapproved_diagnostic_signal": False,
                },
            )
            continue
        window = summary["window"]
        raw_frame = read_real_bars(input_csv, price_column=str(summary["price_column"]))
        frame = frame_with_datetime_index(raw_frame)
        frame = filter_frame_by_time(frame, start=window["test_start"], end=window["test_end"])
        instrument = make_instrument(symbol, input_csv)
        engine.add_instrument(instrument)
        bar_type = make_bar_type(instrument)
        bars = bars_from_frame(frame, instrument, bar_type)
        trade_size = trade_size_for_notional(instrument, frame, args.trade_notional)
        min_position_change = min_position_change_for_instrument(
            instrument,
            args.min_position_change,
        )
        strategy_config_obj = AfmlSignalStrategyConfig(
            strategy_id=f"CVDSLOPE-{symbol}",
            instrument_id=instrument.id,
            bar_type=bar_type,
            signal_path=str(signal_csv),
            trade_size=trade_size,
            strategy_config_path=strategy_config_path,
            confidence_threshold=args.confidence_threshold,
            flatten_on_zero=flatten_on_zero,
            profit_taking_mult=None,
            stop_loss_mult=None,
            vertical_barrier_bars=int(summary["vertical_barrier_bars"]),
            vertical_barrier_days=None,
            order_time_in_force=TimeInForce.GTC,
            min_position_change=min_position_change,
            close_positions_on_stop=True,
            reduce_only_on_stop=True,
            entry_order_type=entry_order_type,
            entry_limit_post_only=entry_limit_post_only,
            entry_limit_offset_bps=entry_limit_offset_bps,
        )
        engine.add_strategy(CvdSlopeAfmlSignalStrategy(config=strategy_config_obj))
        all_bars.extend(bars)
        symbol_rows.append(
            {
                "symbol": symbol,
                "bars": len(bars),
                "drb_start": str(frame.index.min()) if not frame.empty else None,
                "drb_end": str(frame.index.max()) if not frame.empty else None,
                "trade_size": str(trade_size),
                "min_position_change": str(min_position_change),
                "signal_csv": signal_csv,
                "used_unapproved_diagnostic_signal": used_unapproved_diagnostic,
                "diagnostic_signal_reason": diagnostic_reason,
                "instrument_id": str(instrument.id),
            },
        )

    all_bars.sort(key=lambda bar: int(bar.ts_event))
    if not all_bars:
        engine.dispose()
        return {
            "status": "skipped_no_backtestable_symbols",
            "unapproved_diagnostic_backtest": False,
            "symbols": symbol_rows,
        }

    backtest_start = min(pd.Timestamp(summary["window"]["test_start"]) for summary in summaries)
    backtest_end = max(pd.Timestamp(summary["window"]["test_end"]) for summary in summaries)
    print(
        f"REAL BACKTEST: bars={len(all_bars):,} start={backtest_start} end={backtest_end}",
        flush=True,
    )
    engine.add_data(all_bars, validate=False, sort=True)
    engine.run(start=backtest_start, end=backtest_end)

    fills_report = engine.trader.generate_order_fills_report()
    positions_report = engine.trader.generate_positions_report()
    account_report = engine.trader.generate_account_report(BINANCE)
    fills_path = output_dir / "fills.csv"
    positions_path = output_dir / "positions.csv"
    account_path = output_dir / "account.csv"
    fills_report.to_csv(fills_path)
    positions_report.to_csv(positions_path)
    account_report.to_csv(account_path)

    account = engine.cache.account_for_venue(BINANCE)
    ending_balance = account.balance_total(USDT)
    realized_pnl = ending_balance - starting_balance
    realized_pnl_decimal = realized_pnl.as_decimal()
    return_pct = (
        (realized_pnl_decimal / args.starting_balance) * Decimal(100)
        if args.starting_balance != 0
        else Decimal(0)
    )
    stats = position_statistics(positions_report)
    validation_n_trials = (
        int(section(strategy_config, "primary_optuna").get("n_trials", 1))
        if bool(section(strategy_config, "primary_optuna").get("enabled", False))
        else 1
    )
    position_returns = position_return_series(positions_report, args.starting_balance)
    event_starts, event_ends = position_event_span_arrays(positions_report, all_bars)
    validation = afml_validation_diagnostics(
        position_returns,
        validation_config=section(strategy_config, "validation"),
        n_trials=validation_n_trials,
        event_starts=event_starts,
        event_ends=event_ends,
    )
    backtest_summary = {
        "start": backtest_start,
        "end": backtest_end,
        "starting_balance_usdt": str(args.starting_balance),
        "ending_balance_usdt": str(ending_balance),
        "realized_pnl_usdt": str(realized_pnl),
        "return_pct": str(return_pct),
        "n_bars": len(all_bars),
        "n_fills": len(fills_report),
        "n_positions": len(positions_report),
        "pnl_by_symbol_usdt": positions_pnl_by_symbol(positions_report),
        "position_statistics": stats,
        "validation": validation,
        "turnover": fills_turnover_statistics(fills_report, args.starting_balance),
        "execution_cost_assumptions": execution_cost_assumptions(strategy_config),
        "unapproved_diagnostic_backtest": any(
            bool(row.get("used_unapproved_diagnostic_signal", False)) for row in symbol_rows
        ),
        "symbols": symbol_rows,
        "files": {
            "fills": fills_path,
            "positions": positions_path,
            "account": account_path,
        },
    }
    print(
        "REAL BACKTEST DONE: "
        f"fills={len(fills_report):,} positions={len(positions_report):,} "
        f"ending={ending_balance} pnl={realized_pnl} return={return_pct:.4f}%",
        flush=True,
    )
    print_backtest_stage_summary(backtest_summary)
    engine.dispose()
    return backtest_summary


def backtest_signal_csv_for_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    allow_rejected_backtest: bool,
) -> tuple[Path, bool, str]:
    signal_csv = resolve_repo_path(summary["signal_csv"])
    approved = summary_approved_for_backtest(summary)
    if approved or not allow_rejected_backtest:
        reason = "approved_signal" if approved else "diagnostic_backtest_disabled"
        return signal_csv, False, reason

    frame = pd.read_csv(signal_csv, low_memory=False)
    if "diagnostic_signal" not in frame.columns:
        return signal_csv, False, "missing_diagnostic_signal"
    diagnostic_signal = frame["diagnostic_signal"].fillna(0).astype(int)
    if int((diagnostic_signal != 0).sum()) <= 0:
        return signal_csv, False, "diagnostic_signal_all_zero"
    frame = frame.copy()
    frame["signal"] = diagnostic_signal
    frame["meta_prediction"] = frame.get("diagnostic_meta_prediction", frame["signal"].ne(0)).astype(int)
    if "diagnostic_bet_size" in frame.columns:
        frame["bet_size"] = frame["diagnostic_bet_size"].astype(float)
    if "diagnostic_bet_size_abs" in frame.columns:
        frame["bet_size_abs"] = frame["diagnostic_bet_size_abs"].astype(float)
    frame["unapproved_diagnostic_backtest"] = True
    frame["signal_source"] = np.where(frame["signal"].astype(int) != 0, "diagnostic_signal", "none")
    symbol = str(summary.get("symbol", "symbol"))
    diagnostic_path = output_dir / f"{symbol}_signals_test_unapproved_diagnostic.csv"
    frame.to_csv(diagnostic_path, index=False)
    return diagnostic_path, True, "diagnostic_signal_applied"


def run_backtest_for_summaries(
    summaries: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not args.run_backtest or not summaries:
        return None

    rejected = [summary for summary in summaries if not summary_approved_for_backtest(summary)]
    diagnostic_enabled = diagnostic_backtest_enabled_on_rejection(strategy_config)
    allow_rejected_backtest = bool(args.allow_rejected_backtest) or diagnostic_enabled
    if rejected and not allow_rejected_backtest:
        rejected_symbols = [str(summary.get("symbol")) for summary in rejected]
        print(
            "REAL BACKTEST SKIPPED: rejected_by_afml_gate "
            f"symbols={rejected_symbols} override=--allow-rejected-backtest",
            flush=True,
        )
        return {
            "status": "skipped_rejected_by_afml_gate",
            "rejected_symbols": rejected_symbols,
            "override": "--allow-rejected-backtest",
        }

    runnable_summaries = summaries
    skipped_no_diagnostic = [
        str(summary.get("symbol"))
        for summary in rejected
        if diagnostic_active_signal_count(summary) <= 0
    ]
    if rejected and skipped_no_diagnostic:
        runnable_summaries = [
            summary
            for summary in summaries
            if summary_approved_for_backtest(summary) or diagnostic_active_signal_count(summary) > 0
        ]
        print(
            "REAL BACKTEST DIAGNOSTIC SKIP: "
            f"symbols={skipped_no_diagnostic} reason=diagnostic_signal_all_zero",
            flush=True,
        )

    diagnostic_counts = {
        str(summary.get("symbol")): diagnostic_active_signal_count(summary)
        for summary in rejected
        if diagnostic_active_signal_count(summary) > 0
    }
    if diagnostic_counts:
        print(
            "REAL BACKTEST DIAGNOSTIC: "
            "using unapproved diagnostic_signal for rejected symbols "
            f"counts={diagnostic_counts}",
            flush=True,
        )

    if not runnable_summaries:
        print(
            "REAL BACKTEST SKIPPED: no diagnostic signals available for rejected symbols",
            flush=True,
        )
        return {
            "status": "skipped_no_diagnostic_signals",
            "rejected_symbols": [str(summary.get("symbol")) for summary in rejected],
            "reason": "diagnostic_signal_all_zero",
        }

    backtest_summary = run_real_backtest(
        runnable_summaries,
        config=config,
        strategy_config=strategy_config,
        args=args,
        output_dir=output_dir,
    )
    if skipped_no_diagnostic and isinstance(backtest_summary, dict):
        backtest_summary["skipped_no_diagnostic_symbols"] = skipped_no_diagnostic
    return backtest_summary


def run() -> dict[str, Any]:
    args = parse_args()
    config = load_afml_data_config(args.config)
    strategy_config = load_real_strategy_config(config, args.strategy_config)
    args = apply_json_defaults(args, strategy_config)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = configured_symbols(config, args.symbols)
    if args.input_csv is not None and len(symbols) != 1:
        raise ValueError("--input-csv can only be used with one symbol.")

    print(
        "REAL AFML TRAINING: "
        f"symbols={','.join(symbols)} train={args.train_start}->{args.train_end} "
        f"test={args.test_start}->{args.test_end} backtest={args.run_backtest} "
        f"output_dir={output_dir}",
        flush=True,
    )
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        input_csv = input_csv_for_symbol(
            symbol,
            config,
            cli_input_csv=args.input_csv,
            total_symbols=len(symbols),
        )
        try:
            summaries.append(
                run_symbol(
                    symbol=symbol,
                    input_csv=input_csv,
                    output_dir=output_dir,
                    config=config,
                    strategy_config=strategy_config,
                    args=args,
                ),
            )
        except Exception as exc:
            failures.append({"symbol": normalize_symbol(symbol), "error": str(exc)})
            print(f"{normalize_symbol(symbol)}: FAILED: {exc}", flush=True)

    backtest_summary = run_backtest_for_summaries(
        summaries,
        config=config,
        strategy_config=strategy_config,
        args=args,
        output_dir=output_dir,
    )

    summary = {
        "mode": "train_and_backtest" if args.run_backtest else "training_only_no_backtest",
        "symbols": [normalize_symbol(symbol) for symbol in symbols],
        "configured_window": {
            "data_start": args.start,
            "data_end": args.end,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "test_days": args.test_days,
            "train_days": args.train_days,
        },
        "validation_config": section(strategy_config, "validation"),
        "results": summaries,
        "backtest": backtest_summary,
        "failures": failures,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(f"wrote summary -> {summary_path}", flush=True)
    if not summaries:
        raise RuntimeError("No symbols completed training.")
    return summary


if __name__ == "__main__":
    run()
