from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_scripts.backtest_synthetic_barriers_ml import event_state_features  # noqa: E402
from afml_scripts.backtest_synthetic_barriers_ml import (  # noqa: E402
    round_trip_cost_from_labeling_config,
)
from afml_scripts.generate_ou_residual_synthetic import configured_symbols  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import input_csv_for_symbol  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import normalize_symbol  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import read_real_bars  # noqa: E402
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
from nautilus_trader.research.afml_pipeline import AfmlDataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import (  # noqa: E402
    SequentialBootstrapBaggingClassifier,
)
from nautilus_trader.research.afml_pipeline import _pca_dataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import build_afml_meta_dataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import make_bet_size_frame  # noqa: E402
from nautilus_trader.research.afml_pipeline import (  # noqa: E402
    make_ewma_1bar_log_return_volatility_target,
)
from nautilus_trader.research.afml_pipeline import make_pipeline_features  # noqa: E402
from nautilus_trader.research.afml_pipeline import mda_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import mdi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import pca_mdi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import sfi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import write_signal_csv  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("afml_strategies/output/cvdslope_main")
DEFAULT_REAL_STRATEGY_CONFIG = Path("afml_strategies/cvdslope_nooptuna.json")
DEFAULT_CALENDAR_WINDOWS = ("1D", "3D", "7D", "14D", "30D")
BINANCE = Venue("BINANCE")
EXIT_MULTIPLIER_KEYS = (
    "long_profit_taking_mult",
    "long_stop_loss_mult",
    "short_profit_taking_mult",
    "short_stop_loss_mult",
)


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
    parser.add_argument("--trade-notional", type=Decimal, default=None)
    parser.add_argument("--starting-balance", type=Decimal, default=None)
    parser.add_argument("--default-leverage", type=Decimal, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--min-position-change", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--no-backtest", action="store_true")
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


def apply_json_defaults(args: argparse.Namespace, strategy_config: dict[str, Any]) -> argparse.Namespace:
    workflow = section(strategy_config, "real_data_workflow")
    labeling = section(strategy_config, "real_data_labeling")
    features = section(strategy_config, "feature_engineering")
    meta_config = section(strategy_config, "meta_model")
    backtest = section(strategy_config, "backtest")

    args.output_dir = first_not_none(args.output_dir, workflow.get("output_dir"), str(DEFAULT_OUTPUT_DIR))
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
            80,
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
        first_not_none(args.progress_interval, meta_config.get("progress_interval"), 1),
    )
    args.embargo_bars = int(first_not_none(args.embargo_bars, meta_config.get("embargo_bars"), 80))
    args.pca_components = first_not_none(args.pca_components, meta_config.get("pca_components"))
    args.pca_random_state = int(first_not_none(args.pca_random_state, meta_config.get("pca_random_state"), 42))
    args.bet_size_step = float(first_not_none(args.bet_size_step, meta_config.get("bet_size_step"), 0.1))

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


def signal_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "signal" not in frame:
        return {}
    return class_counts(frame["signal"])


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
) -> pd.Series:
    proba = model.predict_proba(features)
    if 1 not in proba.columns:
        return pd.Series(0, index=features.index, name="prediction")
    return (proba[1] >= threshold).astype(int).rename("prediction")


def primary_side_frame_from_rule(
    model: CvdSlopeRuleSideModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    side = model.primary_side(features).astype(int)
    return pd.DataFrame(
        {
            "primary_side": side,
            "primary_prediction": side,
        },
        index=features.index,
    )


def make_rule_meta_features(features: pd.DataFrame, primary_frame: pd.DataFrame) -> pd.DataFrame:
    return features.loc[features.index.intersection(primary_frame.index)].copy()


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
) -> Any:
    if getattr(model, "requires_event_spans", False):
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


def write_importance_csv(frame: pd.DataFrame, path: Path) -> str | None:
    if frame.empty:
        return None
    frame.to_csv(path, index=False)
    return str(path)


def fit_rule_meta_model_with_feature_selection(
    *,
    meta_dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    symbol: str,
    meta_probability_threshold: float,
) -> MetaTrainingResult:
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "method": "original_feature_space",
    }
    final_dataset = meta_dataset
    final_feature_columns = list(meta_dataset.X.columns)

    if pca_feature_selection_enabled(strategy_config, args):
        pca_components = args.pca_components
        if pca_components is None:
            pca_components = section(strategy_config, "meta_model").get("pca_components", 0.95)
        pca_components = float(pca_components)
        selectable_columns = list(meta_dataset.X.columns)
        if not selectable_columns:
            raise ValueError("PCA feature selection requires at least one feature")
        selectable_dataset = replace_dataset_features(meta_dataset, meta_dataset.features[selectable_columns])
        pca_dataset, pca_transformer = _pca_dataset(
            selectable_dataset,
            train_index,
            pca_components=pca_components,
            pca_random_state=args.pca_random_state,
        )
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
        pca_model = fit_meta_model(pca_model, pca_dataset, train_index)
        pca_importance = pca_mdi_feature_importance(
            pca_model,
            pca_transformer,
            selectable_dataset.X.columns,
        )
        top_n = selected_feature_count(
            available=len(selectable_columns),
            strategy_config=strategy_config,
        )
        selected_raw = pca_importance.head(top_n)["feature"].astype(str).tolist()
        final_feature_columns = selected_raw
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
            "raw_feature_count": len(selectable_columns),
            "pca_component_count": int(pca_dataset.X.shape[1]),
            "selected_raw_feature_count": len(selected_raw),
            "primary_context_feature_count": 0,
            "selected_feature_count": len(final_feature_columns),
            "selected_features": final_feature_columns,
            "pca_importance_csv": write_importance_csv(pca_importance, pca_importance_path),
            "pca_train_metrics": pca_train_metrics,
            "pca_test_metrics": pca_test_metrics,
        }

        if mda_pruning_enabled(strategy_config):
            config = feature_selection_config(strategy_config)
            pruning_validation_fraction = float(config.get("mda_pruning_validation_fraction", 0.25))
            pruning_train_index, pruning_validation_index = split_train_validation_indices_for_pruning(
                final_dataset,
                train_index,
                validation_fraction=pruning_validation_fraction,
                embargo_bars=int(args.embargo_bars),
            )
            provisional_model = model_from_config(
                role="meta-mda-prune",
                strategy_config=strategy_config,
                args=args,
                seed_offset=10_000,
            )
            provisional_model = fit_meta_model(provisional_model, final_dataset, pruning_train_index)
            pruning_mda = mda_feature_importance(
                provisional_model,
                final_dataset.X.loc[pruning_validation_index],
                final_dataset.y.loc[pruning_validation_index],
                sample_weight=final_dataset.sample_weight.loc[pruning_validation_index],
                n_repeats=int(config.get("mda_repeats", 3)),
                random_state=int(args.pca_random_state),
            )
            retained_features = pruning_mda.loc[pruning_mda["importance"] >= 0.0, "feature"].astype(str).tolist()
            dropped_features = [
                feature
                for feature in final_feature_columns
                if feature not in set(retained_features)
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
            diagnostics["mda_pruning_validation_fraction"] = pruning_validation_fraction
            diagnostics["mda_pruning_train_events"] = len(pruning_train_index)
            diagnostics["mda_pruning_validation_events"] = len(pruning_validation_index)
            diagnostics["selected_feature_count"] = len(final_feature_columns)
            diagnostics["selected_features"] = final_feature_columns
        else:
            diagnostics["mda_pruning_enabled"] = False

    meta_model = model_from_config(
        role="meta",
        strategy_config=strategy_config,
        args=args,
        seed_offset=10_000,
    )
    meta_model = fit_meta_model(meta_model, final_dataset, train_index)
    train_pred = threshold_binary_prediction(
        meta_model,
        final_dataset.X.loc[train_index],
        threshold=meta_probability_threshold,
    )
    test_pred = threshold_binary_prediction(
        meta_model,
        final_dataset.X.loc[test_index],
        threshold=meta_probability_threshold,
    )

    if diagnostics.get("enabled"):
        config = feature_selection_config(strategy_config)
        importance_seed = int(args.pca_random_state)
        mdi = mdi_feature_importance(meta_model)
        mda = mda_feature_importance(
            meta_model,
            final_dataset.X.loc[test_index],
            final_dataset.y.loc[test_index],
            sample_weight=final_dataset.sample_weight.loc[test_index],
            n_repeats=int(config.get("mda_repeats", 3)),
            random_state=importance_seed,
        )
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
        )
        diagnostics["final_mdi_csv"] = write_importance_csv(
            mdi,
            output_dir / f"{symbol}_meta_selected_mdi_importance.csv",
        )
        diagnostics["final_mda_csv"] = write_importance_csv(
            mda,
            output_dir / f"{symbol}_meta_selected_mda_importance.csv",
        )
        diagnostics["final_sfi_csv"] = write_importance_csv(
            sfi,
            output_dir / f"{symbol}_meta_selected_sfi_importance.csv",
        )
        diagnostics["final_top_mdi_features"] = mdi.head(10).to_dict(orient="records")
        diagnostics["final_top_mda_features"] = mda.head(10).to_dict(orient="records")
        diagnostics["final_top_sfi_features"] = sfi.head(10).to_dict(orient="records")

    return MetaTrainingResult(
        model=meta_model,
        train_pred=train_pred,
        test_pred=test_pred,
        feature_columns=final_feature_columns,
        diagnostics=diagnostics,
    )


def make_rule_meta_signal_frame(
    primary_frame: pd.DataFrame,
    meta_model: Any,
    meta_features: pd.DataFrame,
    *,
    meta_probability_threshold: float,
    active_index: pd.DatetimeIndex,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "signal": 0,
            "prediction": primary_frame["primary_prediction"],
            "confidence": 0.0,
            "primary_side": primary_frame["primary_side"],
            "meta_probability": 0.0,
            "meta_prediction": 0,
        },
        index=primary_frame.index,
    )

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
            meta_prediction = (trade_probability >= meta_probability_threshold).astype(int)
            frame.loc[candidate_features.index, "meta_probability"] = trade_probability
            frame.loc[candidate_features.index, "meta_prediction"] = meta_prediction
            frame.loc[candidate_features.index, "confidence"] = trade_probability
            frame.loc[candidate_features.index, "signal"] = (
                primary_frame.loc[candidate_features.index, "primary_side"] * meta_prediction
            ).astype(int)

    inactive = ~frame.index.isin(active_index)
    frame.loc[inactive, "signal"] = 0
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
) -> SequentialBootstrapBaggingClassifier:
    meta_config = section(strategy_config, "meta_model")
    bagging_config = section(strategy_config, "sequential_bagging")
    base_seed = int(bagging_config.get("random_seed", 2027))
    n_estimators = int(n_estimators_override or args.n_estimators or meta_config.get("n_estimators", 100))
    max_features = parse_model_scalar(args.max_features or meta_config.get("max_features", 1))
    max_samples = resolve_max_samples(args.max_samples, meta_config.get("max_samples"))
    min_samples_leaf = int(args.min_samples_leaf or meta_config.get("min_samples_leaf", 1))
    max_depth = args.max_depth if args.max_depth is not None else meta_config.get("max_depth")
    class_weight = resolve_class_weight(args.class_weight, meta_config.get("class_weight"))

    if announce:
        print(
            f"  {role} model: estimators={n_estimators} max_features={max_features} "
            f"max_samples={max_samples} min_samples_leaf={min_samples_leaf} "
            f"min_weight_fraction_leaf={args.min_weight_fraction_leaf:g} "
            f"class_weight={class_weight}",
            flush=True,
        )
    return SequentialBootstrapBaggingClassifier(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
        class_weight=class_weight,
        random_state=base_seed + seed_offset,
        n_jobs=args.n_jobs,
        verbose=not args.no_progress,
        progress_interval=args.progress_interval,
        progress_label=role,
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


def validate_runtime_candidate_exit_definition(candidate: dict[str, Any] | None, *, symbol: str) -> None:
    if candidate is None:
        return
    exit_definition = candidate.get("exit_definition")
    if not isinstance(exit_definition, dict):
        raise ValueError(
            f"Synthetic runtime candidate for {symbol} has no directional exit_definition",
        )
    missing = [key for key in EXIT_MULTIPLIER_KEYS if key not in exit_definition]
    if missing:
        raise ValueError(
            f"Synthetic runtime candidate for {symbol} is missing exit_definition keys: {missing}",
        )


def _resolve_exit_multiplier(
    *,
    runtime_config: dict[str, Any],
    synthetic_candidate: dict[str, Any] | None,
    key: str,
    default: float,
) -> float:
    for config in (runtime_config, synthetic_candidate):
        value = _exit_config_value(config, key)
        if value is not None:
            return float(value)
    return float(default)


def directional_barrier_settings(
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    *,
    symbol: str,
) -> tuple[dict[str, float], int]:
    barrier_config = section(strategy_config, "barrier_optimization")
    runtime_config = section(strategy_config, "runtime_strategy")
    synthetic_candidate = synthetic_runtime_strategy_candidate(strategy_config, symbol)
    validate_runtime_candidate_exit_definition(synthetic_candidate, symbol=symbol)
    long_pt = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        synthetic_candidate=synthetic_candidate,
        key="long_profit_taking_mult",
        default=1.0,
    )
    long_sl = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        synthetic_candidate=synthetic_candidate,
        key="long_stop_loss_mult",
        default=1.0,
    )
    short_pt = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        synthetic_candidate=synthetic_candidate,
        key="short_profit_taking_mult",
        default=1.0,
    )
    short_sl = _resolve_exit_multiplier(
        runtime_config=runtime_config,
        synthetic_candidate=synthetic_candidate,
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
        or (synthetic_candidate or {}).get("vertical_barrier_bars")
        or barrier_config.get("vertical_barrier_bars", 80)
    )
    if any(value <= 0.0 for value in exit_config.values()):
        raise ValueError("pt/sl multipliers must be positive for training labels.")
    return exit_config, vertical_bars


def synthetic_runtime_strategy_candidate(
    strategy_config: dict[str, Any],
    symbol: str,
) -> dict[str, Any] | None:
    runtime_config = section(strategy_config, "runtime_strategy")
    source = str(runtime_config.get("source", "")).strip().lower()
    if source != "synthetic_ml_summary":
        return None

    summary_dir = runtime_config.get("synthetic_summary_dir") or strategy_config.get(
        "output_dir",
        "afml_scripts/output/ou_synthetic/ml_backtest",
    )
    summary_path = resolve_repo_path(summary_dir) / symbol / f"{symbol}_synthetic_ml_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Configured synthetic runtime summary does not exist for {symbol}: {summary_path}",
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = payload.get("runtime_strategy_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"Synthetic summary has no runtime_strategy_candidate: {summary_path}")
    return candidate


def validate_runtime_candidate_volatility(
    candidate: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    symbol: str,
) -> None:
    if candidate is None:
        return
    metadata = candidate.get("volatility_target")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Synthetic runtime candidate for {symbol} has no volatility_target metadata",
        )
    kind = str(metadata.get("kind", ""))
    span = metadata.get("span")
    if kind != "ewma_1bar_log_return_std":
        raise ValueError(
            f"Synthetic runtime candidate for {symbol} uses volatility_target.kind={kind!r}; "
            "expected 'ewma_1bar_log_return_std'",
        )
    if span is None:
        raise ValueError(f"Synthetic runtime candidate for {symbol} has no volatility_target.span")
    if int(span) != int(args.volatility_ewma_span):
        raise ValueError(
            f"Synthetic runtime candidate for {symbol} uses volatility EWMA span {span}, "
            f"but real workflow is configured for span {args.volatility_ewma_span}",
        )


def deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_config(out[key], value)
        else:
            out[key] = value
    return out


def runtime_event_definition(strategy_config: dict[str, Any], symbol: str) -> dict[str, Any]:
    base = section(strategy_config, "event_definition")
    synthetic_candidate = synthetic_runtime_strategy_candidate(strategy_config, symbol)
    if synthetic_candidate is None:
        return dict(base)
    candidate_event = synthetic_candidate.get("event_definition")
    if not isinstance(candidate_event, dict):
        raise TypeError("runtime_strategy_candidate.event_definition must be a JSON object")
    return deep_merge_config(base, candidate_event)


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
    synthetic_config = section(config, "synthetic_data")
    fair_value_config = section(synthetic_config, "fair_value")
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
    return features


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
    synthetic_config = section(config, "synthetic_data")
    price_model_config = section(synthetic_config, "price_model")
    price_column = args.price_column or str(price_model_config.get("price_column", "close"))
    meta_config = section(strategy_config, "meta_model")
    embargo_bars = int(args.embargo_bars or meta_config.get("embargo_bars", 80))
    synthetic_candidate = synthetic_runtime_strategy_candidate(strategy_config, symbol)
    validate_runtime_candidate_volatility(synthetic_candidate, args=args, symbol=symbol)
    event_config = runtime_event_definition(strategy_config, symbol)
    runtime_strategy_config = deep_merge_config(strategy_config, {"event_definition": event_config})
    exit_config, vertical_bars = directional_barrier_settings(strategy_config, args, symbol=symbol)

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
    volatility = make_ewma_1bar_log_return_volatility_target(
        close,
        span=args.volatility_ewma_span,
        train_start=window.train_start,
        train_end=window.train_end,
        floor_quantile=args.volatility_floor_quantile,
    )

    print(f"{symbol}: building AFML/CVD feature matrix", flush=True)
    features = build_feature_matrix(
        frame=frame,
        config=config,
        strategy_config=runtime_strategy_config,
        price_column=price_column,
        volatility=volatility,
        args=args,
    )
    print(f"{symbol}: features={features.shape[1]:,}", flush=True)

    labeling_config = section(strategy_config, "synthetic_labeling")
    meta_label_min_ret = (
        float(args.meta_label_min_ret)
        if args.meta_label_min_ret is not None
        else round_trip_cost_from_labeling_config(labeling_config)
    )
    meta_probability_threshold = (
        float(args.meta_probability_threshold)
        if args.meta_probability_threshold is not None
        else float(meta_config.get("class_probability_threshold", 0.5))
    )

    print(f"{symbol}: configuring deterministic CVD-slope primary rule", flush=True)
    primary_model = CvdSlopeRuleSideModel(event_config)
    primary_frame = primary_side_frame_from_rule(
        primary_model,
        features,
    )
    primary_side = primary_frame["primary_side"]
    candidate_side = primary_side[primary_side != 0]
    print(
        f"{symbol}: primary rule candidates={len(candidate_side):,} "
        f"sides={class_counts(candidate_side) if not candidate_side.empty else {}}",
        flush=True,
    )
    meta_features = make_rule_meta_features(features, primary_frame)
    print(
        f"{symbol}: labeling rule-primary candidates for meta model "
        f"long_pt={exit_config['long_profit_taking_mult']:g} "
        f"long_sl={exit_config['long_stop_loss_mult']:g} "
        f"short_pt={exit_config['short_profit_taking_mult']:g} "
        f"short_sl={exit_config['short_stop_loss_mult']:g} "
        f"vertical_bars={vertical_bars}",
        flush=True,
    )
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

    raw_signals = make_rule_meta_signal_frame(
        primary_frame,
        meta_training.model,
        meta_features,
        meta_probability_threshold=meta_probability_threshold,
        active_index=test_index,
        feature_columns=meta_training.feature_columns,
    )
    signals = make_bet_size_frame(
        raw_signals,
        probability_col="meta_probability",
        step_size=args.bet_size_step,
    )
    test_signals = signals.loc[signals.index.intersection(test_index)].copy()
    test_signals.insert(0, "symbol", symbol)
    test_signals["trgt"] = volatility.reindex(test_signals.index).ffill().bfill()
    test_signals = apply_directional_exit_columns(test_signals, exit_config)
    signal_path = output_dir / f"{symbol}_signals_test.csv"
    write_signal_csv(test_signals, signal_path)

    print(
        f"{symbol}: done meta_test_precision={meta_test_metrics['positive_precision']:.4f} "
        f"meta_test_recall={meta_test_metrics['positive_recall']:.4f} "
        f"meta_test_f1={meta_test_metrics['positive_f1']:.4f} "
        f"test_signals={signal_counts(test_signals)}",
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
        "volatility_target": {
            "kind": "ewma_1bar_log_return_std",
            "span": args.volatility_ewma_span,
            "floor_quantile": args.volatility_floor_quantile,
            "return_type": "log",
        },
        "meta_label_min_ret": meta_label_min_ret,
        "primary": {
            "model": "cvdslope_rule",
            "candidate_count": len(candidate_side),
            "side_counts": class_counts(candidate_side) if not candidate_side.empty else {},
            "event_definition": event_config,
        },
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
            "probability_threshold": meta_probability_threshold,
            "feature_selection": meta_training.diagnostics,
        },
        "signals": {
            "test_rows": len(test_signals),
            "signal_counts": signal_counts(test_signals),
            "active_signals": int((test_signals["signal"].astype(int) != 0).sum()),
        },
        "model_controls": {
            "primary_model": "cvdslope_rule",
            "meta_model": "sequential_bootstrap_random_forest",
            "max_features": parse_model_scalar(args.max_features or meta_config.get("max_features", 1)),
            "max_samples": resolve_max_samples(args.max_samples, meta_config.get("max_samples")),
            "min_weight_fraction_leaf": args.min_weight_fraction_leaf,
            "class_weight": resolve_class_weight(args.class_weight, meta_config.get("class_weight")),
            "sequential_bootstrap": {"primary": False, "meta": True},
            "sample_weight": "return_attribution_x_time_decay_or_uniqueness",
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
    all_bars: list[Bar] = []
    symbol_rows: list[dict[str, Any]] = []
    for summary in summaries:
        symbol = str(summary["symbol"])
        input_csv = resolve_repo_path(summary["input_csv"])
        signal_csv = resolve_repo_path(summary["signal_csv"])
        window = summary["window"]
        raw_frame = read_real_bars(input_csv, price_column=str(summary["price_column"]))
        frame = frame_with_datetime_index(raw_frame)
        frame = filter_frame_by_time(frame, start=window["test_start"], end=window["test_end"])
        instrument = make_instrument(symbol, input_csv)
        bar_type = make_bar_type(instrument)
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
            profit_taking_mult=None,
            stop_loss_mult=None,
            vertical_barrier_bars=int(summary["vertical_barrier_bars"]),
            vertical_barrier_days=None,
            order_time_in_force=TimeInForce.GTC,
            min_position_change=min_position_change,
            close_positions_on_stop=True,
            reduce_only_on_stop=True,
        )
        engine.add_instrument(instrument)
        engine.add_strategy(CvdSlopeAfmlSignalStrategy(config=strategy_config_obj))
        bars = bars_from_frame(frame, instrument, bar_type)
        all_bars.extend(bars)
        symbol_rows.append(
            {
                "symbol": symbol,
                "bars": len(bars),
                "trade_size": str(trade_size),
                "min_position_change": str(min_position_change),
                "signal_csv": signal_csv,
                "instrument_id": str(instrument.id),
            },
        )

    all_bars.sort(key=lambda bar: int(bar.ts_event))
    if not all_bars:
        raise ValueError("No bars available for backtest.")

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
    engine.dispose()
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

    backtest_summary = None
    if args.run_backtest and summaries:
        backtest_summary = run_real_backtest(
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
