from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm


try:
    from numba import njit
    from numba import prange
except ImportError:
    njit = None
    prange = range


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_scripts.diagnose_ou_residuals import kalman_local_level  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import (  # noqa: E402
    DEFAULT_SYNTHETIC_STRATEGY_CONFIG,
)
from afml_scripts.generate_ou_residual_synthetic import barrier_unit  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import configured_symbols  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import deterministic_symbol_seed  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import fit_joint_trend_ou_model  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import input_csv_for_symbol  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import load_strategy_config  # noqa: E402
from afml_scripts.generate_ou_residual_synthetic import read_real_bars  # noqa: E402
from afml_strategies.config_loader import load_afml_data_config  # noqa: E402
from afml_strategies.config_loader import resolve_repo_path  # noqa: E402
from afml_strategies.config_loader import section  # noqa: E402
from nautilus_trader.research.afml_pipeline import AfmlDataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import _pca_dataset  # noqa: E402
from nautilus_trader.research.afml_pipeline import make_pipeline_features  # noqa: E402
from nautilus_trader.research.afml_pipeline import mda_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import mdi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import pca_mdi_feature_importance  # noqa: E402
from nautilus_trader.research.afml_pipeline import sfi_feature_importance  # noqa: E402


FEATURE_COLUMNS = [
    "side",
    "trend_r2",
    "micro_slope",
    "cvd_z",
    "residual_z",
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "realized_vol_20",
    "signed_notional_z",
]
PRIMARY_STATE_COLUMNS = tuple(FEATURE_COLUMNS[1:])
DEFAULT_CALENDAR_WINDOWS = ("1D", "3D", "7D", "14D", "30D")

LOGGER = logging.getLogger("afml_synthetic_barriers_ml")
CACHE_SCHEMA_VERSION = 4
VOLATILITY_TARGET_KIND = "ewma_1bar_log_return_std"
NEGATIVE_OBJECTIVE = -1.0e12


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass(frozen=True)
class SyntheticCandidateOutcome:
    expected_payoff: np.ndarray
    payoff_std: np.ndarray
    payoff_standard_error: np.ndarray
    pt_hit_rate: np.ndarray
    sl_hit_rate: np.ndarray
    vertical_rate: np.ndarray
    mean_exit_step: np.ndarray
    unit: float
    n_paths: int


@dataclass(frozen=True)
class SyntheticMetaFoldResult:
    model: Any
    dataset: AfmlDataset
    probabilities: pd.Series
    feature_columns: list[str]
    diagnostics: dict[str, Any]
    pca_importance: pd.DataFrame | None = None
    mdi_importance: pd.DataFrame | None = None
    mda_importance: pd.DataFrame | None = None
    sfi_importance: pd.DataFrame | None = None


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def format_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "nan"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{result:.{precision}g}" if math.isfinite(result) else "nan"


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def run_fingerprint(
    *,
    symbol: str,
    input_csv: Path,
    price_model_config: dict[str, Any],
    fair_value_config: dict[str, Any],
    trend_config: dict[str, Any],
    ou_config: dict[str, Any],
    barrier_config: dict[str, Any],
    event_config: dict[str, Any],
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    labeling_config: dict[str, Any],
    primary_optuna_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "symbol": symbol,
        "input_csv": file_signature(input_csv),
        "primary_state_columns": PRIMARY_STATE_COLUMNS,
        "feature_source": "make_pipeline_features_plus_cvdslope_state",
        "feature_engineering": feature_config,
        "price_model": price_model_config,
        "fair_value": fair_value_config,
        "trend_process": trend_config,
        "ou_process": ou_config,
        "barrier_optimization": barrier_config,
        "event_definition": event_config,
        "primary_optuna": primary_optuna_config,
        "meta_model": meta_config,
        "sequential_bagging": bagging_config,
        "synthetic_labeling": labeling_config,
    }
    raw = json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cached_summary_paths_exist(summary: dict[str, Any]) -> bool:
    for key in ("events_csv", "screening_csv", "candidates_csv"):
        value = summary.get(key)
        if not value:
            return False
        path = Path(str(value))
        if not path.exists():
            path = resolve_repo_path(path)
        if not path.exists():
            return False
    return True


def load_cached_summary(summary_path: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(summary, dict):
        return None
    if summary.get("run_fingerprint") != expected_fingerprint:
        return None
    if not cached_summary_paths_exist(summary):
        return None
    summary["summary_path"] = str(summary_path)
    summary["cache_hit"] = True
    return summary


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    return numerator / denominator


def max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def current_streak(mask: np.ndarray) -> int:
    streak = 0
    for value in mask.astype(bool)[::-1]:
        if not value:
            break
        streak += 1
    return streak


def distribution_shape(values: np.ndarray) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    centered = values - float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std <= 0.0:
        return None, None
    z = centered / std
    skew = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    return skew, excess_kurtosis


def trade_path_metrics(
    returns: np.ndarray,
    *,
    event_starts: np.ndarray | None = None,
) -> dict[str, Any]:
    returns = np.asarray(returns, dtype=float)
    if event_starts is not None:
        order = np.argsort(np.asarray(event_starts), kind="stable")
        returns = returns[order]

    n = len(returns)
    wins = returns > 0.0
    losses = returns < 0.0
    breakeven = returns == 0.0
    win_values = returns[wins]
    loss_values = returns[losses]
    q05, q25, q75, q95 = np.quantile(returns, [0.05, 0.25, 0.75, 0.95])

    cumulative_log = np.concatenate(([0.0], np.cumsum(returns)))
    running_peak_log = np.maximum.accumulate(cumulative_log)
    drawdown_log = cumulative_log - running_peak_log
    max_drawdown_log = float(np.min(drawdown_log))
    max_drawdown_log_abs = abs(max_drawdown_log)

    equity = np.exp(cumulative_log)
    running_peak_equity = np.maximum.accumulate(equity)
    drawdown_pct = equity / running_peak_equity - 1.0
    max_drawdown_pct = float(np.min(drawdown_pct))
    ulcer_index = float(np.sqrt(np.mean(np.square(drawdown_pct[drawdown_pct < 0.0])))) if np.any(drawdown_pct < 0.0) else 0.0

    total_win = float(np.sum(win_values)) if len(win_values) else 0.0
    total_loss = float(np.sum(loss_values)) if len(loss_values) else 0.0
    mean_win = float(np.mean(win_values)) if len(win_values) else None
    mean_loss = float(np.mean(loss_values)) if len(loss_values) else None
    median_win = float(np.median(win_values)) if len(win_values) else None
    median_loss = float(np.median(loss_values)) if len(loss_values) else None
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    skew, excess_kurtosis = distribution_shape(returns)
    tail_loss = returns[returns <= q05]

    return {
        "trade_count": int(n),
        "win_count": int(np.sum(wins)),
        "loss_count": int(np.sum(losses)),
        "breakeven_count": int(np.sum(breakeven)),
        "win_rate": float(np.mean(wins)),
        "loss_rate": float(np.mean(losses)),
        "breakeven_rate": float(np.mean(breakeven)),
        "mean_trade_log_return": mean_return,
        "median_trade_log_return": float(np.median(returns)),
        "std_trade_log_return": std_return,
        "min_trade_log_return": float(np.min(returns)),
        "max_trade_log_return": float(np.max(returns)),
        "q05_trade_log_return": float(q05),
        "q25_trade_log_return": float(q25),
        "q75_trade_log_return": float(q75),
        "q95_trade_log_return": float(q95),
        "avg_win_log_return": mean_win,
        "avg_loss_log_return": mean_loss,
        "median_win_log_return": median_win,
        "median_loss_log_return": median_loss,
        "largest_win_log_return": float(np.max(win_values)) if len(win_values) else None,
        "largest_loss_log_return": float(np.min(loss_values)) if len(loss_values) else None,
        "total_win_log_return": total_win,
        "total_loss_log_return": total_loss,
        "profit_factor": safe_ratio(total_win, abs(total_loss)),
        "payoff_ratio": safe_ratio(mean_win or 0.0, abs(mean_loss or 0.0)),
        "downside_std_log_return": downside_std,
        "sortino_like": safe_ratio(mean_return, downside_std),
        "tail_ratio_95_05": safe_ratio(float(q95), abs(float(q05))),
        "expected_shortfall_5pct_log_return": float(np.mean(tail_loss)) if len(tail_loss) else None,
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "max_consecutive_wins": max_consecutive(wins),
        "max_consecutive_losses": max_consecutive(losses),
        "max_consecutive_breakeven": max_consecutive(breakeven),
        "ending_win_streak": current_streak(wins),
        "ending_loss_streak": current_streak(losses),
        "total_expected_log_return": float(cumulative_log[-1]),
        "total_expected_return": float(np.exp(cumulative_log[-1]) - 1.0),
        "max_drawdown_log_return": max_drawdown_log,
        "max_drawdown_log_return_abs": max_drawdown_log_abs,
        "max_drawdown_pct": max_drawdown_pct,
        "ulcer_index_pct": ulcer_index,
        "recovery_factor_log": safe_ratio(float(cumulative_log[-1]), max_drawdown_log_abs),
        "calmar_like": safe_ratio(mean_return, max_drawdown_log_abs),
    }


def rolling_slope_r2(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    slope = np.full(len(values), np.nan, dtype=float)
    r2 = np.full(len(values), np.nan, dtype=float)
    x = np.arange(window, dtype=float)
    x_centered = x - float(np.mean(x))
    x_var = float(np.sum(x_centered * x_centered))
    for idx in range(window - 1, len(values)):
        y = values[idx - window + 1 : idx + 1]
        if not np.all(np.isfinite(y)):
            continue
        y_centered = y - float(np.mean(y))
        y_var = float(np.sum(y_centered * y_centered))
        if y_var <= 0.0:
            slope[idx] = 0.0
            r2[idx] = 0.0
            continue
        covariance = float(np.sum(x_centered * y_centered))
        slope[idx] = covariance / x_var
        r2[idx] = (covariance * covariance) / (x_var * y_var)
    return slope, r2


def rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=1)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.to_numpy(dtype=float)


def signed_notional(df: pd.DataFrame) -> np.ndarray:
    if "signed_notional" in df.columns:
        return pd.to_numeric(df["signed_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if {"buy_notional", "sell_notional"}.issubset(df.columns):
        buy = pd.to_numeric(df["buy_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sell = pd.to_numeric(df["sell_notional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return buy - sell
    if {"buy_ticks", "sell_ticks"}.issubset(df.columns):
        buy = pd.to_numeric(df["buy_ticks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sell = pd.to_numeric(df["sell_ticks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return buy - sell
    raise ValueError("CVD requires signed_notional, buy/sell_notional, or buy/sell_ticks columns.")


def event_state_features(
    df: pd.DataFrame,
    price_column: str,
    fair_value_config: dict[str, Any],
    event_config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    close = pd.to_numeric(df[price_column], errors="coerce").to_numpy(dtype=float)
    log_close = np.log(close)
    trend_window = int(event_config.get("trend_window_bars", 20))
    micro_window = int(event_config.get("micro_slope_window_bars", 5))
    trend_input = pd.DataFrame({"log_price": log_close})
    log_trend = kalman_local_level(
        trend_input,
        price_column="log_price",
        q_over_r=float(fair_value_config.get("q_over_r", 0.001)),
    ).to_numpy(dtype=float)
    residual = log_close - log_trend

    micro_slope, _ = rolling_slope_r2(log_trend, micro_window)
    _, trend_r2 = rolling_slope_r2(log_trend, trend_window)

    cvd_window = int(event_config.get("cvd_z_window_bars", 200))
    cvd_increment = signed_notional(df)
    cvd = np.cumsum(cvd_increment)
    cvd_z = rolling_zscore(cvd, cvd_window)
    signed_notional_z = rolling_zscore(cvd_increment, cvd_window)
    residual_z = rolling_zscore(residual, cvd_window)
    returns = pd.Series(log_close).diff()
    realized_vol_20 = returns.rolling(window=20, min_periods=20).std(ddof=1).to_numpy(dtype=float)

    features = pd.DataFrame(
        {
            "trend_r2": trend_r2,
            "micro_slope": micro_slope,
            "cvd_z": cvd_z,
            "residual_z": residual_z,
            "log_ret_1": pd.Series(log_close).diff(1).to_numpy(dtype=float),
            "log_ret_5": pd.Series(log_close).diff(5).to_numpy(dtype=float),
            "log_ret_20": pd.Series(log_close).diff(20).to_numpy(dtype=float),
            "realized_vol_20": realized_vol_20,
            "signed_notional_z": signed_notional_z,
        },
    )
    return features, log_trend, residual


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
        raise TypeError("feature_engineering.calendar_windows must be a string, list, tuple, or null")
    return windows or None


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


def ewma_volatility_target(
    close: pd.Series,
    *,
    span: int,
    floor_quantile: float | None,
) -> pd.Series:
    log_return = np.log(close.astype(float)).diff()
    volatility = log_return.ewm(
        span=int(span),
        adjust=False,
        min_periods=max(2, int(span) // 2),
    ).std()
    if floor_quantile is not None:
        valid = volatility.dropna()
        if not valid.empty:
            floor = float(valid.quantile(float(floor_quantile)))
            volatility = volatility.clip(lower=floor)
    return volatility.rename("trgt")


def build_synthetic_feature_frame(
    frame: pd.DataFrame,
    *,
    price_column: str,
    fair_value_config: dict[str, Any],
    event_config: dict[str, Any],
    barrier_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    close = frame[price_column].dropna().astype(float)
    volatility = ewma_volatility_target(
        close,
        span=int(barrier_config.get("volatility_ewma_span", 80)),
        floor_quantile=feature_config.get("volatility_floor_quantile"),
    )
    calendar_windows_value = feature_config.get("calendar_windows", DEFAULT_CALENDAR_WINDOWS)
    pipeline_features = make_pipeline_features(
        frame,
        volatility=volatility,
        microstructure_window=int(feature_config.get("microstructure_window", 50)),
        information_window=int(feature_config.get("information_window", 64)),
        calendar_windows=parse_calendar_windows(calendar_windows_value),
        calendar_min_periods=int(feature_config.get("calendar_min_periods", 8)),
        tail_quantile=float(feature_config.get("tail_quantile", 0.99)),
        robust_z_clip=(
            None
            if feature_config.get("robust_z_clip", 5.0) is None
            else float(feature_config.get("robust_z_clip", 5.0))
        ),
        fracdiff_d=float(feature_config.get("fracdiff_d", 0.4)),
        fracdiff_threshold=float(feature_config.get("fracdiff_threshold", 0.01)),
    )
    state_features, log_trend, log_residual = event_state_features(
        frame,
        price_column=price_column,
        fair_value_config=fair_value_config,
        event_config=event_config,
    )
    state_features.index = frame.index

    features = pd.concat([pipeline_features, state_features], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    features = features.select_dtypes(include=[np.number])
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all").ffill()
    return features, log_trend, log_residual, volatility


def direction_config(event_config: dict[str, Any], side: str) -> dict[str, Any]:
    value = event_config.get(side)
    if not isinstance(value, dict):
        raise TypeError(f"event_definition.{side} must be a JSON object")
    return value


def required_float(config: dict[str, Any], key: str, path: str) -> float:
    if key not in config:
        raise ValueError(f"{path}.{key} is required")
    return float(config[key])


def deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_config(out[key], value)
        else:
            out[key] = value
    return out


def explicit_event_definition(
    event_config: dict[str, Any],
    *,
    long_r2_threshold: float | None = None,
    short_r2_threshold: float | None = None,
    long_cvd_z_min: float | None = None,
    short_cvd_z_max: float | None = None,
) -> dict[str, Any]:
    long_config = direction_config(event_config, "long")
    short_config = direction_config(event_config, "short")
    long_event = {
        "r2_threshold": float(
            long_r2_threshold
            if long_r2_threshold is not None
            else required_float(long_config, "r2_threshold", "event_definition.long"),
        ),
        "cvd_z_min": float(
            long_cvd_z_min
            if long_cvd_z_min is not None
            else required_float(long_config, "cvd_z_min", "event_definition.long"),
        ),
        "micro_slope_min": required_float(long_config, "micro_slope_min", "event_definition.long"),
    }
    short_event = {
        "r2_threshold": float(
            short_r2_threshold
            if short_r2_threshold is not None
            else required_float(short_config, "r2_threshold", "event_definition.short"),
        ),
        "cvd_z_max": float(
            short_cvd_z_max
            if short_cvd_z_max is not None
            else required_float(short_config, "cvd_z_max", "event_definition.short"),
        ),
        "micro_slope_max": required_float(short_config, "micro_slope_max", "event_definition.short"),
    }
    out = dict(event_config)
    out["long"] = deep_merge_config(long_config, long_event)
    out["short"] = deep_merge_config(short_config, short_event)
    return out


def volatility_target_metadata(barrier_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": VOLATILITY_TARGET_KIND,
        "span": int(barrier_config.get("volatility_ewma_span", 80)),
    }


def runtime_candidate_payload(
    *,
    event_definition: dict[str, Any],
    exit_definition: dict[str, float],
    barrier_config: dict[str, Any],
    vertical_barrier_bars: int,
) -> dict[str, Any]:
    return {
        "event_definition": {
            "long": dict(event_definition.get("long", {})),
            "short": dict(event_definition.get("short", {})),
        },
        "exit_definition": {
            "long_profit_taking_mult": float(exit_definition["long_profit_taking_mult"]),
            "long_stop_loss_mult": float(exit_definition["long_stop_loss_mult"]),
            "short_profit_taking_mult": float(exit_definition["short_profit_taking_mult"]),
            "short_stop_loss_mult": float(exit_definition["short_stop_loss_mult"]),
        },
        "vertical_barrier_bars": int(vertical_barrier_bars),
        "vertical_barrier_days": None,
        "tie_policy": barrier_config.get("tie_policy", "stop_loss_first"),
        "volatility_target": volatility_target_metadata(barrier_config),
    }


def build_primary_events(  # noqa: C901
    symbol: str,
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    log_trend: np.ndarray,
    log_residual: np.ndarray,
    event_config: dict[str, Any],
    horizon: int,
) -> pd.DataFrame:
    long_config = direction_config(event_config, "long")
    short_config = direction_config(event_config, "short")
    long_r2_threshold = required_float(long_config, "r2_threshold", "event_definition.long")
    short_r2_threshold = required_float(short_config, "r2_threshold", "event_definition.short")
    long_slope_min = required_float(long_config, "micro_slope_min", "event_definition.long")
    short_slope_max = required_float(short_config, "micro_slope_max", "event_definition.short")
    long_cvd_min = required_float(long_config, "cvd_z_min", "event_definition.long")
    short_cvd_max = required_float(short_config, "cvd_z_max", "event_definition.short")
    min_gap = int(event_config.get("min_event_gap_bars", 1))

    rows: list[dict[str, Any]] = []
    last_event_idx = -min_gap - 1
    max_idx = len(feature_frame) - horizon - 1
    missing_state = [column for column in PRIMARY_STATE_COLUMNS if column not in feature_frame.columns]
    if missing_state:
        raise ValueError(f"Missing primary state feature columns for {symbol}: {missing_state}")
    feature_columns = tuple(str(column) for column in feature_frame.columns)
    bar_index = pd.DatetimeIndex(feature_frame.index)
    for idx in range(max_idx + 1):
        if idx - last_event_idx < min_gap:
            continue
        row = feature_frame.iloc[idx]
        if not np.all(np.isfinite(row.loc[list(PRIMARY_STATE_COLUMNS)].to_numpy(dtype=float))):
            continue
        if not np.all(np.isfinite(row.loc[list(feature_columns)].to_numpy(dtype=float))):
            continue
        if not math.isfinite(float(log_trend[idx])) or not math.isfinite(float(log_residual[idx])):
            continue
        side = 0
        if (
            row["trend_r2"] > long_r2_threshold
            and row["micro_slope"] > long_slope_min
            and row["cvd_z"] > long_cvd_min
        ):
            side = 1
        elif (
            row["trend_r2"] > short_r2_threshold
            and row["micro_slope"] < short_slope_max
            and row["cvd_z"] < short_cvd_max
        ):
            side = -1
        if side == 0:
            continue
        payload = {
            "symbol": symbol,
            "event_idx": idx,
            "event_end_idx": idx + horizon,
            "event_time": bar_index[idx].isoformat(),
            "event_end_time": bar_index[idx + horizon].isoformat(),
            "side": side,
            "initial_log_trend": float(log_trend[idx]),
            "initial_log_residual": float(log_residual[idx]),
        }
        for column in feature_columns:
            payload[column] = float(row[column])
        rows.append(payload)
        last_event_idx = idx

    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError(f"No primary model events found for {symbol}.")
    return events


def round_trip_cost_from_labeling_config(labeling_config: dict[str, Any]) -> float:
    required = {
        "slippage_per_side_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "fee_liquidity",
    }
    missing = sorted(required.difference(labeling_config))
    if missing:
        raise ValueError(f"synthetic_labeling missing explicit cost fields: {missing}")
    slippage = float(labeling_config["slippage_per_side_rate"])
    liquidity = str(labeling_config["fee_liquidity"])
    if liquidity not in {"maker", "taker"}:
        raise ValueError("synthetic_labeling.fee_liquidity must be 'maker' or 'taker'")
    fee = float(labeling_config[f"{liquidity}_fee_rate"])
    return 2.0 * (slippage + fee)


def first_hit(mask: np.ndarray, horizon: int) -> np.ndarray:
    has_hit = mask.any(axis=1)
    first = np.full(mask.shape[0], horizon + 1, dtype=np.int32)
    first[has_hit] = np.argmax(mask[has_hit], axis=1).astype(np.int32) + 1
    return first


if njit is not None:

    @njit(cache=True, parallel=True)
    def _barrier_outcome_for_candidate_numba(  # noqa: C901
        signed_returns: np.ndarray,
        pt_levels: np.ndarray,
        sl_levels: np.ndarray,
        cost: float,
        stop_loss_first: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = signed_returns.shape[0]
        n_paths = signed_returns.shape[1]
        horizon = signed_returns.shape[2]
        expected = np.empty(batch_size, dtype=np.float32)
        payoff_std = np.empty(batch_size, dtype=np.float32)
        pt_rate = np.empty(batch_size, dtype=np.float32)
        sl_rate = np.empty(batch_size, dtype=np.float32)
        vertical_rate = np.empty(batch_size, dtype=np.float32)
        mean_exit_step = np.empty(batch_size, dtype=np.float32)

        for event_pos in prange(batch_size):
            pt = pt_levels[event_pos]
            sl = sl_levels[event_pos]
            payoff_sum = 0.0
            payoff_sq_sum = 0.0
            pt_count = 0
            sl_count = 0
            vertical_count = 0
            exit_step_sum = 0

            for path_idx in range(n_paths):
                up_first = horizon + 1
                down_first = horizon + 1
                for step_idx in range(horizon):
                    value = signed_returns[event_pos, path_idx, step_idx]
                    if up_first == horizon + 1 and value >= pt:
                        up_first = step_idx + 1
                    if down_first == horizon + 1 and value <= -sl:
                        down_first = step_idx + 1
                    if up_first != horizon + 1 and down_first != horizon + 1:
                        break

                if up_first == horizon + 1 and down_first == horizon + 1:
                    payoff = signed_returns[event_pos, path_idx, horizon - 1] - cost
                    vertical_count += 1
                    exit_step = horizon
                elif stop_loss_first and down_first <= up_first:
                    payoff = -sl - cost
                    sl_count += 1
                    exit_step = down_first
                elif (not stop_loss_first and up_first <= down_first) or up_first < down_first:
                    payoff = pt - cost
                    pt_count += 1
                    exit_step = up_first
                else:
                    payoff = -sl - cost
                    sl_count += 1
                    exit_step = down_first

                payoff_sum += payoff
                payoff_sq_sum += payoff * payoff
                exit_step_sum += exit_step

            inv_paths = 1.0 / n_paths
            mean = payoff_sum * inv_paths
            variance = payoff_sq_sum * inv_paths - mean * mean
            if variance < 0.0:
                variance = 0.0
            expected[event_pos] = mean
            payoff_std[event_pos] = math.sqrt(variance)
            pt_rate[event_pos] = pt_count * inv_paths
            sl_rate[event_pos] = sl_count * inv_paths
            vertical_rate[event_pos] = vertical_count * inv_paths
            mean_exit_step[event_pos] = exit_step_sum * inv_paths

        return expected, payoff_std, pt_rate, sl_rate, vertical_rate, mean_exit_step

else:
    _barrier_outcome_for_candidate_numba = None


def simulate_event_signed_returns(
    model: Any,
    initial_trend: np.ndarray,
    initial_residual: np.ndarray,
    side: np.ndarray,
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size = len(initial_trend)
    trend = np.repeat(initial_trend[:, None], n_paths, axis=1)
    residual = np.repeat(initial_residual[:, None], n_paths, axis=1)
    log_price_0 = trend + residual
    signed_returns = np.empty((batch_size, n_paths, horizon), dtype=np.float32)
    mean = np.array([model.trend_drift, 0.0], dtype=float)
    for step in range(horizon):
        innovations = rng.multivariate_normal(
            mean,
            model.innovation_covariance,
            size=(batch_size, n_paths),
            check_valid="raise",
        )
        trend = trend + innovations[:, :, 0]
        residual = (
            model.ou_mean
            + model.ou_phi * (residual - model.ou_mean)
            + innovations[:, :, 1]
        )
        signed_returns[:, :, step] = (side[:, None] * ((trend + residual) - log_price_0)).astype(
            np.float32,
        )
    return signed_returns


def candidate_exit_levels(
    events: pd.DataFrame,
    exit_definition: dict[str, float],
    unit: float,
) -> tuple[np.ndarray, np.ndarray]:
    side = events["side"].to_numpy(dtype=float)
    long_pt = float(exit_definition["long_profit_taking_mult"])
    long_sl = float(exit_definition["long_stop_loss_mult"])
    short_pt = float(exit_definition["short_profit_taking_mult"])
    short_sl = float(exit_definition["short_stop_loss_mult"])
    if min(long_pt, long_sl, short_pt, short_sl) <= 0.0:
        raise ValueError("directional PT/SL multipliers must be positive")
    pt = np.where(side >= 0.0, long_pt, short_pt).astype(np.float32) * float(unit)
    sl = np.where(side >= 0.0, long_sl, short_sl).astype(np.float32) * float(unit)
    return pt.astype(np.float32, copy=False), sl.astype(np.float32, copy=False)


def barrier_outcome_for_candidate(
    signed_returns: np.ndarray,
    *,
    pt_levels: np.ndarray,
    sl_levels: np.ndarray,
    horizon: int,
    cost: float,
    tie_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if signed_returns.ndim != 3:
        raise ValueError("signed_returns must have shape (events, paths, horizon)")
    if signed_returns.shape[2] != horizon:
        raise ValueError("horizon must match signed_returns.shape[2]")
    if signed_returns.shape[0] != len(pt_levels) or signed_returns.shape[0] != len(sl_levels):
        raise ValueError("PT/SL level arrays must align to signed_returns events")
    if signed_returns.shape[1] < 1:
        raise ValueError("signed_returns must contain at least one path")
    if tie_policy not in {"stop_loss_first", "profit_taking_first"}:
        raise ValueError(f"Unsupported tie_policy: {tie_policy}")
    if _barrier_outcome_for_candidate_numba is None:
        raise ImportError("Numba is required for accelerated directional OTR evaluation")

    stop_loss_first = tie_policy == "stop_loss_first"
    return _barrier_outcome_for_candidate_numba(
        signed_returns.astype(np.float32, copy=False),
        pt_levels.astype(np.float32, copy=False),
        sl_levels.astype(np.float32, copy=False),
        float(cost),
        stop_loss_first,
    )


def synthetic_outcome_for_runtime_candidate(
    events: pd.DataFrame,
    model: Any,
    barrier_config: dict[str, Any],
    labeling_config: dict[str, Any],
    symbol_seed: int,
    *,
    exit_definition: dict[str, float],
    symbol: str | None = None,
    show_progress: bool = True,
) -> SyntheticCandidateOutcome:
    horizon = int(barrier_config.get("vertical_barrier_bars", 80))
    n_paths = int(labeling_config.get("n_paths_per_event", 25_000))
    batch_size = int(labeling_config.get("event_batch_size", 16))
    cost = round_trip_cost_from_labeling_config(labeling_config)
    tie_policy = str(barrier_config.get("tie_policy", "stop_loss_first"))
    unit = barrier_unit(model, barrier_config)
    n_events = len(events)
    expected = np.empty(n_events, dtype=np.float32)
    payoff_std = np.empty(n_events, dtype=np.float32)
    pt_rate = np.empty(n_events, dtype=np.float32)
    sl_rate = np.empty(n_events, dtype=np.float32)
    vertical_rate = np.empty(n_events, dtype=np.float32)
    mean_exit_step = np.empty(n_events, dtype=np.float32)
    rng = np.random.default_rng(symbol_seed)

    LOGGER.info(
        "%s synthetic candidate OTR: events=%s paths/event=%s horizon=%s "
        "long_pt=%.4g long_sl=%.4g short_pt=%.4g short_sl=%.4g cost=%.6g",
        symbol or "symbol",
        f"{n_events:,}",
        f"{n_paths:,}",
        horizon,
        float(exit_definition["long_profit_taking_mult"]),
        float(exit_definition["long_stop_loss_mult"]),
        float(exit_definition["short_profit_taking_mult"]),
        float(exit_definition["short_stop_loss_mult"]),
        cost,
    )
    for start in tqdm(
        range(0, n_events, batch_size),
        desc=f"{symbol or 'symbol'} synthetic candidate",
        disable=not show_progress,
        dynamic_ncols=True,
        leave=False,
        unit="batch",
    ):
        end = min(start + batch_size, n_events)
        batch = events.iloc[start:end]
        signed_returns = simulate_event_signed_returns(
            model,
            initial_trend=batch["initial_log_trend"].to_numpy(dtype=float),
            initial_residual=batch["initial_log_residual"].to_numpy(dtype=float),
            side=batch["side"].to_numpy(dtype=float),
            n_paths=n_paths,
            horizon=horizon,
            rng=rng,
        )
        pt_levels, sl_levels = candidate_exit_levels(batch, exit_definition, unit)
        batch_result = barrier_outcome_for_candidate(
            signed_returns,
            pt_levels=pt_levels,
            sl_levels=sl_levels,
            horizon=horizon,
            cost=cost,
            tie_policy=tie_policy,
        )
        expected[start:end] = batch_result[0]
        payoff_std[start:end] = batch_result[1]
        pt_rate[start:end] = batch_result[2]
        sl_rate[start:end] = batch_result[3]
        vertical_rate[start:end] = batch_result[4]
        mean_exit_step[start:end] = batch_result[5]

    return SyntheticCandidateOutcome(
        expected_payoff=expected,
        payoff_std=payoff_std,
        payoff_standard_error=payoff_std / math.sqrt(float(n_paths)),
        pt_hit_rate=pt_rate,
        sl_hit_rate=sl_rate,
        vertical_rate=vertical_rate,
        mean_exit_step=mean_exit_step,
        unit=unit,
        n_paths=n_paths,
    )


def event_uniqueness(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    max_end = int(np.max(ends)) + 2
    diff = np.zeros(max_end + 1, dtype=float)
    for start, end in zip(starts, ends, strict=True):
        diff[int(start)] += 1.0
        diff[int(end) + 1] -= 1.0
    concurrency = np.cumsum(diff)
    uniqueness = np.empty(len(starts), dtype=float)
    for idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        uniqueness[idx] = float(np.mean(1.0 / concurrency[int(start) : int(end) + 1]))
    return uniqueness


if njit is not None:

    @njit(cache=True)
    def _sequential_bootstrap_numba(
        starts: np.ndarray,
        ends: np.ndarray,
        sample_length: int,
        seed: int,
    ) -> np.ndarray:
        np.random.seed(seed)  # noqa: NPY002
        max_end = int(np.max(ends)) + 2
        concurrency = np.zeros(max_end + 1, dtype=np.float64)
        selected = np.empty(sample_length, dtype=np.int32)
        avg_uniqueness = np.empty(len(starts), dtype=np.float64)
        for draw in range(sample_length):
            probability_sum = 0.0
            for idx in range(len(starts)):
                start = int(starts[idx])
                end = int(ends[idx])
                total = 0.0
                for time_idx in range(start, end + 1):
                    total += 1.0 / (concurrency[time_idx] + 1.0)
                avg = total / float(end - start + 1)
                avg_uniqueness[idx] = avg
                probability_sum += avg

            threshold = np.random.random() * probability_sum  # noqa: NPY002
            cumulative = 0.0
            chosen = len(starts) - 1
            for idx in range(len(starts)):
                cumulative += avg_uniqueness[idx]
                if cumulative >= threshold:
                    chosen = idx
                    break

            selected[draw] = chosen
            for time_idx in range(int(starts[chosen]), int(ends[chosen]) + 1):
                concurrency[time_idx] += 1.0
        return selected

else:
    _sequential_bootstrap_numba = None


def sequential_bootstrap(
    starts: np.ndarray,
    ends: np.ndarray,
    sample_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if _sequential_bootstrap_numba is not None:
        seed = int(rng.integers(0, 2**31 - 1))
        return _sequential_bootstrap_numba(
            starts.astype(np.int32, copy=False),
            ends.astype(np.int32, copy=False),
            int(sample_length),
            seed,
        )

    max_end = int(np.max(ends)) + 2
    concurrency = np.zeros(max_end + 1, dtype=float)
    selected = np.empty(sample_length, dtype=np.int32)
    for draw in range(sample_length):
        avg_uniqueness = np.empty(len(starts), dtype=float)
        for idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
            avg_uniqueness[idx] = float(np.mean(1.0 / (concurrency[int(start) : int(end) + 1] + 1.0)))
        probabilities = avg_uniqueness / float(np.sum(avg_uniqueness))
        chosen = int(rng.choice(len(starts), p=probabilities))
        selected[draw] = chosen
        concurrency[int(starts[chosen]) : int(ends[chosen]) + 1] += 1.0
    return selected


def bagging_sample_length(
    max_samples: Any,
    n_samples: int,
    sample_length_multiplier: float,
    *,
    sample_weight: np.ndarray | None = None,
) -> int:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    multiplier = float(sample_length_multiplier)
    if max_samples is None:
        base_length = n_samples
    elif isinstance(max_samples, str):
        value = max_samples.strip().lower()
        if value in {"avg_uniqueness", "avgu"}:
            if sample_weight is None or len(sample_weight) == 0:
                raise ValueError("sample_weight is required when max_samples='avg_uniqueness'")
            base_length = math.ceil(float(np.mean(sample_weight)) * n_samples)
        else:
            base_length = math.ceil(float(max_samples) * n_samples)
    elif isinstance(max_samples, int):
        base_length = max_samples
    else:
        base_length = math.ceil(float(max_samples) * n_samples)
    return max(1, math.ceil(base_length * multiplier))


class SequentialBaggedRandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int,
        max_features: int,
        max_samples: Any,
        min_samples_leaf: int,
        max_depth: int | None,
        random_seed: int,
        sample_length_multiplier: float,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_features = int(max_features)
        self.max_samples = max_samples
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.random_seed = int(random_seed)
        self.sample_length_multiplier = float(sample_length_multiplier)
        self.estimators: list[DecisionTreeClassifier] = []
        self.estimators_: list[DecisionTreeClassifier] = []
        self.feature_names_: list[str] | None = None
        self.n_features_in_: int | None = None
        self.classes_: np.ndarray | None = None
        self.requires_event_spans = True

    def fit(
        self,
        x: Any,
        y: Any,
        starts: np.ndarray | None = None,
        ends: np.ndarray | None = None,
        sample_weight: Any | None = None,
        precomputed_bags: Sequence[np.ndarray] | None = None,
        *,
        t1: Any | None = None,
        bar_index: Any | None = None,
    ) -> SequentialBaggedRandomForestClassifier:
        if isinstance(x, pd.DataFrame):
            self.feature_names_ = [str(column) for column in x.columns]
            x_index = pd.DatetimeIndex(x.index)
            x_values = x.to_numpy(dtype=float)
        else:
            x_index = None
            x_values = np.asarray(x, dtype=float)
            self.feature_names_ = [f"feature_{idx}" for idx in range(x_values.shape[1])]

        y_values = np.asarray(y, dtype=np.int32)
        if len(np.unique(y)) < 2:
            raise ValueError("Meta-model training requires at least two classes.")
        if sample_weight is None:
            weight_values = np.ones(len(y_values), dtype=float)
        else:
            weight_values = np.asarray(sample_weight, dtype=float)
        if starts is None or ends is None:
            if t1 is None or bar_index is None or x_index is None:
                starts = np.arange(len(y_values), dtype=np.int32)
                ends = starts.copy()
            else:
                full_bar_index = pd.DatetimeIndex(bar_index)
                start_values = full_bar_index.searchsorted(x_index, side="left").astype(np.int32)
                t1_index = pd.DatetimeIndex(pd.Series(t1, index=x_index).reindex(x_index))
                end_values = full_bar_index.searchsorted(t1_index, side="left").astype(np.int32)
                starts = start_values
                ends = np.maximum(start_values, end_values)
        else:
            starts = np.asarray(starts, dtype=np.int32)
            ends = np.asarray(ends, dtype=np.int32)

        self.n_features_in_ = int(x_values.shape[1])
        self.classes_ = np.array(sorted(np.unique(y_values)), dtype=np.int32)
        rng = np.random.default_rng(self.random_seed)
        sample_length = bagging_sample_length(
            self.max_samples,
            len(y_values),
            self.sample_length_multiplier,
            sample_weight=weight_values,
        )
        if precomputed_bags is None:
            bags = [
                sequential_bootstrap(starts, ends, sample_length=sample_length, rng=rng)
                for _ in range(self.n_estimators)
            ]
        else:
            bags = [np.asarray(bag, dtype=np.int32) for bag in precomputed_bags]
            if not bags:
                raise ValueError("precomputed_bags must contain at least one bag.")

        self.estimators = []
        for bag in bags:
            counts = np.bincount(bag, minlength=len(y_values)).astype(float)
            weights = weight_values * counts
            used = counts > 0
            tree = DecisionTreeClassifier(
                max_features=max(1, min(self.max_features, x_values.shape[1])),
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(x_values[used], y_values[used], sample_weight=weights[used])
            self.estimators.append(tree)
        self.estimators_ = self.estimators
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        if not self.estimators:
            raise RuntimeError("Model is not fitted.")
        x_values = x.to_numpy(dtype=float) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
        proba = np.zeros((len(x_values), 2), dtype=float)
        for tree in self.estimators:
            tree_proba = tree.predict_proba(x_values)
            for class_pos, class_value in enumerate(tree.classes_):
                proba[:, int(class_value)] += tree_proba[:, class_pos]
        return proba / len(self.estimators)

    def predict(self, x: Any) -> Any:
        prediction = np.argmax(self.predict_proba(x), axis=1).astype(np.int32)
        if isinstance(x, pd.DataFrame):
            return pd.Series(prediction, index=x.index, name="prediction")
        return prediction


def purged_walk_forward_splits(
    starts: np.ndarray,
    ends: np.ndarray,
    n_splits: int,
    embargo_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(starts)
    folds = [fold for fold in np.array_split(order, n_splits) if len(fold) > 0]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_idx in folds:
        test_start = int(np.min(starts[test_idx]))
        test_end = int(np.max(ends[test_idx]))
        train_mask = (ends < test_start - embargo_bars) | (starts > test_end + embargo_bars)
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def precompute_bootstrap_bags_by_fold(
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    symbol: str | None,
    show_progress: bool,
) -> dict[int, list[np.ndarray]]:
    n_estimators = int(meta_config.get("n_estimators", 100))
    max_samples = meta_config.get("max_samples", 1.0)
    sample_length_multiplier = float(bagging_config.get("sample_length_multiplier", 1.0))
    base_seed = int(bagging_config.get("random_seed", 2027))
    bags_by_fold: dict[int, list[np.ndarray]] = {}
    progress = tqdm(
        total=len(splits) * n_estimators,
        desc=f"{symbol or 'symbol'} bootstrap bags",
        disable=not show_progress,
        dynamic_ncols=True,
        leave=False,
        unit="bag",
    )
    try:
        for fold_idx, (train_idx, _) in enumerate(splits):
            rng = np.random.default_rng(base_seed + fold_idx)
            train_starts = starts[train_idx]
            train_ends = ends[train_idx]
            sample_length = bagging_sample_length(
                max_samples,
                len(train_idx),
                sample_length_multiplier,
                sample_weight=event_uniqueness(train_starts, train_ends),
            )
            bags = []
            for _ in range(n_estimators):
                bags.append(
                    sequential_bootstrap(
                        train_starts,
                        train_ends,
                        sample_length=sample_length,
                        rng=rng,
                    ),
                )
                progress.update(1)
            bags_by_fold[fold_idx] = bags
    finally:
        progress.close()
    return bags_by_fold


def feature_selection_config(meta_config: dict[str, Any]) -> dict[str, Any]:
    config = meta_config.get("feature_selection", {})
    return config if isinstance(config, dict) else {}


def pca_feature_selection_enabled(meta_config: dict[str, Any]) -> bool:
    config = feature_selection_config(meta_config)
    return bool(config.get("enabled", meta_config.get("pca_components") is not None))


def selected_feature_count(meta_config: dict[str, Any], available: int) -> int:
    if available < 1:
        raise ValueError("At least one feature is required")
    config = feature_selection_config(meta_config)
    top_n = config.get("top_n")
    if top_n is None:
        return available
    return max(1, min(int(top_n), available))


def write_importance_csv(frame: pd.DataFrame | None, path: Path) -> str | None:
    if frame is None or frame.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def aggregate_importance_frames(frames: list[pd.DataFrame], *, method: str) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        column
        for column in ("importance", "std", "baseline_score")
        if column in combined.columns
    ]
    grouped = (
        combined.groupby("feature", as_index=False)[numeric_columns]
        .mean()
        .sort_values("importance", ascending=False, ignore_index=True)
    )
    grouped.insert(0, "method", method)
    grouped["fold_count"] = combined.groupby("feature")["fold"].nunique().reindex(grouped["feature"]).to_numpy()
    return grouped


def replace_synthetic_dataset_features(dataset: AfmlDataset, features: pd.DataFrame) -> AfmlDataset:
    return AfmlDataset(
        close=dataset.close,
        features=features.reindex(dataset.features.index),
        events=dataset.events,
        labels=dataset.labels,
        sample_weight=dataset.sample_weight,
        volatility=dataset.volatility,
        t_events=dataset.t_events,
    )


def synthetic_meta_model_from_config(
    *,
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    seed_offset: int = 0,
    n_estimators_override: int | None = None,
) -> SequentialBaggedRandomForestClassifier:
    return SequentialBaggedRandomForestClassifier(
        n_estimators=(
            int(n_estimators_override)
            if n_estimators_override is not None
            else int(meta_config.get("n_estimators", 100))
        ),
        max_features=int(meta_config.get("max_features", 1)),
        max_samples=meta_config.get("max_samples", 1.0),
        min_samples_leaf=int(meta_config.get("min_samples_leaf", 5)),
        max_depth=meta_config.get("max_depth"),
        random_seed=int(bagging_config.get("random_seed", 2027)) + int(seed_offset),
        sample_length_multiplier=float(bagging_config.get("sample_length_multiplier", 1.0)),
    )


def fit_synthetic_meta_model(
    model: SequentialBaggedRandomForestClassifier,
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    *,
    precomputed_bags: Sequence[np.ndarray] | None = None,
) -> SequentialBaggedRandomForestClassifier:
    train_events = dataset.events.loc[train_index]
    model.fit(
        dataset.X.loc[train_index],
        dataset.y.loc[train_index],
        starts=train_events["event_idx"].to_numpy(dtype=np.int32),
        ends=train_events["event_end_idx"].to_numpy(dtype=np.int32),
        sample_weight=dataset.sample_weight.loc[train_index].to_numpy(dtype=float),
        precomputed_bags=precomputed_bags,
    )
    return model


def build_synthetic_meta_dataset(
    events: pd.DataFrame,
    outcome: SyntheticCandidateOutcome,
    feature_frame: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    *,
    label_threshold: float,
) -> tuple[AfmlDataset, pd.DataFrame]:
    event_index = pd.DatetimeIndex(pd.to_datetime(events["event_time"], utc=True))
    event_end_index = pd.DatetimeIndex(pd.to_datetime(events["event_end_time"], utc=True))
    feature_columns = [column for column in feature_frame.columns if column in events.columns]
    features = events[feature_columns].copy()
    features.index = event_index
    features = features.replace([np.inf, -np.inf], np.nan)

    labels = pd.DataFrame(
        {
            "ret": outcome.expected_payoff.astype(float),
            "bin": (outcome.expected_payoff > float(label_threshold)).astype(np.int32),
            "side": events["side"].to_numpy(dtype=np.int32),
        },
        index=event_index,
    )
    aligned_events = pd.DataFrame(
        {
            "t1": event_end_index,
            "trgt": float(outcome.unit),
            "side": events["side"].to_numpy(dtype=np.int32),
            "event_idx": events["event_idx"].to_numpy(dtype=np.int32),
            "event_end_idx": events["event_end_idx"].to_numpy(dtype=np.int32),
        },
        index=event_index,
    )
    outcome_frame = pd.DataFrame(
        {
            "expected_payoff": outcome.expected_payoff.astype(float),
            "payoff_std": outcome.payoff_std.astype(float),
            "payoff_standard_error": outcome.payoff_standard_error.astype(float),
            "pt_hit_rate": outcome.pt_hit_rate.astype(float),
            "sl_hit_rate": outcome.sl_hit_rate.astype(float),
            "vertical_rate": outcome.vertical_rate.astype(float),
            "mean_exit_step": outcome.mean_exit_step.astype(float),
        },
        index=event_index,
    )

    valid = features.notna().all(axis=1) & labels["bin"].notna()
    features = features.loc[valid]
    labels = labels.loc[valid]
    aligned_events = aligned_events.loc[valid]
    outcome_frame = outcome_frame.loc[valid]
    if labels.empty:
        raise ValueError("No synthetic labels remain after full feature alignment")

    starts = aligned_events["event_idx"].to_numpy(dtype=np.int32)
    ends = aligned_events["event_end_idx"].to_numpy(dtype=np.int32)
    sample_weight = pd.Series(event_uniqueness(starts, ends), index=labels.index, name="sample_weight")
    dataset = AfmlDataset(
        close=close,
        features=features,
        events=aligned_events,
        labels=labels,
        sample_weight=sample_weight,
        volatility=volatility,
        t_events=pd.DatetimeIndex(labels.index),
    )
    return dataset, outcome_frame


def split_train_validation_indices_for_pruning(
    dataset: AfmlDataset,
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
    validation_start_idx = int(dataset.events.loc[validation_index[0], "event_idx"])

    if int(embargo_bars) > 0:
        cutoff = validation_start_idx - int(embargo_bars)
        event_ends = dataset.events.loc[inner_train_index, "event_end_idx"].astype(int)
        inner_train_index = pd.DatetimeIndex(event_ends.index[event_ends <= cutoff])
    else:
        event_ends = dataset.events.loc[inner_train_index, "event_end_idx"].astype(int)
        inner_train_index = pd.DatetimeIndex(event_ends.index[event_ends < validation_start_idx])

    if inner_train_index.empty or validation_index.empty:
        raise ValueError("MDA pruning train/validation split is empty after purge and embargo")
    return inner_train_index, validation_index


def fit_synthetic_fold_with_feature_selection(
    *,
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    precomputed_bags: Sequence[np.ndarray] | None,
    fold_idx: int,
) -> SyntheticMetaFoldResult:
    diagnostics: dict[str, Any] = {
        "fold": fold_idx,
        "feature_selection_enabled": False,
        "raw_feature_count": int(dataset.X.shape[1]),
    }
    final_dataset = dataset
    final_feature_columns = list(dataset.X.columns)
    pca_importance: pd.DataFrame | None = None

    if pca_feature_selection_enabled(meta_config):
        pca_components = float(meta_config.get("pca_components", 0.95))
        pca_dataset, pca_transformer = _pca_dataset(
            dataset,
            train_index,
            pca_components=pca_components,
            pca_random_state=int(meta_config.get("pca_random_state", 42)) + fold_idx,
        )
        if pca_transformer is None:
            raise ValueError("PCA feature selection requires a fitted transformer")
        pca_model = synthetic_meta_model_from_config(
            meta_config=meta_config,
            bagging_config=bagging_config,
            seed_offset=20_000 + fold_idx,
        )
        pca_model = fit_synthetic_meta_model(
            pca_model,
            pca_dataset,
            train_index,
            precomputed_bags=precomputed_bags,
        )
        pca_importance = pca_mdi_feature_importance(
            pca_model,
            pca_transformer,
            dataset.X.columns,
        )
        top_n = selected_feature_count(meta_config, available=dataset.X.shape[1])
        final_feature_columns = pca_importance.head(top_n)["feature"].astype(str).tolist()
        final_dataset = replace_synthetic_dataset_features(
            dataset,
            dataset.features[final_feature_columns],
        )
        diagnostics.update(
            {
                "feature_selection_enabled": True,
                "pca_components": pca_components,
                "pca_component_count": int(pca_dataset.X.shape[1]),
                "selected_raw_feature_count": len(final_feature_columns),
            },
        )

        config = feature_selection_config(meta_config)
        if bool(config.get("mda_prune_negative", True)):
            pruning_train_index, pruning_validation_index = split_train_validation_indices_for_pruning(
                final_dataset,
                train_index,
                validation_fraction=float(config.get("mda_pruning_validation_fraction", 0.25)),
                embargo_bars=int(meta_config.get("embargo_bars", 80)),
            )
            provisional_model = synthetic_meta_model_from_config(
                meta_config=meta_config,
                bagging_config=bagging_config,
                seed_offset=30_000 + fold_idx,
            )
            provisional_model = fit_synthetic_meta_model(
                provisional_model,
                final_dataset,
                pruning_train_index,
            )
            pruning_mda = mda_feature_importance(
                provisional_model,
                final_dataset.X.loc[pruning_validation_index],
                final_dataset.y.loc[pruning_validation_index],
                sample_weight=final_dataset.sample_weight.loc[pruning_validation_index],
                n_repeats=int(config.get("mda_repeats", 3)),
                random_state=int(meta_config.get("pca_random_state", 42)) + fold_idx,
            )
            retained = pruning_mda.loc[pruning_mda["importance"] >= 0.0, "feature"].astype(str).tolist()
            if retained:
                final_feature_columns = [feature for feature in final_feature_columns if feature in set(retained)]
                final_dataset = replace_synthetic_dataset_features(
                    dataset,
                    dataset.features[final_feature_columns],
                )
            diagnostics.update(
                {
                    "mda_pruning_enabled": True,
                    "mda_pruning_retained_feature_count": len(final_feature_columns),
                    "mda_pruning_dropped_feature_count": int(len(pruning_mda) - len(final_feature_columns)),
                },
            )
        else:
            diagnostics["mda_pruning_enabled"] = False

    model = synthetic_meta_model_from_config(
        meta_config=meta_config,
        bagging_config=bagging_config,
        seed_offset=fold_idx,
    )
    model = fit_synthetic_meta_model(
        model,
        final_dataset,
        train_index,
        precomputed_bags=precomputed_bags,
    )
    probabilities = pd.Series(
        model.predict_proba(final_dataset.X.loc[test_index])[:, 1],
        index=test_index,
        name="meta_probability",
    )

    mdi: pd.DataFrame | None = None
    mda: pd.DataFrame | None = None
    sfi: pd.DataFrame | None = None
    if diagnostics["feature_selection_enabled"]:
        config = feature_selection_config(meta_config)
        mdi = mdi_feature_importance(model)
        mda = mda_feature_importance(
            model,
            final_dataset.X.loc[test_index],
            final_dataset.y.loc[test_index],
            sample_weight=final_dataset.sample_weight.loc[test_index],
            n_repeats=int(config.get("mda_repeats", 3)),
            random_state=int(meta_config.get("pca_random_state", 42)) + fold_idx,
        )
        sfi = sfi_feature_importance(
            final_dataset,
            train_index=train_index,
            test_index=test_index,
            model_factory=lambda: synthetic_meta_model_from_config(
                meta_config=meta_config,
                bagging_config=bagging_config,
                seed_offset=40_000 + fold_idx,
                n_estimators_override=int(config.get("sfi_n_estimators", 10)),
            ),
        )
    diagnostics["selected_feature_count"] = len(final_feature_columns)
    diagnostics["selected_features"] = final_feature_columns
    return SyntheticMetaFoldResult(
        model=model,
        dataset=final_dataset,
        probabilities=probabilities,
        feature_columns=final_feature_columns,
        diagnostics=diagnostics,
        pca_importance=pca_importance,
        mdi_importance=mdi,
        mda_importance=mda,
        sfi_importance=sfi,
    )


def candidate_screen_row(
    *,
    candidate_id: str,
    event_definition: dict[str, Any],
    exit_definition: dict[str, float],
    outcome: SyntheticCandidateOutcome,
    label_threshold: float,
    n_events: int,
) -> dict[str, Any]:
    expected = outcome.expected_payoff.astype(float)
    std_expected = float(np.std(expected, ddof=1)) if n_events > 1 else 0.0
    standard_error = std_expected / math.sqrt(float(n_events)) if n_events > 1 else 0.0
    mean_expected = float(np.mean(expected)) if n_events else float("nan")
    positive_rate = float(np.mean(expected > label_threshold)) if n_events else 0.0
    return {
        "candidate_id": candidate_id,
        "long_R2": float(event_definition["long"]["r2_threshold"]),
        "short_R2": float(event_definition["short"]["r2_threshold"]),
        "long_Z": float(event_definition["long"]["cvd_z_min"]),
        "short_Z": float(event_definition["short"]["cvd_z_max"]),
        "long_PT": float(exit_definition["long_profit_taking_mult"]),
        "long_SL": float(exit_definition["long_stop_loss_mult"]),
        "short_PT": float(exit_definition["short_profit_taking_mult"]),
        "short_SL": float(exit_definition["short_stop_loss_mult"]),
        "screen_n_events": int(n_events),
        "screen_label_positive_rate": positive_rate,
        "screen_eligible_for_meta": 0.0 < positive_rate < 1.0,
        "screen_mean_expected_log_return": mean_expected,
        "screen_median_expected_log_return": float(np.median(expected)) if n_events else None,
        "screen_sum_expected_log_return": float(np.sum(expected)) if n_events else 0.0,
        "screen_std_expected_log_return": std_expected,
        "screen_standard_error": standard_error,
        "screen_robust_expected_log_return": mean_expected - 2.0 * standard_error,
        "screen_sharpe_like": mean_expected / std_expected if std_expected > 0.0 else None,
        "screen_mean_path_payoff_std": float(np.mean(outcome.payoff_std)) if n_events else None,
        "screen_mean_pt_hit_rate": float(np.mean(outcome.pt_hit_rate)) if n_events else None,
        "screen_mean_sl_hit_rate": float(np.mean(outcome.sl_hit_rate)) if n_events else None,
        "screen_mean_vertical_rate": float(np.mean(outcome.vertical_rate)) if n_events else None,
        "screen_mean_exit_step": float(np.mean(outcome.mean_exit_step)) if n_events else None,
    }


def screening_only_candidate_result(
    *,
    screen_row: dict[str, Any],
    events: pd.DataFrame,
    outcome: SyntheticCandidateOutcome,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    expected = outcome.expected_payoff.astype(float)
    n_events = len(expected)
    std_expected = float(screen_row["screen_std_expected_log_return"] or 0.0)
    standard_error = float(screen_row["screen_standard_error"] or 0.0)
    robust_expected = float(screen_row["screen_robust_expected_log_return"])
    path_metrics = trade_path_metrics(
        expected,
        event_starts=events["event_idx"].to_numpy(dtype=np.int32),
    )
    return {
        "candidate_id": screen_row["candidate_id"],
        "long_R2": screen_row["long_R2"],
        "short_R2": screen_row["short_R2"],
        "long_Z": screen_row["long_Z"],
        "short_Z": screen_row["short_Z"],
        "long_PT": screen_row["long_PT"],
        "long_SL": screen_row["long_SL"],
        "short_PT": screen_row["short_PT"],
        "short_SL": screen_row["short_SL"],
        "n_test_events": 0,
        "n_accepted_events": 0,
        "accept_rate": 0.0,
        "mean_expected_log_return": float(screen_row["screen_mean_expected_log_return"]),
        "median_expected_log_return": screen_row["screen_median_expected_log_return"],
        "sum_expected_log_return": float(screen_row["screen_sum_expected_log_return"]),
        "std_expected_log_return": std_expected,
        "cross_event_standard_error": standard_error,
        "mean_path_payoff_std": screen_row["screen_mean_path_payoff_std"],
        "monte_carlo_standard_error": float(np.mean(outcome.payoff_standard_error)) if n_events else 0.0,
        "total_standard_error": standard_error,
        "robust_expected_log_return": robust_expected,
        "sharpe_like": screen_row["screen_sharpe_like"],
        "robust_sharpe_like": robust_expected / std_expected if std_expected > 0.0 else None,
        "meta_precision": None,
        "mean_predicted_probability": None,
        "mean_pt_hit_rate": screen_row["screen_mean_pt_hit_rate"],
        "mean_sl_hit_rate": screen_row["screen_mean_sl_hit_rate"],
        "mean_vertical_rate": screen_row["screen_mean_vertical_rate"],
        "mean_exit_step": screen_row["screen_mean_exit_step"],
        "meta_validation_status": status,
        "meta_validation_error": error,
        **path_metrics,
    }


def run_meta_model_candidate(
    events: pd.DataFrame,
    outcome: SyntheticCandidateOutcome,
    feature_frame: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    labeling_config: dict[str, Any],
    *,
    candidate_id: str,
    event_definition: dict[str, Any],
    exit_definition: dict[str, float],
    min_accepted_events: int,
    symbol: str | None = None,
    output_dir: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    dataset, outcome_frame = build_synthetic_meta_dataset(
        events,
        outcome,
        feature_frame,
        close,
        volatility,
        label_threshold=float(labeling_config.get("label_threshold", 0.0)),
    )
    starts = dataset.events["event_idx"].to_numpy(dtype=np.int32)
    ends = dataset.events["event_end_idx"].to_numpy(dtype=np.int32)
    splits = purged_walk_forward_splits(
        starts,
        ends,
        n_splits=int(meta_config.get("n_splits", 5)),
        embargo_bars=int(meta_config.get("embargo_bars", 80)),
    )
    if not splits:
        raise ValueError("Purged walk-forward produced no train/test splits")

    threshold = float(meta_config.get("class_probability_threshold", 0.5))
    y = dataset.y.to_numpy(dtype=np.int32)
    if len(np.unique(y)) < 2:
        raise ValueError("Candidate synthetic labels contain fewer than two classes")

    bags_by_fold = precompute_bootstrap_bags_by_fold(
        starts=starts,
        ends=ends,
        splits=splits,
        meta_config=meta_config,
        bagging_config=bagging_config,
        symbol=symbol,
        show_progress=show_progress,
    )
    accepted_returns: list[float] = []
    accepted_path_std: list[float] = []
    accepted_path_se: list[float] = []
    accepted_labels: list[int] = []
    accepted_probabilities: list[float] = []
    accepted_event_starts: list[int] = []
    pca_frames: list[pd.DataFrame] = []
    mdi_frames: list[pd.DataFrame] = []
    mda_frames: list[pd.DataFrame] = []
    sfi_frames: list[pd.DataFrame] = []
    fold_diagnostics: list[dict[str, Any]] = []
    n_test_total = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(np.unique(y[train_idx])) < 2:
            continue
        train_index = pd.DatetimeIndex(dataset.X.index[train_idx])
        test_index = pd.DatetimeIndex(dataset.X.index[test_idx])
        fold_result = fit_synthetic_fold_with_feature_selection(
            dataset=dataset,
            train_index=train_index,
            test_index=test_index,
            meta_config=meta_config,
            bagging_config=bagging_config,
            precomputed_bags=bags_by_fold[fold_idx],
            fold_idx=fold_idx,
        )
        proba = fold_result.probabilities.reindex(test_index).to_numpy(dtype=float)
        accept = proba >= threshold
        n_test_total += len(test_idx)
        if np.any(accept):
            test_outcome = outcome_frame.loc[test_index]
            accepted_returns.extend(test_outcome["expected_payoff"].to_numpy(dtype=float)[accept].tolist())
            accepted_path_std.extend(test_outcome["payoff_std"].to_numpy(dtype=float)[accept].tolist())
            accepted_path_se.extend(
                test_outcome["payoff_standard_error"].to_numpy(dtype=float)[accept].tolist(),
            )
            accepted_labels.extend(y[test_idx][accept].astype(int).tolist())
            accepted_probabilities.extend(proba[accept].astype(float).tolist())
            accepted_event_starts.extend(starts[test_idx][accept].astype(int).tolist())
        fold_diagnostics.append(fold_result.diagnostics)
        for frame, target in (
            (fold_result.pca_importance, pca_frames),
            (fold_result.mdi_importance, mdi_frames),
            (fold_result.mda_importance, mda_frames),
            (fold_result.sfi_importance, sfi_frames),
        ):
            if frame is not None and not frame.empty:
                with_fold = frame.copy()
                with_fold["fold"] = fold_idx
                target.append(with_fold)

    if len(accepted_returns) < min_accepted_events:
        raise ValueError(
            f"Candidate accepted {len(accepted_returns)} events, below minimum {min_accepted_events}",
        )

    accepted_returns_array = np.asarray(accepted_returns, dtype=float)
    accepted_path_std_array = np.asarray(accepted_path_std, dtype=float)
    accepted_path_se_array = np.asarray(accepted_path_se, dtype=float)
    accepted_labels_array = np.asarray(accepted_labels, dtype=float)
    accepted_probabilities_array = np.asarray(accepted_probabilities, dtype=float)
    accepted_event_starts_array = np.asarray(accepted_event_starts, dtype=np.int32)
    n_accepted = len(accepted_returns_array)
    std_expected = float(np.std(accepted_returns_array, ddof=1)) if n_accepted > 1 else 0.0
    cross_event_standard_error = (
        std_expected / math.sqrt(float(n_accepted)) if n_accepted > 1 else 0.0
    )
    monte_carlo_standard_error = float(
        math.sqrt(float(np.sum(accepted_path_se_array * accepted_path_se_array)))
        / float(n_accepted),
    )
    total_standard_error = math.sqrt(
        cross_event_standard_error * cross_event_standard_error
        + monte_carlo_standard_error * monte_carlo_standard_error,
    )
    mean_expected = float(np.mean(accepted_returns_array))
    robust_expected = mean_expected - 2.0 * total_standard_error
    path_metrics = trade_path_metrics(accepted_returns_array, event_starts=accepted_event_starts_array)
    feature_selection_enabled = pca_feature_selection_enabled(meta_config)
    pca_importance = aggregate_importance_frames(pca_frames, method="PCA_MDI_BACKPROJECTED")
    mdi_importance = aggregate_importance_frames(mdi_frames, method="MDI")
    mda_importance = aggregate_importance_frames(mda_frames, method="MDA")
    sfi_importance = aggregate_importance_frames(sfi_frames, method="SFI")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    pca_importance_csv = (
        write_importance_csv(pca_importance, output_dir / f"{candidate_id}_pca_backprojected_importance.csv")
        if output_dir is not None
        else None
    )
    mdi_importance_csv = (
        write_importance_csv(mdi_importance, output_dir / f"{candidate_id}_selected_mdi_importance.csv")
        if output_dir is not None
        else None
    )
    mda_importance_csv = (
        write_importance_csv(mda_importance, output_dir / f"{candidate_id}_selected_mda_importance.csv")
        if output_dir is not None
        else None
    )
    sfi_importance_csv = (
        write_importance_csv(sfi_importance, output_dir / f"{candidate_id}_selected_sfi_importance.csv")
        if output_dir is not None
        else None
    )
    return {
        "candidate_id": candidate_id,
        "long_R2": float(event_definition["long"]["r2_threshold"]),
        "short_R2": float(event_definition["short"]["r2_threshold"]),
        "long_Z": float(event_definition["long"]["cvd_z_min"]),
        "short_Z": float(event_definition["short"]["cvd_z_max"]),
        "long_PT": float(exit_definition["long_profit_taking_mult"]),
        "long_SL": float(exit_definition["long_stop_loss_mult"]),
        "short_PT": float(exit_definition["short_profit_taking_mult"]),
        "short_SL": float(exit_definition["short_stop_loss_mult"]),
        "n_test_events": int(n_test_total),
        "n_accepted_events": n_accepted,
        "accept_rate": float(n_accepted / max(n_test_total, 1)),
        "mean_expected_log_return": mean_expected,
        "median_expected_log_return": float(np.median(accepted_returns_array)),
        "sum_expected_log_return": float(np.sum(accepted_returns_array)),
        "std_expected_log_return": std_expected,
        "cross_event_standard_error": cross_event_standard_error,
        "mean_path_payoff_std": float(np.mean(accepted_path_std_array)),
        "monte_carlo_standard_error": monte_carlo_standard_error,
        "total_standard_error": total_standard_error,
        "robust_expected_log_return": robust_expected,
        "sharpe_like": mean_expected / std_expected if std_expected > 0.0 else None,
        "robust_sharpe_like": robust_expected / std_expected if std_expected > 0.0 else None,
        "meta_precision": float(np.mean(accepted_labels_array)),
        "mean_predicted_probability": float(np.mean(accepted_probabilities_array)),
        "mean_pt_hit_rate": float(np.mean(outcome.pt_hit_rate)),
        "mean_sl_hit_rate": float(np.mean(outcome.sl_hit_rate)),
        "mean_vertical_rate": float(np.mean(outcome.vertical_rate)),
        "mean_exit_step": float(np.mean(outcome.mean_exit_step)),
        "meta_feature_count": int(dataset.X.shape[1]),
        "feature_selection_enabled": feature_selection_enabled,
        "pca_importance_csv": pca_importance_csv,
        "final_mdi_csv": mdi_importance_csv,
        "final_mda_csv": mda_importance_csv,
        "final_sfi_csv": sfi_importance_csv,
        "fold_diagnostics": fold_diagnostics,
        "top_mdi_features": mdi_importance.head(10).to_dict(orient="records"),
        "top_mda_features": mda_importance.head(10).to_dict(orient="records"),
        "top_sfi_features": sfi_importance.head(10).to_dict(orient="records"),
        **path_metrics,
    }


def search_space_config(primary_optuna_config: dict[str, Any], name: str) -> dict[str, Any]:
    search_space = section(primary_optuna_config, "search_space")
    value = search_space.get(name)
    return value if isinstance(value, dict) else {}


def suggest_float(
    trial: Any,
    primary_optuna_config: dict[str, Any],
    trial_name: str,
    *,
    default_low: float,
    default_high: float,
    log: bool = False,
) -> float:
    config = search_space_config(primary_optuna_config, trial_name)
    low = float(config.get("low", default_low))
    high = float(config.get("high", default_high))
    if high <= low:
        raise ValueError(f"Invalid Optuna search range for {trial_name}: low={low} high={high}")
    step = config.get("step")
    if step is not None:
        return float(trial.suggest_float(trial_name, low, high, step=float(step)))
    return float(trial.suggest_float(trial_name, low, high, log=bool(config.get("log", log))))


def runtime_candidate_from_trial(
    trial: Any,
    *,
    event_config: dict[str, Any],
    primary_optuna_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    long_r2 = suggest_float(
        trial,
        primary_optuna_config,
        "long_R2",
        default_low=0.40,
        default_high=0.70,
    )
    short_r2 = suggest_float(
        trial,
        primary_optuna_config,
        "short_R2",
        default_low=0.35,
        default_high=0.65,
    )
    long_z = suggest_float(
        trial,
        primary_optuna_config,
        "long_Z",
        default_low=1.0,
        default_high=2.5,
    )
    short_z = suggest_float(
        trial,
        primary_optuna_config,
        "short_Z",
        default_low=-2.2,
        default_high=-0.8,
    )
    long_pt = suggest_float(
        trial,
        primary_optuna_config,
        "long_PT",
        default_low=0.5,
        default_high=4.0,
    )
    long_sl = suggest_float(
        trial,
        primary_optuna_config,
        "long_SL",
        default_low=0.5,
        default_high=4.0,
    )
    short_pt = suggest_float(
        trial,
        primary_optuna_config,
        "short_PT",
        default_low=0.5,
        default_high=4.0,
    )
    short_sl = suggest_float(
        trial,
        primary_optuna_config,
        "short_SL",
        default_low=0.5,
        default_high=4.0,
    )
    event_definition = explicit_event_definition(
        event_config,
        long_r2_threshold=long_r2,
        short_r2_threshold=short_r2,
        long_cvd_z_min=long_z,
        short_cvd_z_max=short_z,
    )
    exit_definition = {
        "long_profit_taking_mult": long_pt,
        "long_stop_loss_mult": long_sl,
        "short_profit_taking_mult": short_pt,
        "short_stop_loss_mult": short_sl,
    }
    return event_definition, exit_definition


def sort_candidate_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "robust_expected_log_return",
            "sharpe_like",
            "mean_expected_log_return",
            "sum_expected_log_return",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def static_exit_definition(runtime_config: dict[str, Any]) -> dict[str, float]:
    return {
        "long_profit_taking_mult": required_float(
            runtime_config,
            "long_profit_taking_mult",
            "runtime_strategy",
        ),
        "long_stop_loss_mult": required_float(
            runtime_config,
            "long_stop_loss_mult",
            "runtime_strategy",
        ),
        "short_profit_taking_mult": required_float(
            runtime_config,
            "short_profit_taking_mult",
            "runtime_strategy",
        ),
        "short_stop_loss_mult": required_float(
            runtime_config,
            "short_stop_loss_mult",
            "runtime_strategy",
        ),
    }


def evaluate_static_runtime_candidate(
    *,
    symbol: str,
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    log_trend: np.ndarray,
    log_residual: np.ndarray,
    model: Any,
    event_config: dict[str, Any],
    barrier_config: dict[str, Any],
    runtime_config: dict[str, Any],
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    labeling_config: dict[str, Any],
    primary_optuna_config: dict[str, Any],
    base_seed: int,
    diagnostics_dir: Path | None,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizon = int(barrier_config.get("vertical_barrier_bars", 80))
    min_primary_events = int(primary_optuna_config.get("min_primary_events", 50))
    min_accepted_events = int(primary_optuna_config.get("min_accepted_events", 10))
    candidate_id = "static_runtime"
    event_definition = explicit_event_definition(event_config)
    exit_definition = static_exit_definition(runtime_config)
    payload = runtime_candidate_payload(
        event_definition=event_definition,
        exit_definition=exit_definition,
        barrier_config=barrier_config,
        vertical_barrier_bars=horizon,
    )
    events = build_primary_events(
        symbol,
        df=df,
        feature_frame=feature_frame,
        log_trend=log_trend,
        log_residual=log_residual,
        event_config=event_definition,
        horizon=horizon,
    )
    if len(events) < min_primary_events:
        raise ValueError(f"static primary_events {len(events)} < min_primary_events {min_primary_events}")

    outcome = synthetic_outcome_for_runtime_candidate(
        events,
        model=model,
        barrier_config=barrier_config,
        labeling_config=labeling_config,
        symbol_seed=deterministic_symbol_seed(base_seed, symbol),
        exit_definition=exit_definition,
        symbol=symbol,
        show_progress=show_progress,
    )
    screen_row = candidate_screen_row(
        candidate_id=candidate_id,
        event_definition=event_definition,
        exit_definition=exit_definition,
        outcome=outcome,
        label_threshold=float(labeling_config.get("label_threshold", 0.0)),
        n_events=len(events),
    )
    screen_result = pd.DataFrame([screen_row])
    screen_result["screen_rank"] = 1
    if not bool(screen_row["screen_eligible_for_meta"]):
        reason = (
            "static runtime candidate labels are not meta-eligible: "
            f"positive_rate={screen_row['screen_label_positive_rate']:.6g}"
        )
        LOGGER.warning("%s %s", symbol, reason)
        result = screening_only_candidate_result(
            screen_row=screen_row,
            events=events,
            outcome=outcome,
            status="skipped_single_class_labels",
            error=reason,
        )
    else:
        try:
            result = run_meta_model_candidate(
                events,
                outcome=outcome,
                feature_frame=feature_frame,
                close=close,
                volatility=volatility,
                meta_config=meta_config,
                bagging_config=bagging_config,
                labeling_config=labeling_config,
                candidate_id=candidate_id,
                event_definition=event_definition,
                exit_definition=exit_definition,
                min_accepted_events=min_accepted_events,
                symbol=symbol,
                output_dir=diagnostics_dir / candidate_id if diagnostics_dir is not None else None,
                show_progress=show_progress,
            )
            result["meta_validation_status"] = "success"
            result["meta_validation_error"] = None
        except ValueError as exc:
            reason = str(exc)
            LOGGER.warning("%s static runtime candidate meta validation failed: %s", symbol, reason)
            result = screening_only_candidate_result(
                screen_row=screen_row,
                events=events,
                outcome=outcome,
                status="failed",
                error=reason,
            )
    candidate_result = sort_candidate_results(pd.DataFrame([result]))
    candidate_result["selected_for_runtime"] = True
    top_candidates = candidate_result.copy()
    top_candidates["runtime_strategy_candidate"] = [payload]
    static_summary = {
        "enabled": False,
        "mode": "static_runtime_candidate",
        "candidate_id": candidate_id,
        "min_primary_events": min_primary_events,
        "min_accepted_events": min_accepted_events,
        "n_paths_per_event": int(labeling_config.get("n_paths_per_event", 25_000)),
        "meta_validation_status": result["meta_validation_status"],
        "meta_validation_error": result["meta_validation_error"],
    }
    return events, screen_result, candidate_result, top_candidates, {
        "runtime_strategy_candidate": payload,
        "primary_optuna": static_summary,
    }


def optimize_runtime_candidates_with_optuna(
    *,
    symbol: str,
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    log_trend: np.ndarray,
    log_residual: np.ndarray,
    model: Any,
    event_config: dict[str, Any],
    barrier_config: dict[str, Any],
    meta_config: dict[str, Any],
    bagging_config: dict[str, Any],
    labeling_config: dict[str, Any],
    primary_optuna_config: dict[str, Any],
    base_seed: int,
    diagnostics_dir: Path | None,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("primary_optuna.enabled requires optuna") from exc

    horizon = int(barrier_config.get("vertical_barrier_bars", 80))
    min_primary_events = int(primary_optuna_config.get("min_primary_events", 50))
    min_accepted_events = int(primary_optuna_config.get("min_accepted_events", 10))
    n_trials = int(primary_optuna_config.get("n_trials", 32))
    timeout = primary_optuna_config.get("timeout_seconds")
    timeout = None if timeout is None else float(timeout)
    optuna_labeling_config = dict(labeling_config)
    if primary_optuna_config.get("n_paths_per_event") is not None:
        optuna_labeling_config["n_paths_per_event"] = int(primary_optuna_config["n_paths_per_event"])

    screen_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    event_frames: dict[str, pd.DataFrame] = {}
    candidate_payloads: dict[str, dict[str, Any]] = {}

    def objective(trial: Any) -> float:
        candidate_id = f"trial_{trial.number:04d}"
        event_definition, exit_definition = runtime_candidate_from_trial(
            trial,
            event_config=event_config,
            primary_optuna_config=primary_optuna_config,
        )
        candidate_payloads[candidate_id] = runtime_candidate_payload(
            event_definition=event_definition,
            exit_definition=exit_definition,
            barrier_config=barrier_config,
            vertical_barrier_bars=horizon,
        )
        try:
            events = build_primary_events(
                symbol,
                df=df,
                feature_frame=feature_frame,
                log_trend=log_trend,
                log_residual=log_residual,
                event_config=event_definition,
                horizon=horizon,
            )
        except ValueError as exc:
            row = {
                "candidate_id": candidate_id,
                "failure_reason": str(exc),
                "screen_n_events": 0,
                "screen_eligible_for_meta": False,
            }
            screen_rows.append(row)
            trial.set_user_attr("failure_reason", str(exc))
            return NEGATIVE_OBJECTIVE

        if len(events) < min_primary_events:
            reason = f"primary_events {len(events)} < min_primary_events {min_primary_events}"
            screen_rows.append(
                {
                    "candidate_id": candidate_id,
                    "failure_reason": reason,
                    "screen_n_events": len(events),
                    "screen_eligible_for_meta": False,
                },
            )
            trial.set_user_attr("failure_reason", reason)
            return NEGATIVE_OBJECTIVE

        seed = deterministic_symbol_seed(base_seed + trial.number, symbol)
        outcome = synthetic_outcome_for_runtime_candidate(
            events,
            model=model,
            barrier_config=barrier_config,
            labeling_config=optuna_labeling_config,
            symbol_seed=seed,
            exit_definition=exit_definition,
            symbol=symbol,
            show_progress=show_progress,
        )
        screen_row = candidate_screen_row(
            candidate_id=candidate_id,
            event_definition=event_definition,
            exit_definition=exit_definition,
            outcome=outcome,
            label_threshold=float(optuna_labeling_config.get("label_threshold", 0.0)),
            n_events=len(events),
        )
        screen_rows.append(screen_row)
        if not screen_row["screen_eligible_for_meta"]:
            trial.set_user_attr("failure_reason", "candidate labels are not meta-eligible")
            return NEGATIVE_OBJECTIVE

        try:
            result = run_meta_model_candidate(
                events,
                outcome=outcome,
                feature_frame=feature_frame,
                close=close,
                volatility=volatility,
                meta_config=meta_config,
                bagging_config=bagging_config,
                labeling_config=optuna_labeling_config,
                candidate_id=candidate_id,
                event_definition=event_definition,
                exit_definition=exit_definition,
                min_accepted_events=min_accepted_events,
                symbol=symbol,
                output_dir=diagnostics_dir / candidate_id if diagnostics_dir is not None else None,
                show_progress=show_progress,
            )
        except ValueError as exc:
            trial.set_user_attr("failure_reason", str(exc))
            return NEGATIVE_OBJECTIVE

        result_rows.append(result)
        event_frames[candidate_id] = events
        trial.set_user_attr("candidate_id", candidate_id)
        trial.set_user_attr("n_events", len(events))
        trial.set_user_attr("n_accepted_events", int(result["n_accepted_events"]))
        trial.set_user_attr("robust_expected_log_return", float(result["robust_expected_log_return"]))
        return float(result["robust_expected_log_return"])

    sampler = optuna.samplers.TPESampler(seed=int(primary_optuna_config.get("random_seed", base_seed)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress)

    if not result_rows:
        raise ValueError("Optuna primary/exit search produced no candidate satisfying trade constraints")

    screen_result = pd.DataFrame(screen_rows)
    if not screen_result.empty and "screen_robust_expected_log_return" in screen_result:
        screen_result = screen_result.sort_values(
            [
                "screen_eligible_for_meta",
                "screen_robust_expected_log_return",
                "screen_sharpe_like",
                "screen_mean_expected_log_return",
            ],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
        screen_result["screen_rank"] = np.arange(1, len(screen_result) + 1)

    candidate_result = sort_candidate_results(pd.DataFrame(result_rows))
    best_candidate_id = str(candidate_result.iloc[0]["candidate_id"])
    events = event_frames[best_candidate_id]
    best_payload = candidate_payloads[best_candidate_id]
    candidate_result["selected_for_runtime"] = candidate_result["candidate_id"] == best_candidate_id
    top_n = int(labeling_config.get("candidate_export_top_n", 10))
    top_candidates = candidate_result.head(top_n).copy()
    top_candidates["runtime_strategy_candidate"] = top_candidates["candidate_id"].map(candidate_payloads)
    optuna_summary = {
        "enabled": True,
        "n_trials": n_trials,
        "completed_trials": len(study.trials),
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "min_primary_events": min_primary_events,
        "min_accepted_events": min_accepted_events,
        "n_paths_per_event": int(optuna_labeling_config.get("n_paths_per_event", 25_000)),
    }
    return events, screen_result, candidate_result, top_candidates, {
        "runtime_strategy_candidate": best_payload,
        "primary_optuna": optuna_summary,
    }


def backtest_symbol(
    symbol: str,
    input_csv: Path,
    output_dir: Path,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    cli_n_paths_per_event: int | None,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    LOGGER.info("%s start: input=%s", symbol, input_csv)
    synthetic_config = section(config, "synthetic_data")
    price_model_config = section(synthetic_config, "price_model")
    fair_value_config = section(synthetic_config, "fair_value")
    trend_config = section(synthetic_config, "trend_process")
    ou_config = section(synthetic_config, "ou_process")
    barrier_config = section(strategy_config, "barrier_optimization")
    event_config = section(strategy_config, "event_definition")
    runtime_config = section(strategy_config, "runtime_strategy")
    primary_optuna_config = section(strategy_config, "primary_optuna")
    meta_config = section(strategy_config, "meta_model")
    bagging_config = section(strategy_config, "sequential_bagging")
    feature_config = section(strategy_config, "feature_engineering")
    labeling_config = dict(section(strategy_config, "synthetic_labeling"))
    if cli_n_paths_per_event is not None:
        labeling_config["n_paths_per_event"] = cli_n_paths_per_event

    symbol_output_dir = output_dir / symbol
    summary_path = symbol_output_dir / f"{symbol}_synthetic_ml_summary.json"
    fingerprint = run_fingerprint(
        symbol=symbol,
        input_csv=input_csv,
        price_model_config=price_model_config,
        fair_value_config=fair_value_config,
        trend_config=trend_config,
        ou_config=ou_config,
        barrier_config=barrier_config,
        event_config=event_config,
        meta_config=meta_config,
        bagging_config=bagging_config,
        labeling_config=labeling_config,
        primary_optuna_config=primary_optuna_config,
        feature_config=feature_config,
    )
    if not force:
        cached = load_cached_summary(summary_path, fingerprint)
        if cached is not None:
            LOGGER.info("%s cache hit: summary=%s", symbol, summary_path)
            return cached

    price_column = str(price_model_config.get("price_column", "close"))
    df = frame_with_datetime_index(read_real_bars(input_csv, price_column=price_column))
    LOGGER.info("%s loaded bars: rows=%s price_column=%s", symbol, f"{len(df):,}", price_column)
    model = fit_joint_trend_ou_model(
        df,
        price_column=price_column,
        fair_value_config=fair_value_config,
        trend_config=trend_config,
        ou_config=ou_config,
        barrier_config=barrier_config,
    )
    LOGGER.info(
        "%s fitted model: ou_phi=%.6g half_life_bars=%.3f ewma_1bar_vol=%.6g",
        symbol,
        model.ou_phi,
        model.ou_half_life_bars,
        model.observed_ewma_1bar_log_return_volatility,
    )
    features, log_trend, log_residual, volatility = build_synthetic_feature_frame(
        df,
        price_column=price_column,
        fair_value_config=fair_value_config,
        event_config=event_config,
        barrier_config=barrier_config,
        feature_config=feature_config,
    )
    close = df[price_column].dropna().astype(float)
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = symbol_output_dir / "feature_diagnostics"
    horizon = int(barrier_config.get("vertical_barrier_bars", 80))

    if bool(primary_optuna_config.get("enabled", False)):
        LOGGER.info("%s primary Optuna search enabled: n_trials=%s", symbol, primary_optuna_config.get("n_trials", 32))
        events, screen_result, candidate_result, top_candidates_frame, optuna_artifacts = (
            optimize_runtime_candidates_with_optuna(
                symbol=symbol,
                df=df,
                feature_frame=features,
                close=close,
                volatility=volatility,
                log_trend=log_trend,
                log_residual=log_residual,
                model=model,
                event_config=event_config,
                barrier_config=barrier_config,
                meta_config=meta_config,
                bagging_config=bagging_config,
                labeling_config=labeling_config,
                primary_optuna_config=primary_optuna_config,
                base_seed=int(labeling_config.get("random_seed", 1729)),
                diagnostics_dir=diagnostics_dir,
                show_progress=show_progress,
            )
        )
        selection_rule = "Optuna primary/exit candidates with purged walk-forward meta validation"
    else:
        LOGGER.info("%s primary Optuna disabled: evaluating static runtime candidate", symbol)
        events, screen_result, candidate_result, top_candidates_frame, optuna_artifacts = (
            evaluate_static_runtime_candidate(
                symbol=symbol,
                df=df,
                feature_frame=features,
                close=close,
                volatility=volatility,
                log_trend=log_trend,
                log_residual=log_residual,
                model=model,
                event_config=event_config,
                barrier_config=barrier_config,
                runtime_config=runtime_config,
                meta_config=meta_config,
                bagging_config=bagging_config,
                labeling_config=labeling_config,
                primary_optuna_config=primary_optuna_config,
                base_seed=int(labeling_config.get("random_seed", 1729)),
                diagnostics_dir=diagnostics_dir,
                show_progress=show_progress,
            )
        )
        selection_rule = "Static primary/exit runtime candidate with purged walk-forward meta validation"
    outcome_unit = barrier_unit(model, barrier_config)

    events_path = symbol_output_dir / f"{symbol}_primary_events.csv"
    screen_path = symbol_output_dir / f"{symbol}_synthetic_ml_screening.csv"
    candidates_path = symbol_output_dir / f"{symbol}_synthetic_ml_runtime_candidates.csv"
    events.to_csv(events_path, index=False)
    screen_result.to_csv(screen_path, index=False)
    candidate_result.to_csv(candidates_path, index=False)
    best = candidate_result.iloc[0].to_dict()
    candidate_export_top_n = int(labeling_config.get("candidate_export_top_n", 10))
    top_candidates = top_candidates_frame.head(candidate_export_top_n).to_dict(orient="records")
    runtime_strategy_candidate = optuna_artifacts["runtime_strategy_candidate"]
    summary = {
        "symbol": symbol,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "source_csv": str(input_csv),
        "events_csv": str(events_path),
        "screening_csv": str(screen_path),
        "candidates_csv": str(candidates_path),
        "n_events": len(events),
        "n_long_events": int(np.sum(events["side"] == 1)),
        "n_short_events": int(np.sum(events["side"] == -1)),
        "n_paths_per_event": int(labeling_config.get("n_paths_per_event", 25_000)),
        "vertical_barrier_bars": horizon,
        "barrier_unit": outcome_unit,
        "volatility_target": volatility_target_metadata(barrier_config),
        "round_trip_cost": round_trip_cost_from_labeling_config(labeling_config),
        "screening": {
            "candidate_points": len(candidate_result),
            "selection_rule": selection_rule,
            "top_synthetic_candidate": screen_result.iloc[0].to_dict(),
        },
        "primary_optuna": optuna_artifacts["primary_optuna"],
        "event_definition": event_config,
        "meta_model": meta_config,
        "feature_engineering": feature_config,
        "n_meta_features": int(features.shape[1]),
        "sequential_bagging": bagging_config,
        "synthetic_labeling": labeling_config,
        "strategy_config_path": strategy_config.get("_strategy_config_path"),
        "model_fit": {
            "ou_phi": model.ou_phi,
            "ou_kappa": model.ou_kappa,
            "ou_half_life_bars": model.ou_half_life_bars,
            "ou_innovation_std": model.ou_innovation_std,
            "observed_ewma_1bar_log_return_volatility": model.observed_ewma_1bar_log_return_volatility,
        },
        "best": best,
        "top_candidates": top_candidates,
        "runtime_strategy_candidate": runtime_strategy_candidate,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    LOGGER.info(
        "%s complete: best=%s robust_E=%.6g summary=%s",
        symbol,
        best["candidate_id"],
        best["robust_expected_log_return"],
        summary_path,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AFML synthetic-path ML barrier backtest with primary trend/CVD events, "
            "triple-barrier synthetic labels, and sequential-bagged RF meta-model."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to AFML data config JSON.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--input-csv", default=None, help="Only valid with one --symbols value.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-paths-per-event", type=int, default=None)
    parser.add_argument(
        "--strategy-config",
        default=None,
        help=f"Override strategy JSON path (default: {DEFAULT_SYNTHETIC_STRATEGY_CONFIG}).",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached symbol outputs and retrain.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_afml_data_config(args.config)
    strategy_config = load_strategy_config(config, args.strategy_config or DEFAULT_SYNTHETIC_STRATEGY_CONFIG)
    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else resolve_repo_path(strategy_config.get("output_dir", "afml_scripts/output/ou_synthetic/ml_backtest"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = configured_symbols(config, args.symbols)
    LOGGER.info("run start: symbols=%s output_dir=%s", ", ".join(symbols), output_dir)
    summaries = []
    for symbol in symbols:
        input_csv = input_csv_for_symbol(
            symbol,
            config,
            cli_input_csv=args.input_csv,
            total_symbols=len(symbols),
        )
        summary = backtest_symbol(
            symbol,
            input_csv=input_csv,
            output_dir=output_dir,
            config=config,
            strategy_config=strategy_config,
            cli_n_paths_per_event=args.n_paths_per_event,
            force=args.force,
            show_progress=not args.no_progress,
        )
        summaries.append(summary)
        best = summary["best"]
        cache_suffix = " cache=hit" if summary.get("cache_hit") else ""
        best_label = f"candidate={best['candidate_id']}"
        print(
            f"{symbol}: events={summary['n_events']:,} "
            f"{best_label} "
            f"sharpe={format_metric(best.get('sharpe_like'))} "
            f"robust_E={best['robust_expected_log_return']:.6g}{cache_suffix} "
            f"E={best['mean_expected_log_return']:.6g} accepted={best['n_accepted_events']}",
        )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "symbols": symbols,
        "summaries": summaries,
    }
    manifest_path = output_dir / "synthetic_ml_backtest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    LOGGER.info("run complete: manifest=%s", manifest_path)
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
