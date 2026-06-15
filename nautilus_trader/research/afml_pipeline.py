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
"""
AFML-style event sampling, labeling, weighting, and signal generation.

This module is organized in the same broad order as de Prado's *Advances in
Financial Machine Learning*: Chapter 2 data structures/event sampling, Chapter
3 labeling, Chapter 4 sample weighting/bootstrap, Chapter 5 feature
fractional differentiation, Chapter 6 ensemble fitting, then later diagnostics,
bet sizing, backtest statistics, and structural/entropy/microstructural
features.

The executable workflow remains:
``close -> CUSUM events -> volatility targets -> triple-barrier labels -> model signals``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


try:
    from sklearn.base import BaseEstimator
    from sklearn.base import ClassifierMixin
except ImportError:

    class BaseEstimator:
        pass

    class ClassifierMixin:
        pass


from nautilus_trader.model.data import Bar
from nautilus_trader.persistence.catalog import ParquetDataCatalog


NANOSECONDS_PER_SECOND = 1_000_000_000
TimeLike = str | pd.Timestamp | None
DEFAULT_FEATURE_FRACDIFF_D = 0.4
DEFAULT_FEATURE_FRACDIFF_THRESHOLD = 0.01
DEFAULT_MAX_SAMPLE_WEIGHT = 10.0

# -------------------------------------------------------------------------------------------------
# Shared contracts and utilities
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AfmlDataset:
    """
    A fully aligned training dataset.
    """

    close: pd.Series
    features: pd.DataFrame
    events: pd.DataFrame
    labels: pd.DataFrame
    sample_weight: pd.Series
    volatility: pd.Series
    t_events: pd.DatetimeIndex
    cusum_threshold: pd.Series | None = None

    @property
    def X(self) -> pd.DataFrame:
        return self.features.loc[self.labels.index]

    @property
    def y(self) -> pd.Series:
        return self.labels["bin"].astype(int)


@dataclass(frozen=True)
class AfmlFitResult:
    """
    Result from fitting an AFML classifier and creating out-of-sample signals.
    """

    model: Any
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    train_accuracy: float
    test_accuracy: float
    signals: pd.DataFrame
    feature_transformer: Any | None = None


@dataclass(frozen=True)
class AfmlMetaFitResult:
    """
    Result from fitting a high-recall side model and high-precision meta model.
    """

    primary_model: Any
    meta_model: Any
    primary_dataset: AfmlDataset
    meta_dataset: AfmlDataset
    primary_train_index: pd.DatetimeIndex
    primary_test_index: pd.DatetimeIndex
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    primary_train_accuracy: float
    primary_test_accuracy: float
    meta_train_precision: float
    meta_train_recall: float
    meta_test_precision: float
    meta_test_recall: float
    signals: pd.DataFrame
    primary_feature_transformer: Any | None = None
    meta_feature_transformer: Any | None = None


def _as_utc_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if not dt_index.is_monotonic_increasing:
        dt_index = dt_index.sort_values()
    return dt_index


def _time_bound(value: TimeLike, *, is_end: bool = False) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, str):
        timestamp = pd.Timestamp(value, tz="UTC")
        if is_end and len(value) == 10:
            return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return timestamp
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _date_mask(index: pd.DatetimeIndex, start: TimeLike, end: TimeLike) -> np.ndarray:
    index = _as_utc_datetime_index(index)
    mask = np.ones(len(index), dtype=bool)
    start_ts = _time_bound(start)
    end_ts = _time_bound(end, is_end=True)
    if start_ts is not None:
        mask &= index >= start_ts
    if end_ts is not None:
        mask &= index <= end_ts
    return mask


# -------------------------------------------------------------------------------------------------
# Chapter 2 - Financial data structures and event sampling
# -------------------------------------------------------------------------------------------------


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    """
    Convert Nautilus bars to an OHLCV DataFrame indexed by UTC event timestamp.
    """
    if not bars:
        raise ValueError("Cannot build a DataFrame from an empty bar list")

    records = [
        {
            "timestamp": pd.Timestamp(bar.ts_event, unit="ns", tz="UTC"),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "bar_type": str(bar.bar_type),
            "ts_event": int(bar.ts_event),
            "ts_init": int(bar.ts_init),
        }
        for bar in bars
    ]
    frame = pd.DataFrame.from_records(records).set_index("timestamp")
    frame = frame.sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def load_bars_from_catalog(
    catalog_path: str | Path,
    *,
    bar_types: list[str] | None = None,
    instrument_ids: list[str] | None = None,
    start: int | str | None = None,
    end: int | str | None = None,
) -> list[Bar]:
    """
    Load bars from a Nautilus ``ParquetDataCatalog``.
    """
    catalog = ParquetDataCatalog(Path(catalog_path).resolve())
    bars = catalog.bars(
        bar_types=bar_types,
        instrument_ids=instrument_ids,
        start=start,
        end=end,
    )
    if not bars:
        raise ValueError("No bars were found in the requested catalog slice")
    return bars


def load_close_from_catalog(
    catalog_path: str | Path,
    *,
    bar_types: list[str] | None = None,
    instrument_ids: list[str] | None = None,
    start: int | str | None = None,
    end: int | str | None = None,
) -> pd.Series:
    """
    Load a close series from Nautilus catalog bars.
    """
    bars = load_bars_from_catalog(
        catalog_path,
        bar_types=bar_types,
        instrument_ids=instrument_ids,
        start=start,
        end=end,
    )
    return bars_to_frame(bars)["close"].rename("close")


def load_ohlcv_csv(
    csv_path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    timestamp_unit: str | None = None,
) -> pd.DataFrame:
    """
    Load a CSV containing OHLCV-like columns into a UTC-indexed DataFrame.

    ``timestamp_unit`` can be set to values accepted by ``pd.to_datetime`` such as
    ``"ns"``, ``"ms"``, or ``"s"`` when the timestamp column is numeric.
    """
    frame = pd.read_csv(csv_path)
    if timestamp_col not in frame.columns:
        raise ValueError(f"CSV is missing timestamp column {timestamp_col!r}")
    if close_col not in frame.columns:
        raise ValueError(f"CSV is missing close column {close_col!r}")

    timestamps = frame.pop(timestamp_col)
    if timestamp_unit is not None:
        index = pd.to_datetime(timestamps, unit=timestamp_unit, utc=True)
    else:
        index = pd.to_datetime(timestamps, utc=True)

    rename = {
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
        volume_col: "volume",
    }
    present = {source: target for source, target in rename.items() if source in frame.columns}
    frame = frame.rename(columns=present)
    frame.index = pd.DatetimeIndex(index)

    required = ["close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns after renaming: {missing}")

    keep = [column for column in ["open", "high", "low", "close", "volume"] if column in frame]
    frame = frame[keep].apply(pd.to_numeric, errors="coerce").sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def load_afml_bar_csv(
    csv_path: str | Path,
    *,
    timestamp_col: str = "ts_event",
    timestamp_ns_col: str = "ts_event_ns",
) -> pd.DataFrame:
    """
    Load an AFML bar CSV while preserving bar-generation metadata.

    The generated Binance AFML CSV files include OHLCV fields plus Chapter
    18/19-style microstructure columns such as ``buy_notional``,
    ``sell_notional``, ``signed_notional``, ``theta`` and threshold state. This
    loader keeps those numeric columns for feature engineering instead of
    reducing the file to plain OHLCV.
    """
    frame = pd.read_csv(csv_path)
    if timestamp_ns_col in frame.columns:
        index = pd.to_datetime(frame[timestamp_ns_col].astype("int64"), unit="ns", utc=True)
    elif timestamp_col in frame.columns:
        index = pd.to_datetime(frame[timestamp_col], utc=True)
    else:
        raise ValueError(
            f"CSV must contain {timestamp_ns_col!r} or {timestamp_col!r}",
        )

    skip = {timestamp_col, timestamp_ns_col, "timestamp", "bar_type"}
    keep = [column for column in frame.columns if column not in skip]
    out = frame[keep].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(index)
    out = out.sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def symmetric_cusum_filter(
    series: pd.Series,
    threshold: float | pd.Series,
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM event filter.

    ``series`` is usually ``np.log(close)``. ``threshold`` may be a scalar or a
    time-varying Series such as the EWMA daily volatility estimate.
    """
    series = series.dropna().astype(float).sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by a pandas DatetimeIndex")

    index = _as_utc_datetime_index(series.index)
    series = pd.Series(series.to_numpy(dtype=float), index=index)
    diff = series.diff().dropna()

    if isinstance(threshold, pd.Series):
        threshold_series = threshold.astype(float).sort_index()
        threshold_series.index = _as_utc_datetime_index(threshold_series.index)
        threshold_series = threshold_series.reindex(index).ffill()
    else:
        threshold_series = pd.Series(float(threshold), index=index)

    t_events: list[pd.Timestamp] = []
    s_pos = 0.0
    s_neg = 0.0
    for timestamp, value in diff.items():
        h_value = float(threshold_series.loc[timestamp])
        if not np.isfinite(value) or not np.isfinite(h_value) or h_value <= 0.0:
            continue
        s_pos = max(0.0, s_pos + value)
        s_neg = min(0.0, s_neg + value)
        if s_neg < -h_value:
            s_neg = 0.0
            t_events.append(timestamp)
        elif s_pos > h_value:
            s_pos = 0.0
            t_events.append(timestamp)
    return pd.DatetimeIndex(t_events, tz=index.tz)


# -------------------------------------------------------------------------------------------------
# Chapter 3 - Labeling and meta-labeling
# -------------------------------------------------------------------------------------------------


def get_daily_vol(close: pd.Series, span: int = 100, lookback_days: int = 1) -> pd.Series:
    """
    Estimate daily volatility using the AFML EWMA trick.

    For each intraday timestamp, the function finds the latest observation at or
    before ``lookback_days`` ago, computes that return, and applies
    ``Series.ewm(span=span).std()``.
    """
    close = close.dropna().astype(float).sort_index()
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a pandas DatetimeIndex")
    if span < 1:
        raise ValueError("span must be positive")

    index = _as_utc_datetime_index(close.index)
    close = pd.Series(close.to_numpy(dtype=float), index=index, name=close.name)
    search_positions = index.searchsorted(index - pd.Timedelta(days=lookback_days))
    valid = search_positions > 0
    if not valid.any():
        return pd.Series(dtype=float, index=index, name="daily_vol")

    current_index = index[valid]
    previous_index = index[search_positions[valid] - 1]
    returns = (
        close.loc[current_index].to_numpy(dtype=float)
        / close.loc[previous_index].to_numpy(dtype=float)
        - 1.0
    )
    daily_returns = pd.Series(returns, index=current_index)
    return daily_returns.ewm(span=span).std().rename("daily_vol")


def get_rolling_volatility(
    close: pd.Series,
    *,
    window: int = 100,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Estimate causal rolling log-return volatility for adaptive event sampling.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    close = close.dropna().astype(float).sort_index()
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a pandas DatetimeIndex")
    index = _as_utc_datetime_index(close.index)
    close = pd.Series(close.to_numpy(dtype=float), index=index, name=close.name)
    min_periods = min_periods if min_periods is not None else max(2, window // 2)
    log_return = np.log(close).diff()
    return log_return.rolling(window, min_periods=min_periods).std().rename("rolling_vol")


def get_horizon_log_return_volatility(
    close: pd.Series,
    *,
    horizon_bars: int = 80,
    window_bars: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Estimate causal rolling volatility of horizon log returns.

    This matches the synthetic barrier unit more closely than daily-vol targets:
    each observation is the rolling standard deviation of
    ``log(close_t / close_{t-horizon_bars})``.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    window_bars = int(window_bars if window_bars is not None else horizon_bars)
    if window_bars < 2:
        raise ValueError("window_bars must be at least 2")

    close = close.dropna().astype(float).sort_index()
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a pandas DatetimeIndex")
    index = _as_utc_datetime_index(close.index)
    close = pd.Series(close.to_numpy(dtype=float), index=index, name=close.name)
    min_periods = min_periods if min_periods is not None else window_bars
    horizon_log_return = np.log(close).diff(horizon_bars)
    return horizon_log_return.rolling(window_bars, min_periods=min_periods).std().rename(
        "horizon_log_return_vol",
    )


def get_ewma_1bar_log_return_volatility(
    close: pd.Series,
    *,
    span: int = 80,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Estimate causal EWMA volatility of one-bar log returns.
    """
    if span < 1:
        raise ValueError("span must be positive")

    close = close.dropna().astype(float).sort_index()
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a pandas DatetimeIndex")
    index = _as_utc_datetime_index(close.index)
    close = pd.Series(close.to_numpy(dtype=float), index=index, name=close.name)
    min_periods = min_periods if min_periods is not None else max(2, span // 2)
    log_return = np.log(close).diff()
    return log_return.ewm(span=span, adjust=False, min_periods=min_periods).std().rename(
        "ewma_1bar_log_return_vol",
    )


def _series_time_range(
    series: pd.Series,
    *,
    start: TimeLike = None,
    end: TimeLike = None,
) -> pd.Series:
    out = series.sort_index()
    start_bound = _time_bound(start)
    end_bound = _time_bound(end, is_end=True)
    if start_bound is not None:
        out = out.loc[out.index >= start_bound]
    if end_bound is not None:
        out = out.loc[out.index <= end_bound]
    return out


def _apply_train_volatility_floor(
    volatility: pd.Series,
    *,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    floor_quantile: float | None = 0.10,
) -> pd.Series:
    if floor_quantile is None:
        return volatility
    if not 0.0 <= floor_quantile <= 1.0:
        raise ValueError("floor_quantile must be in [0, 1]")
    train_values = _series_time_range(volatility.dropna(), start=train_start, end=train_end)
    if train_values.empty:
        return volatility
    floor = float(train_values.quantile(floor_quantile))
    if not np.isfinite(floor) or floor <= 0.0:
        return volatility
    return volatility.clip(lower=floor)


def make_rolling_cusum_threshold(
    close: pd.Series,
    *,
    window: int = 100,
    min_periods: int | None = None,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    floor_quantile: float | None = 0.10,
) -> pd.Series:
    """
    Build an adaptive CUSUM threshold from rolling volatility.

    The rolling volatility itself is causal. The optional floor is calibrated
    only from the supplied training window so the threshold does not collapse in
    unusually quiet periods without using test-period distribution information.
    """
    threshold = get_rolling_volatility(close, window=window, min_periods=min_periods)
    threshold = _apply_train_volatility_floor(
        threshold,
        train_start=train_start,
        train_end=train_end,
        floor_quantile=floor_quantile,
    )
    return threshold.rename("cusum_threshold")


def make_ewma_volatility_target(
    close: pd.Series,
    *,
    span: int = 100,
    lookback_days: int = 1,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    floor_quantile: float | None = 0.05,
) -> pd.Series:
    """
    Build the per-event EWMA volatility target used by triple barriers.

    The target remains time-varying per trade; the optional lower floor is
    calibrated from the training split only.
    """
    target = get_daily_vol(close, span=span, lookback_days=lookback_days)
    target = _apply_train_volatility_floor(
        target,
        train_start=train_start,
        train_end=train_end,
        floor_quantile=floor_quantile,
    )
    return target.rename("trgt")


def make_horizon_log_return_volatility_target(
    close: pd.Series,
    *,
    horizon_bars: int = 80,
    window_bars: int | None = None,
    min_periods: int | None = None,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    floor_quantile: float | None = 0.05,
) -> pd.Series:
    """
    Build the per-event horizon log-return volatility target used by barriers.
    """
    target = get_horizon_log_return_volatility(
        close,
        horizon_bars=horizon_bars,
        window_bars=window_bars,
        min_periods=min_periods,
    )
    target = _apply_train_volatility_floor(
        target,
        train_start=train_start,
        train_end=train_end,
        floor_quantile=floor_quantile,
    )
    return target.rename("trgt")


def make_ewma_1bar_log_return_volatility_target(
    close: pd.Series,
    *,
    span: int = 80,
    min_periods: int | None = None,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    floor_quantile: float | None = 0.05,
) -> pd.Series:
    """
    Build the unified one-bar EWMA log-return volatility target used by barriers.
    """
    target = get_ewma_1bar_log_return_volatility(
        close,
        span=span,
        min_periods=min_periods,
    )
    target = _apply_train_volatility_floor(
        target,
        train_start=train_start,
        train_end=train_end,
        floor_quantile=floor_quantile,
    )
    return target.rename("trgt")


def _path_returns(
    path: pd.Series,
    entry_price: float,
    *,
    side: float,
    return_type: str,
) -> pd.Series:
    if return_type == "simple":
        return (path / entry_price - 1.0) * side
    if return_type == "log":
        return np.log(path / entry_price) * side
    raise ValueError("return_type must be 'simple' or 'log'")


def _pt_sl_series(
    events: pd.DataFrame,
    pt_sl: tuple[float, float] | dict[str, float],
) -> tuple[pd.Series | None, pd.Series | None]:
    if isinstance(pt_sl, dict):
        base_pt = float(pt_sl.get("profit_taking_mult", 1.0))
        base_sl = float(pt_sl.get("stop_loss_mult", 1.0))
        long_pt = float(pt_sl.get("long_profit_taking_mult", base_pt))
        long_sl = float(pt_sl.get("long_stop_loss_mult", base_sl))
        short_pt = float(pt_sl.get("short_profit_taking_mult", base_pt))
        short_sl = float(pt_sl.get("short_stop_loss_mult", base_sl))
        values = (long_pt, long_sl, short_pt, short_sl)
        if any(value < 0.0 for value in values):
            raise ValueError("pt_sl values must be non-negative")
        side = events["side"].astype(float) if "side" in events else pd.Series(1.0, index=events.index)
        pt_mult = pd.Series(long_pt, index=events.index, dtype=float).where(side >= 0.0, short_pt)
        sl_mult = pd.Series(long_sl, index=events.index, dtype=float).where(side >= 0.0, short_sl)
    else:
        if pt_sl[0] < 0.0 or pt_sl[1] < 0.0:
            raise ValueError("pt_sl values must be non-negative")
        pt_mult = pd.Series(float(pt_sl[0]), index=events.index, dtype=float)
        sl_mult = pd.Series(float(pt_sl[1]), index=events.index, dtype=float)

    profit_taking = pt_mult * events["trgt"]
    stop_loss = -sl_mult * events["trgt"]
    return (
        profit_taking if bool((pt_mult > 0.0).any()) else None,
        stop_loss if bool((sl_mult > 0.0).any()) else None,
    )


def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    *,
    num_days: float | None = 1,
    num_bars: int | None = None,
) -> pd.Series:
    """
    Add vertical barriers by elapsed calendar time or by elapsed bars.
    """
    if num_days is None and num_bars is None:
        raise ValueError("Either num_days or num_bars must be provided")
    if num_bars is not None and num_bars < 1:
        raise ValueError("num_bars must be positive")

    close_index = _as_utc_datetime_index(close.dropna().sort_index().index)
    t_events = _as_utc_datetime_index(t_events)

    if num_bars is not None:
        positions = close_index.searchsorted(t_events)
        barrier_positions = positions + num_bars
    else:
        barrier_positions = close_index.searchsorted(t_events + pd.Timedelta(days=float(num_days)))

    values = [
        close_index[position] if position < len(close_index) else pd.NaT
        for position in barrier_positions
    ]
    return pd.Series(values, index=t_events, name="t1")


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] | dict[str, float] = (1.0, 1.0),
    *,
    return_type: str = "simple",
) -> pd.DataFrame:
    """
    Find the first profit-taking and stop-loss touches before each vertical barrier.
    """
    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    events = events.sort_index()

    out = pd.DataFrame(index=events.index)
    out["t1"] = events["t1"].astype("object")
    out["sl"] = pd.Series(pd.NaT, index=events.index, dtype="object")
    out["pt"] = pd.Series(pd.NaT, index=events.index, dtype="object")

    profit_taking, stop_loss = _pt_sl_series(events, pt_sl)

    for loc, vertical_barrier in events["t1"].fillna(close.index[-1]).items():
        if loc not in close.index:
            continue
        path = close.loc[loc:vertical_barrier]
        if path.empty:
            continue

        side = float(events.loc[loc, "side"]) if "side" in events else 1.0
        path_returns = _path_returns(
            path,
            float(close.loc[loc]),
            side=side,
            return_type=return_type,
        )
        if stop_loss is not None:
            touched = path_returns[path_returns < stop_loss.loc[loc]]
            if not touched.empty:
                out.loc[loc, "sl"] = touched.index[0]
        if profit_taking is not None:
            touched = path_returns[path_returns > profit_taking.loc[loc]]
            if not touched.empty:
                out.loc[loc, "pt"] = touched.index[0]

    return out


def get_triple_barrier_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float] | dict[str, float],
    target: pd.Series,
    *,
    min_ret: float = 0.0,
    t1: pd.Series | None = None,
    side: pd.Series | None = None,
    return_type: str = "simple",
) -> pd.DataFrame:
    """
    Create triple-barrier events and record each first touched barrier.
    """
    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    t_events = _as_utc_datetime_index(t_events)
    target = target.astype(float).sort_index()
    target.index = _as_utc_datetime_index(target.index)

    target = target.reindex(t_events).dropna()
    target = target[target > min_ret]
    if target.empty:
        return pd.DataFrame(
            columns=["t1", "trgt", "side", "vertical_barrier", "sl", "pt", "barrier"]
        )

    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events, name="t1")
    else:
        t1 = t1.reindex(t_events)

    if side is None:
        side_ = pd.Series(1.0, index=target.index, name="side")
    else:
        side = side.astype(float).sort_index()
        side.index = _as_utc_datetime_index(side.index)
        side_ = side.reindex(target.index).dropna().rename("side")
        target = target.reindex(side_.index)

    events = pd.concat(
        {
            "t1": t1.reindex(target.index),
            "trgt": target,
            "side": side_,
        },
        axis=1,
    ).dropna(subset=["trgt"])

    touches = apply_pt_sl_on_t1(
        close=close,
        events=events,
        pt_sl=pt_sl,
        return_type=return_type,
    )
    touch_times = touches[["sl", "pt", "t1"]]
    first_touch = []
    barrier = []
    for _, row in touch_times.iterrows():
        valid = row.dropna()
        if valid.empty:
            first_touch.append(pd.NaT)
            barrier.append(pd.NA)
            continue
        first = valid.min()
        first_touch.append(first)
        names = [name for name, value in valid.items() if value == first]
        barrier.append(names[0])

    events = events.rename(columns={"t1": "vertical_barrier"})
    events["t1"] = pd.Series(first_touch, index=events.index)
    events["sl"] = touches["sl"]
    events["pt"] = touches["pt"]
    events["barrier"] = pd.Series(barrier, index=events.index, dtype="string")
    return events[["t1", "trgt", "side", "vertical_barrier", "sl", "pt", "barrier"]]


def get_bins(
    events: pd.DataFrame,
    close: pd.Series,
    *,
    zero_on_vertical: bool = False,
    meta_label: bool = False,
    meta_label_min_ret: float = 0.0,
    return_type: str = "simple",
) -> pd.DataFrame:
    """
    Compute realized returns and labels from triple-barrier events.
    """
    if meta_label_min_ret < 0.0:
        raise ValueError("meta_label_min_ret cannot be negative")

    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    events = events.dropna(subset=["t1"]).copy()
    if events.empty:
        return pd.DataFrame(columns=["ret", "bin", "trgt", "barrier"])

    out = pd.DataFrame(index=events.index)
    aligned_start = close.reindex(events.index)
    aligned_end = close.reindex(pd.DatetimeIndex(events["t1"]))
    if return_type == "simple":
        returns = aligned_end.to_numpy(dtype=float) / aligned_start.to_numpy(dtype=float) - 1.0
    elif return_type == "log":
        returns = np.log(aligned_end.to_numpy(dtype=float) / aligned_start.to_numpy(dtype=float))
    else:
        raise ValueError("return_type must be 'simple' or 'log'")
    if "side" in events:
        returns *= events["side"].to_numpy(dtype=float)

    out["ret"] = returns
    out["trgt"] = events["trgt"]
    out["barrier"] = events["barrier"]

    if meta_label:
        out["bin"] = np.where(out["ret"] > meta_label_min_ret, 1, 0)
    else:
        out["bin"] = np.sign(out["ret"]).astype(int)
        if zero_on_vertical:
            out.loc[events["barrier"].eq("t1"), "bin"] = 0
    return out


def drop_rare_labels(labels: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    Recursively drop classes that are too rare while more than two classes remain.
    """
    if not 0.0 <= min_pct < 1.0:
        raise ValueError("min_pct must be in [0, 1)")

    labels = labels.copy()
    while not labels.empty:
        distribution = labels["bin"].value_counts(normalize=True)
        if distribution.empty or distribution.min() > min_pct or distribution.shape[0] < 3:
            break
        labels = labels[labels["bin"] != distribution.idxmin()]
    return labels


def drop_neutral_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ``bin == 0`` labels before fitting side models.
    """
    if "bin" not in labels:
        raise ValueError("labels must contain a 'bin' column")
    return labels[labels["bin"] != 0].copy()


def moving_average_crossover_side(
    close: pd.Series,
    *,
    fast_window: int = 10,
    slow_window: int = 20,
) -> pd.Series:
    """
    Deterministic side from fast/slow moving-average state.
    """
    if fast_window < 1 or slow_window < 1:
        raise ValueError("moving-average windows must be positive")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    close = close.dropna().astype(float).sort_index()
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a pandas DatetimeIndex")
    index = _as_utc_datetime_index(close.index)
    close = pd.Series(close.to_numpy(dtype=float), index=index, name=close.name)
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    diff = fast - slow
    side = pd.Series(0, index=index, dtype=int, name="side")
    side.loc[diff > 0.0] = 1
    side.loc[diff < 0.0] = -1
    return side


class MovingAverageCrossoverSideModel:
    """
    Deterministic primary side model based on fast/slow moving-average crosses.
    """

    requires_event_spans = False

    def __init__(
        self,
        close: pd.Series,
        *,
        fast_window: int = 10,
        slow_window: int = 20,
    ) -> None:
        if fast_window < 1 or slow_window < 1:
            raise ValueError("moving-average windows must be positive")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")

        close = close.dropna().astype(float).sort_index()
        close.index = _as_utc_datetime_index(close.index)
        self.close = close
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.side_ = moving_average_crossover_side(
            close,
            fast_window=fast_window,
            slow_window=slow_window,
        )
        self.classes_ = np.array([-1, 1], dtype=int)
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> MovingAverageCrossoverSideModel:
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        side = self.side_.reindex(_as_utc_datetime_index(X.index)).fillna(0).astype(int)
        proba = pd.DataFrame(0.5, index=X.index, columns=self.classes_)
        proba.loc[side < 0, -1] = 1.0
        proba.loc[side < 0, 1] = 0.0
        proba.loc[side > 0, -1] = 0.0
        proba.loc[side > 0, 1] = 1.0
        return proba

    def predict(self, X: pd.DataFrame) -> pd.Series:
        side = self.side_.reindex(_as_utc_datetime_index(X.index)).fillna(0).astype(int)
        return pd.Series(side.to_numpy(dtype=int), index=X.index, name="prediction")

# -------------------------------------------------------------------------------------------------
# Chapter 4 - Sample weights and sequential bootstrap
# -------------------------------------------------------------------------------------------------


def num_concurrent_events(close_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """
    Count how many event outcomes overlap each price bar.
    """
    close_index = _as_utc_datetime_index(close_index)
    t1 = t1.dropna().sort_index()
    if t1.empty:
        return pd.Series(0.0, index=close_index, name="num_concurrent")

    counts = np.zeros(len(close_index) + 1, dtype=float)
    for t_in, t_out in t1.items():
        start = close_index.searchsorted(t_in)
        end = close_index.searchsorted(t_out, side="right")
        if start >= len(close_index):
            continue
        counts[start] += 1.0
        counts[min(end, len(close_index))] -= 1.0
    return pd.Series(np.cumsum(counts[:-1]), index=close_index, name="num_concurrent")


def average_uniqueness(t1: pd.Series, num_co_events: pd.Series) -> pd.Series:
    """
    Compute each label's average uniqueness over its lifespan.
    """
    num_co_events = num_co_events.astype(float).sort_index()
    index = _as_utc_datetime_index(num_co_events.index)
    inv_concurrency = (1.0 / num_co_events.replace(0.0, np.nan)).fillna(0.0)
    cumsum = np.concatenate([[0.0], inv_concurrency.to_numpy(dtype=float).cumsum()])

    weights = {}
    for t_in, t_out in t1.dropna().sort_index().items():
        start = index.searchsorted(t_in)
        end = index.searchsorted(t_out, side="right")
        if end <= start:
            continue
        weights[t_in] = (cumsum[end] - cumsum[start]) / (end - start)
    return pd.Series(weights, name="tW").sort_index()


def sample_weights_by_return(
    t1: pd.Series,
    num_co_events: pd.Series,
    close: pd.Series,
    *,
    max_weight: float | None = DEFAULT_MAX_SAMPLE_WEIGHT,
) -> pd.Series:
    """
    Weight labels by absolute log-return attribution adjusted for concurrency.

    ``max_weight`` caps each normalized sample weight to prevent a small number
    of shock events from dominating tree fitting.
    """
    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    num_co_events = num_co_events.reindex(close.index).astype(float)
    adjusted_returns = np.log(close).diff().fillna(0.0) / num_co_events.replace(0.0, np.nan)
    adjusted_returns = adjusted_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cumsum = np.concatenate([[0.0], adjusted_returns.to_numpy(dtype=float).cumsum()])

    weights = {}
    for t_in, t_out in t1.dropna().sort_index().items():
        start = close.index.searchsorted(t_in)
        end = close.index.searchsorted(t_out, side="right")
        if end <= start:
            continue
        weights[t_in] = abs(cumsum[end] - cumsum[start])

    out = pd.Series(weights, name="w").sort_index()
    return _normalize_sample_weights(out, max_weight=max_weight).rename("w")


def _normalize_sample_weights(
    weights: pd.Series,
    *,
    max_weight: float | None = DEFAULT_MAX_SAMPLE_WEIGHT,
) -> pd.Series:
    weights = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0)
    if weights.empty or weights.sum() <= 0.0:
        return weights

    if max_weight is None:
        return weights * weights.shape[0] / weights.sum()
    if max_weight < 1.0:
        raise ValueError("max_weight must be at least 1.0")

    target_sum = float(weights.shape[0])
    out = pd.Series(0.0, index=weights.index, name=weights.name)
    capped = pd.Series(False, index=weights.index)
    remaining_sum = target_sum

    while remaining_sum > 0.0:
        available = ~capped
        active = available & (weights > 0.0)
        if not active.any():
            count = int(available.sum())
            if count > 0:
                out.loc[available] = remaining_sum / count
            break

        scaled = weights.loc[active] * remaining_sum / weights.loc[active].sum()
        over_cap = scaled > max_weight
        if not over_cap.any():
            out.loc[active] = scaled
            break

        over_index = scaled.index[over_cap]
        out.loc[over_index] = max_weight
        capped.loc[over_index] = True
        remaining_sum -= float(max_weight) * len(over_index)

    return out.rename(weights.name)


def time_decay_weights(tw: pd.Series, oldest_weight: float = 1.0) -> pd.Series:
    """
    Apply AFML piecewise-linear time decay over cumulative uniqueness.
    """
    tw = tw.dropna().sort_index().astype(float)
    if tw.empty:
        return tw.rename("decay")
    cumulative = tw.cumsum()
    total = cumulative.iloc[-1]
    if total <= 0.0:
        return pd.Series(1.0, index=tw.index, name="decay")

    if oldest_weight >= 0.0:
        slope = (1.0 - oldest_weight) / total
    else:
        slope = 1.0 / ((oldest_weight + 1.0) * total)
    const = 1.0 - slope * total
    decay = (const + slope * cumulative).clip(lower=0.0)
    return decay.rename("decay")


def get_indicator_matrix(
    bar_index: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.DataFrame:
    """
    Build the AFML Chapter 4 event indicator matrix.

    Rows are bars. Columns are feature observations. A cell is 1 when a bar is
    inside the label span ``[t0, t1]`` for that feature observation.
    """
    bar_index = _as_utc_datetime_index(bar_index)
    t1 = t1.dropna().sort_index()
    t1.index = _as_utc_datetime_index(t1.index)
    if t1.empty:
        return pd.DataFrame(index=bar_index)

    matrix = np.zeros((len(bar_index), len(t1)), dtype=np.int8)
    for column, (t_in, t_out) in enumerate(t1.items()):
        start = bar_index.searchsorted(t_in)
        end = bar_index.searchsorted(t_out, side="right")
        if start >= len(bar_index) or end <= start:
            continue
        matrix[start : min(end, len(bar_index)), column] = 1

    return pd.DataFrame(matrix, index=bar_index, columns=t1.index)


def _get_indicator_csc(bar_index: pd.DatetimeIndex, t1: pd.Series):
    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError("Sparse sequential bootstrap requires scipy") from exc

    bar_index = _as_utc_datetime_index(bar_index)
    t1 = t1.dropna().sort_index()
    t1.index = _as_utc_datetime_index(t1.index)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for column, (t_in, t_out) in enumerate(t1.items()):
        start = bar_index.searchsorted(t_in)
        end = min(bar_index.searchsorted(t_out, side="right"), len(bar_index))
        if start >= len(bar_index) or end <= start:
            continue
        active_rows = np.arange(start, end, dtype=np.int32)
        rows.append(active_rows)
        cols.append(np.full(active_rows.shape[0], column, dtype=np.int32))

    if rows:
        row_values = np.concatenate(rows)
        col_values = np.concatenate(cols)
        data = np.ones(row_values.shape[0], dtype=np.int8)
    else:
        row_values = np.array([], dtype=np.int32)
        col_values = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.int8)

    matrix = sparse.csc_matrix(
        (data, (row_values, col_values)),
        shape=(len(bar_index), len(t1)),
        dtype=np.int8,
    )
    return matrix, pd.Index(t1.index)


def average_uniqueness_from_indicator(indicator_matrix: pd.DataFrame) -> pd.Series:
    """
    Compute average uniqueness for each event from an indicator matrix.
    """
    if indicator_matrix.empty:
        return pd.Series(dtype=float, name="avg_uniqueness")

    concurrency = indicator_matrix.sum(axis=1).replace(0, np.nan)
    uniqueness = indicator_matrix.div(concurrency, axis=0)
    return uniqueness.where(indicator_matrix > 0).mean().rename("avg_uniqueness")


def _mean_average_uniqueness_from_indicator(indicator_matrix: Any) -> float:
    matrix, is_sparse = _coerce_indicator_matrix(indicator_matrix)
    _n_bars, n_events = matrix.shape
    if n_events == 0:
        return 0.0

    if is_sparse:
        concurrency = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
        inv_concurrency = np.divide(
            1.0,
            concurrency,
            out=np.zeros_like(concurrency, dtype=float),
            where=concurrency > 0.0,
        )
        event_counts = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
        avg_uniqueness = np.asarray(matrix.T @ inv_concurrency).ravel().astype(float)
    else:
        values = np.asarray(matrix, dtype=np.int8)
        concurrency = values.sum(axis=1).astype(float)
        inv_concurrency = np.divide(
            1.0,
            concurrency,
            out=np.zeros_like(concurrency, dtype=float),
            where=concurrency > 0.0,
        )
        event_counts = values.sum(axis=0).astype(float)
        avg_uniqueness = values.T @ inv_concurrency

    avg_uniqueness = np.divide(
        avg_uniqueness,
        event_counts,
        out=np.zeros_like(avg_uniqueness, dtype=float),
        where=event_counts > 0.0,
    )
    valid = event_counts > 0.0
    if not valid.any():
        return 0.0
    return float(avg_uniqueness[valid].mean())


def _coerce_indicator_matrix(indicator_matrix: Any):
    try:
        from scipy import sparse
    except ImportError:
        sparse = None

    if sparse is not None and sparse.issparse(indicator_matrix):
        return indicator_matrix.tocsc(copy=False), True
    if isinstance(indicator_matrix, pd.DataFrame):
        values = indicator_matrix.to_numpy(dtype=np.int8, copy=False)
        if sparse is not None:
            return sparse.csc_matrix(values), True
        return values, False
    return indicator_matrix, bool(sparse is not None and sparse.issparse(indicator_matrix))


def _sequential_bootstrap_sparse(
    matrix: Any, sample_length: int, rng: np.random.Generator
) -> np.ndarray:
    n_bars, n_events = matrix.shape
    selected = np.empty(sample_length, dtype=int)
    concurrency = np.zeros(n_bars, dtype=float)
    event_counts = np.asarray(matrix.sum(axis=0)).ravel()
    for draw in range(sample_length):
        inv_concurrency = 1.0 / (concurrency + 1.0)
        avg_uniqueness = np.asarray(matrix.T @ inv_concurrency).ravel()
        avg_uniqueness = np.divide(
            avg_uniqueness,
            event_counts,
            out=np.zeros_like(avg_uniqueness),
            where=event_counts > 0.0,
        )
        total = avg_uniqueness.sum()
        probabilities = None if total <= 0.0 else avg_uniqueness / total
        choice = int(rng.choice(n_events, p=probabilities))
        selected[draw] = choice
        start = matrix.indptr[choice]
        end = matrix.indptr[choice + 1]
        concurrency[matrix.indices[start:end]] += 1.0
    return selected


def _sequential_bootstrap_dense(
    matrix: Any, sample_length: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.int8)
    n_bars, n_events = values.shape
    selected = np.empty(sample_length, dtype=int)
    concurrency = np.zeros(n_bars, dtype=float)
    for draw in range(sample_length):
        avg_uniqueness = np.zeros(n_events, dtype=float)
        for event_pos in range(n_events):
            active = values[:, event_pos] > 0
            if active.any():
                avg_uniqueness[event_pos] = np.mean(1.0 / (concurrency[active] + 1.0))
        total = avg_uniqueness.sum()
        probabilities = None if total <= 0.0 else avg_uniqueness / total
        choice = int(rng.choice(n_events, p=probabilities))
        selected[draw] = choice
        concurrency += values[:, choice]
    return selected


def sequential_bootstrap_indices(
    indicator_matrix: Any,
    sample_length: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return integer column positions sampled by AFML sequential bootstrap.

    This implements Snippet 4.5 without repeatedly materializing reduced
    DataFrames. At every draw, candidate probabilities are proportional to the
    average uniqueness that would result from adding that candidate to the
    already selected bootstrap sample.
    """
    if isinstance(indicator_matrix, pd.DataFrame) and indicator_matrix.empty:
        return np.array([], dtype=int)

    matrix, is_sparse = _coerce_indicator_matrix(indicator_matrix)
    _n_bars, n_events = matrix.shape
    if n_events == 0:
        return np.array([], dtype=int)
    if sample_length is None:
        sample_length = n_events
    if sample_length < 1:
        raise ValueError("sample_length must be positive")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if is_sparse:
        return _sequential_bootstrap_sparse(matrix, sample_length, rng)
    return _sequential_bootstrap_dense(matrix, sample_length, rng)


def sequential_bootstrap(
    indicator_matrix: pd.DataFrame,
    sample_length: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.Index:
    """
    Return event labels sampled by AFML sequential bootstrap.
    """
    positions = sequential_bootstrap_indices(
        indicator_matrix,
        sample_length=sample_length,
        random_state=random_state,
    )
    return pd.Index(indicator_matrix.columns.take(positions))


# -------------------------------------------------------------------------------------------------
# Chapter 5 - Fractionally differentiated features
# -------------------------------------------------------------------------------------------------


def fracdiff_ffd_weights(
    d: float = DEFAULT_FEATURE_FRACDIFF_D,
    *,
    threshold: float = DEFAULT_FEATURE_FRACDIFF_THRESHOLD,
) -> np.ndarray:
    """
    Return fixed-width fractional differentiation weights.
    """
    if d < 0.0:
        raise ValueError("d must be non-negative")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1.0) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
        if k > 1_000_000:
            raise RuntimeError("FFD weight generation did not converge")
    return np.asarray(weights, dtype=float)


def fracdiff_ffd(
    series: pd.Series,
    d: float = DEFAULT_FEATURE_FRACDIFF_D,
    *,
    threshold: float = DEFAULT_FEATURE_FRACDIFF_THRESHOLD,
) -> pd.Series:
    """
    Fixed-width fractional differentiation of a pandas Series.
    """
    series = series.dropna().astype(float).sort_index()
    weights = fracdiff_ffd_weights(d, threshold=threshold)
    width = weights.shape[0] - 1
    if series.shape[0] <= width:
        return pd.Series(dtype=float, index=series.index[:0], name=series.name)

    values = series.to_numpy(dtype=float)
    output = np.zeros(values.shape[0] - width, dtype=float)
    for offset, weight in enumerate(weights):
        output += weight * values[width - offset : values.shape[0] - offset]
    return pd.Series(output, index=series.index[width:], name=series.name)


def _fracdiff_column_prefix(d: float) -> str:
    value = f"{d:.2f}".replace("-", "m").replace(".", "_")
    return f"ffd_d_{value}"


def make_default_features(
    close: pd.Series,
    volatility: pd.Series | None = None,
    *,
    fast_span: int = 8,
    slow_span: int = 32,
    fracdiff_d: float = DEFAULT_FEATURE_FRACDIFF_D,
    fracdiff_threshold: float = DEFAULT_FEATURE_FRACDIFF_THRESHOLD,
) -> pd.DataFrame:
    """
    Build a compact, leak-free feature set from close prices.
    """
    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    log_close = np.log(close)
    fracdiff = fracdiff_ffd(log_close, d=fracdiff_d, threshold=fracdiff_threshold)
    fracdiff = fracdiff.reindex(close.index)
    fracdiff_prefix = _fracdiff_column_prefix(fracdiff_d)

    features = pd.DataFrame(index=close.index)
    for lag in (1, 2, 4, 8, 16):
        features[f"{fracdiff_prefix}_lag_{lag}"] = fracdiff.shift(lag - 1)

    ewm_fast = fracdiff.ewm(span=fast_span).mean()
    ewm_slow = fracdiff.ewm(span=slow_span).mean()
    ewm_vol = fracdiff.ewm(span=slow_span).std()
    rolling_mean = log_close.rolling(slow_span).mean()
    rolling_std = log_close.rolling(slow_span).std()

    features[f"{fracdiff_prefix}_ewm_fast"] = ewm_fast
    features[f"{fracdiff_prefix}_ewm_slow"] = ewm_slow
    features[f"{fracdiff_prefix}_ewm_diff"] = ewm_fast - ewm_slow
    features[f"{fracdiff_prefix}_ewm_vol"] = ewm_vol
    features["zscore_slow"] = (log_close - rolling_mean) / rolling_std
    features[f"{fracdiff_prefix}_autocorr"] = fracdiff.rolling(slow_span).corr(
        fracdiff.shift(1),
    )

    if volatility is not None:
        vol = volatility.astype(float).sort_index()
        vol.index = _as_utc_datetime_index(vol.index)
        vol = vol.reindex(close.index).ffill()
        features["daily_vol"] = vol
        features["daily_vol_chg"] = vol.pct_change().replace([np.inf, -np.inf], np.nan)

    return features.replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------------------------------------------------------
# Chapter 6 - Ensemble methods
# -------------------------------------------------------------------------------------------------


def _decision_tree_classifier_cls() -> Any:
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise ImportError(
            "SequentialBootstrapBaggingClassifier requires scikit-learn. "
            "Install it with `uv pip install --python .\\.venv\\Scripts\\python.exe scikit-learn scipy`.",
        ) from exc
    return DecisionTreeClassifier


def _fit_single_decision_tree_estimator(
    estimator_args: tuple[int, int],
    *,
    decision_tree_classifier: Any,
    indicator_matrix: Any,
    sample_length: int,
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray | None,
    max_features: int | str | None,
    max_depth: int | None,
    min_samples_leaf: int,
    min_weight_fraction_leaf: float,
    class_weight: str | dict[int, float] | None,
) -> tuple[int, Any, np.ndarray, np.ndarray]:
    estimator_num, seed = estimator_args
    sampled = sequential_bootstrap_indices(
        indicator_matrix,
        sample_length=sample_length,
        random_state=seed,
    )
    tree_class_weight = "balanced" if class_weight == "balanced_subsample" else class_weight
    estimator = decision_tree_classifier(
        criterion="entropy",
        max_features=max_features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        class_weight=tree_class_weight,
        random_state=seed,
    )
    fit_weight = weights[sampled] if weights is not None else None
    estimator.fit(x_values[sampled], y_values[sampled], sample_weight=fit_weight)
    return estimator_num, estimator, estimator.classes_.copy(), sampled


class SequentialBootstrapBaggingClassifier(ClassifierMixin, BaseEstimator):
    """
    Bagged decision trees trained with AFML sequential bootstrap samples.

    This follows the AFML Chapter 6 bagging setup while replacing IID bootstrap
    sampling with Chapter 4 sequential bootstrap over triple-barrier event
    spans.
    """

    requires_event_spans = True

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_samples: float | str | None = "avg_uniqueness",
        max_features: float | str | None = 1,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int | None = None,
        n_jobs: int | None = 1,
        verbose: bool = False,
        progress_interval: int = 10,
        progress_label: str | None = None,
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

    def _validate_hyperparameters(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if isinstance(self.max_samples, int) and self.max_samples < 1:
            raise ValueError("integer max_samples must be positive")
        if isinstance(self.max_samples, float) and not 0.0 < self.max_samples <= 1.0:
            raise ValueError("float max_samples must be in (0, 1]")
        if isinstance(self.max_samples, str) and self.max_samples not in {
            "avg_uniqueness",
            "avgU",
        }:
            raise ValueError("string max_samples must be 'avg_uniqueness' or 'avgU'")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        if not 0.0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must be in [0, 0.5]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be positive")

    def _sample_length(self, n_samples: int, indicator_matrix: Any | None = None) -> int:
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, str):
            if indicator_matrix is None:
                raise ValueError("indicator_matrix is required when max_samples='avg_uniqueness'")
            avg_uniqueness = _mean_average_uniqueness_from_indicator(indicator_matrix)
            return max(1, int(np.ceil(avg_uniqueness * n_samples)))
        if isinstance(self.max_samples, float):
            return max(1, int(np.ceil(self.max_samples * n_samples)))
        return int(self.max_samples)

    def _effective_n_jobs(self) -> int:
        if self.n_jobs is None:
            return 1
        if self.n_jobs < 0:
            cpu_count = os.cpu_count() or 1
            return max(1, cpu_count + 1 + self.n_jobs)
        return self.n_jobs

    def _fit_estimator_jobs(
        self,
        jobs: list[tuple[int, int]],
        *,
        decision_tree_classifier: Any,
        indicator_matrix: Any,
        sample_length: int,
        x_values: np.ndarray,
        y_values: np.ndarray,
        weights: np.ndarray | None,
    ):
        fit_kwargs = {
            "decision_tree_classifier": decision_tree_classifier,
            "indicator_matrix": indicator_matrix,
            "sample_length": sample_length,
            "x_values": x_values,
            "y_values": y_values,
            "weights": weights,
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "class_weight": self.class_weight,
        }
        n_jobs = min(self._effective_n_jobs(), self.n_estimators)
        if n_jobs == 1:
            for job in jobs:
                yield _fit_single_decision_tree_estimator(job, **fit_kwargs)
            return
        try:
            from joblib import Parallel
            from joblib import delayed
            from joblib import parallel_backend
        except ImportError as exc:
            raise ImportError("Parallel fitting requires joblib") from exc

        with parallel_backend("loky"):
            fitted = Parallel(n_jobs=n_jobs)(
                delayed(_fit_single_decision_tree_estimator)(job, **fit_kwargs) for job in jobs
            )
        yield from fitted

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        t1: pd.Series | None = None,
        bar_index: pd.DatetimeIndex | None = None,
        indicator_matrix: pd.DataFrame | None = None,
    ) -> SequentialBootstrapBaggingClassifier:
        """
        Fit bagged trees on sequential-bootstrap samples.
        """
        self._validate_hyperparameters()
        decision_tree_classifier = _decision_tree_classifier_cls()

        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.reindex(X.index).dropna().astype(int)
        X = X.loc[y.index]
        if X.empty:
            raise ValueError("Cannot fit model on an empty feature matrix")

        classes = np.array(sorted(y.unique()), dtype=int)
        if classes.shape[0] < 2:
            raise ValueError("At least two classes are required to fit a classifier")

        event_columns: pd.Index
        if indicator_matrix is None:
            if t1 is None or bar_index is None:
                raise ValueError(
                    "t1 and bar_index are required when indicator_matrix is not provided"
                )
            indicator_matrix, event_columns = _get_indicator_csc(bar_index, t1.reindex(X.index))
        else:
            if isinstance(indicator_matrix, pd.DataFrame):
                indicator_matrix = (
                    indicator_matrix.reindex(columns=X.index).fillna(0).astype(np.int8)
                )
                event_columns = pd.Index(indicator_matrix.columns)
            else:
                event_columns = pd.Index(X.index)

        if indicator_matrix.shape[1] != X.shape[0]:
            raise ValueError("indicator_matrix columns must align to X rows")
        if not event_columns.equals(pd.Index(X.index)):
            raise ValueError("indicator_matrix columns must align to X rows")

        x_values = X.to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=int)
        weights = (
            sample_weight.reindex(X.index).fillna(0.0).to_numpy(dtype=float)
            if sample_weight is not None
            else None
        )

        rng = np.random.default_rng(self.random_state)
        sample_length = self._sample_length(X.shape[0], indicator_matrix)
        self.estimators_ = []
        self.estimator_classes_ = []
        self.bootstrap_indices_ = []
        seeds = [int(rng.integers(0, np.iinfo(np.int32).max)) for _ in range(self.n_estimators)]
        jobs = list(enumerate(seeds, start=1))
        fitted = self._fit_estimator_jobs(
            jobs,
            decision_tree_classifier=decision_tree_classifier,
            indicator_matrix=indicator_matrix,
            sample_length=sample_length,
            x_values=x_values,
            y_values=y_values,
            weights=weights,
        )
        for estimator_num, estimator, estimator_classes, sampled in fitted:
            self.estimators_.append(estimator)
            self.estimator_classes_.append(estimator_classes)
            self.bootstrap_indices_.append(sampled)
            if self.verbose and (
                estimator_num == 1
                or estimator_num == self.n_estimators
                or estimator_num % self.progress_interval == 0
            ):
                label = f"{self.progress_label}: " if self.progress_label else ""
                print(
                    f"  {label}trained bagging estimator {estimator_num}/{self.n_estimators}",
                    flush=True,
                )

        self.classes_ = classes
        self.feature_names_ = list(X.columns)
        self.indicator_matrix_ = indicator_matrix
        return self

    def _check_fitted(self) -> None:
        if (
            getattr(self, "classes_", None) is None
            or getattr(self, "feature_names_", None) is None
            or not getattr(self, "estimators_", None)
        ):
            raise ValueError("Model is not fitted")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        assert self.classes_ is not None
        assert self.feature_names_ is not None

        X = X.reindex(columns=self.feature_names_)
        x_values = X.to_numpy(dtype=float)
        proba = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=float)
        class_to_pos = {label: pos for pos, label in enumerate(self.classes_)}
        for estimator, estimator_classes in zip(
            self.estimators_,
            self.estimator_classes_,
            strict=True,
        ):
            estimator_proba = estimator.predict_proba(x_values)
            for local_pos, label in enumerate(estimator_classes):
                proba[:, class_to_pos[int(label)]] += estimator_proba[:, local_pos]

        proba /= len(self.estimators_)
        return pd.DataFrame(proba, index=X.index, columns=self.classes_)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        return pd.Series(proba.idxmax(axis=1).astype(int), index=X.index, name="prediction")

    def save_joblib(self, path: str | Path) -> Path:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Saving this model requires joblib") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load_joblib(cls, path: str | Path) -> SequentialBootstrapBaggingClassifier:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Loading this model requires joblib") from exc

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model


# -------------------------------------------------------------------------------------------------
# Chapter 8 - Feature importance
# -------------------------------------------------------------------------------------------------


def _weighted_accuracy(
    truth: pd.Series,
    prediction: pd.Series,
    sample_weight: pd.Series | None = None,
) -> float:
    truth = truth.astype(int)
    prediction = prediction.reindex(truth.index).astype(int)
    correct = (prediction == truth).astype(float)
    if sample_weight is None:
        return float(correct.mean())
    weights = sample_weight.reindex(truth.index).fillna(0.0).astype(float)
    if weights.sum() <= 0.0:
        return float(correct.mean())
    return float(np.average(correct.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def mdi_feature_importance(model: Any) -> pd.DataFrame:
    """
    Compute mean decrease impurity feature importance from fitted tree estimators.
    """
    feature_names = getattr(model, "feature_names_", None)
    if feature_names is None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None and hasattr(model, "feature_importances_"):
            n_features = len(model.feature_importances_)
        if n_features is None:
            raise ValueError("Cannot infer feature names from model")
        feature_names = [f"feature_{i}" for i in range(int(n_features))]

    rows: list[np.ndarray] = []
    estimators = getattr(model, "estimators_", None)
    if estimators:
        for estimator in estimators:
            if hasattr(estimator, "feature_importances_"):
                rows.append(np.asarray(estimator.feature_importances_, dtype=float))
    elif hasattr(model, "feature_importances_"):
        rows.append(np.asarray(model.feature_importances_, dtype=float))

    if not rows:
        raise ValueError("Model does not expose tree impurity importances")

    matrix = np.vstack(rows)
    importance = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    total = importance.sum()
    if total > 0.0:
        importance = importance / total

    return pd.DataFrame(
        {
            "method": "MDI",
            "feature": list(feature_names),
            "importance": importance,
            "std": std,
        },
    ).sort_values("importance", ascending=False, ignore_index=True)


def mda_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: pd.Series | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Compute mean decrease accuracy by permuting each feature out-of-sample.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.reindex(X.index).dropna().astype(int)
    X = X.loc[y.index]
    if sample_weight is not None:
        sample_weight = sample_weight.reindex(X.index)
    if X.empty:
        raise ValueError("Cannot compute MDA on an empty feature matrix")

    rng = np.random.default_rng(random_state)
    baseline = _weighted_accuracy(y, model.predict(X), sample_weight)
    rows: list[dict[str, float | str]] = []
    for feature in X.columns:
        decreases = []
        values = X[feature].to_numpy(dtype=float)
        for _ in range(n_repeats):
            permuted = X.copy()
            permuted[feature] = rng.permutation(values)
            score = _weighted_accuracy(y, model.predict(permuted), sample_weight)
            decreases.append(baseline - score)
        rows.append(
            {
                "method": "MDA",
                "feature": str(feature),
                "importance": float(np.mean(decreases)),
                "std": float(np.std(decreases, ddof=0)),
                "baseline_score": baseline,
            },
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


def sfi_feature_importance(
    dataset: AfmlDataset,
    *,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    model_factory: Any,
) -> pd.DataFrame:
    """
    Compute single feature importance by fitting one model per feature.
    """
    rows: list[dict[str, float | str]] = []
    for feature in dataset.X.columns:
        feature_dataset = AfmlDataset(
            close=dataset.close,
            features=dataset.features[[feature]],
            events=dataset.events,
            labels=dataset.labels,
            sample_weight=dataset.sample_weight,
            volatility=dataset.volatility,
            t_events=dataset.t_events,
        )
        model = model_factory()
        _fit_classifier(model, feature_dataset, train_index)
        score = _weighted_accuracy(
            feature_dataset.y.loc[test_index],
            model.predict(feature_dataset.X.loc[test_index]),
            feature_dataset.sample_weight.loc[test_index],
        )
        rows.append(
            {
                "method": "SFI",
                "feature": str(feature),
                "importance": score,
                "std": 0.0,
            },
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


# -------------------------------------------------------------------------------------------------
# Chapter 10 - Bet sizing
# -------------------------------------------------------------------------------------------------


def _normal_cdf(value: float | np.ndarray) -> float | np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(np.asarray(value, dtype=float) / math.sqrt(2.0)))


def bet_size_from_probability(
    probability: pd.Series | np.ndarray | float,
    *,
    num_classes: int = 2,
) -> pd.Series | np.ndarray | float:
    """
    Convert classifier probability to Chapter-10 style bet-size magnitude.

    The returned value is unsigned and clipped to ``[0, 1]``. Apply the
    predicted side separately to get a signed target.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    is_series = isinstance(probability, pd.Series)
    index = probability.index if is_series else None
    values = np.asarray(probability, dtype=float)
    values = np.clip(values, 1e-12, 1.0 - 1e-12)
    benchmark = 1.0 / float(num_classes)
    z = (values - benchmark) / np.sqrt(values * (1.0 - values))
    magnitude = np.clip(2.0 * _normal_cdf(z) - 1.0, 0.0, 1.0)
    if is_series:
        return pd.Series(magnitude, index=index, name="bet_size_magnitude")
    if np.ndim(probability) == 0:
        return float(magnitude)
    return magnitude


def make_bet_size_frame(
    signals: pd.DataFrame,
    *,
    probability_col: str | None = None,
    signal_col: str = "signal",
    num_classes: int = 2,
    min_abs_size: float = 0.0,
    max_abs_size: float = 1.0,
    step_size: float | None = None,
) -> pd.DataFrame:
    """
    Add signed Chapter-10 bet sizes to a signal frame.
    """
    if signal_col not in signals.columns:
        raise ValueError(f"signals must contain {signal_col!r}")
    if not 0.0 <= min_abs_size <= max_abs_size <= 1.0:
        raise ValueError("size bounds must satisfy 0 <= min_abs_size <= max_abs_size <= 1")
    if step_size is not None and not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must satisfy 0 < step_size <= 1")

    out = signals.copy()
    if probability_col is None:
        for candidate in ("meta_probability", "confidence"):
            if candidate in out.columns:
                probability_col = candidate
                break
    if probability_col is None or probability_col not in out.columns:
        probability = pd.Series(1.0, index=out.index)
    else:
        probability = out[probability_col].astype(float)

    magnitude = bet_size_from_probability(probability, num_classes=num_classes)
    assert isinstance(magnitude, pd.Series)
    magnitude = magnitude.clip(lower=min_abs_size, upper=max_abs_size)
    active = out[signal_col].astype(int) != 0
    magnitude = magnitude.where(active, 0.0)
    bet_size = out[signal_col].astype(int) * magnitude
    if step_size is not None:
        bet_size = (bet_size / step_size).round() * step_size
        bet_size = bet_size.clip(lower=-max_abs_size, upper=max_abs_size)
        bet_size = bet_size.where(active, 0.0)
    out["bet_size"] = bet_size
    out["bet_size_abs"] = bet_size.abs()
    return out


# -------------------------------------------------------------------------------------------------
# Chapter 14 - Backtest statistics
# -------------------------------------------------------------------------------------------------


def annualized_sharpe_ratio(returns: pd.Series, periods_per_year: float = 365.0) -> float:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(math.sqrt(periods_per_year) * returns.mean() / std)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sr: float = 0.0,
    periods_per_year: float = 365.0,
) -> float:
    """
    Probability that the strategy Sharpe ratio exceeds ``benchmark_sr``.
    """
    returns = returns.dropna().astype(float)
    n = len(returns)
    if n < 3:
        return 0.0
    sr = annualized_sharpe_ratio(returns, periods_per_year=periods_per_year)
    skew = float(returns.skew())
    kurt = float(returns.kurt() + 3.0)
    denominator = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = (sr - benchmark_sr) * math.sqrt(n - 1.0) / denominator
    return float(_normal_cdf(z))


# -------------------------------------------------------------------------------------------------
# Chapter 17 - Structural breaks
# -------------------------------------------------------------------------------------------------


def rolling_adf_t_stat(series: pd.Series, window: int = 64) -> pd.Series:
    """
    Compute a rolling right-edge ADF t-statistic for ``diff(y_t) ~ 1 + y_{t-1}``.
    """
    if window < 4:
        raise ValueError("window must be at least 4")

    y = series.dropna().astype(float).sort_index()
    y.index = _as_utc_datetime_index(y.index)
    dy = y.diff()
    x = y.shift(1)
    n = x.rolling(window, min_periods=window).count()
    sx = x.rolling(window, min_periods=window).sum()
    sy = dy.rolling(window, min_periods=window).sum()
    sxx = (x * x).rolling(window, min_periods=window).sum()
    syy = (dy * dy).rolling(window, min_periods=window).sum()
    sxy = (x * dy).rolling(window, min_periods=window).sum()

    centered_xx = sxx - sx * sx / n
    centered_xy = sxy - sx * sy / n
    beta = centered_xy / centered_xx
    alpha = (sy - beta * sx) / n
    sse = syy + n * alpha * alpha + beta * beta * sxx - 2.0 * alpha * sy - 2.0 * beta * sxy
    sse += 2.0 * alpha * beta * sx
    sigma2 = sse / (n - 2.0)
    se_beta = np.sqrt(sigma2 / centered_xx)
    t_stat = beta / se_beta
    t_stat = t_stat.where((n > 2.0) & (centered_xx > 0.0) & (se_beta > 0.0))
    return t_stat.rename(f"adf_t_{window}")


def sadf_feature(
    close: pd.Series,
    *,
    windows: tuple[int, ...] = (32, 64, 128),
) -> pd.Series:
    """
    Causal SADF-style feature as the supremum over rolling ADF t-stat windows.

    This is a practical feature-engineering approximation of the Chapter 17
    right-tail SADF idea: each timestamp only uses windows ending at that
    timestamp, then keeps the largest ADF t-statistic across candidate window
    lengths.
    """
    if not windows:
        raise ValueError("windows cannot be empty")
    log_close = np.log(close.dropna().astype(float).sort_index())
    log_close.index = _as_utc_datetime_index(log_close.index)
    stats = [rolling_adf_t_stat(log_close, window=window) for window in windows]
    return pd.concat(stats, axis=1).max(axis=1).rename("sadf")


# -------------------------------------------------------------------------------------------------
# Chapter 18 - Entropy features
# -------------------------------------------------------------------------------------------------


def rolling_binary_entropy(series: pd.Series, window: int = 64) -> pd.Series:
    """
    Causal rolling Shannon entropy over the sign of a series.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    binary = (series.astype(float) > 0.0).astype(float)
    p = binary.rolling(window, min_periods=max(2, window // 2)).mean()
    q = 1.0 - p
    p_term = pd.Series(0.0, index=series.index)
    q_term = pd.Series(0.0, index=series.index)
    p_mask = p > 0.0
    q_mask = q > 0.0
    p_term.loc[p_mask] = p.loc[p_mask] * np.log2(p.loc[p_mask])
    q_term.loc[q_mask] = q.loc[q_mask] * np.log2(q.loc[q_mask])
    entropy = -(p_term + q_term)
    return entropy.replace([np.inf, -np.inf], np.nan)


def lempel_ziv_complexity(sequence: str) -> int:
    """
    Compute Lempel-Ziv 1976 complexity for a finite symbol sequence.
    """
    sequence = "".join(sequence)
    n = len(sequence)
    if n == 0:
        return 0

    dictionary: set[str] = set()
    word = ""
    complexity = 0
    for symbol in sequence:
        candidate = word + symbol
        if candidate in dictionary:
            word = candidate
            continue
        dictionary.add(candidate)
        complexity += 1
        word = ""
    if word:
        complexity += 1
    return complexity


def rolling_lz_complexity(series: pd.Series, window: int = 64) -> pd.Series:
    """
    Causal rolling normalized Lempel-Ziv complexity over return signs.
    """
    if window < 2:
        raise ValueError("window must be at least 2")

    signs = np.where(series.astype(float).to_numpy() > 0.0, "1", "0")
    values = np.full(len(signs), np.nan, dtype=float)
    min_periods = max(2, window // 2)
    for end in range(min_periods, len(signs) + 1):
        start = max(0, end - window)
        chunk = "".join(signs[start:end])
        values[end - 1] = lempel_ziv_complexity(chunk) / len(chunk)
    return pd.Series(values, index=series.index, name="lz_complexity")


# -------------------------------------------------------------------------------------------------
# Chapter 19 - Microstructural features
# -------------------------------------------------------------------------------------------------


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0.0, np.nan)
    return numerator / denominator


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return _safe_divide(series - mean, std)


def _rolling_iqr_zscore(
    series: pd.Series,
    window: int | str,
    *,
    min_periods: int,
    clip: float | None,
) -> pd.Series:
    rolling = series.astype(float).rolling(window=window, min_periods=min_periods)
    median = rolling.median()
    iqr = rolling.quantile(0.75) - rolling.quantile(0.25)
    zscore = _safe_divide(series.astype(float) - median, iqr / 1.349)
    if clip is not None:
        zscore = zscore.clip(lower=-float(clip), upper=float(clip))
    return zscore


def _rolling_ols_slope(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    min_periods = max(3, window // 2)
    x = x.astype(float)
    y = y.astype(float).reindex(x.index)
    cov = x.rolling(window, min_periods=min_periods).cov(y)
    var = x.rolling(window, min_periods=min_periods).var()
    return _safe_divide(cov, var)


def _first_numeric_column(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    index: pd.DatetimeIndex,
) -> pd.Series | None:
    for name in names:
        if name in frame.columns:
            return frame[name].astype(float).reindex(index)
    return None


def _calendar_window_label(window: str) -> str:
    days = pd.Timedelta(window).total_seconds() / 86_400.0
    if math.isclose(days, 1.0):
        return "1d"
    if math.isclose(days, 3.0):
        return "3d"
    if math.isclose(days, 7.0):
        return "1w"
    if math.isclose(days, 14.0):
        return "2w"
    if math.isclose(days, 30.0):
        return "1m"
    if days.is_integer():
        return f"{int(days)}d"
    return f"{days:g}d".replace(".", "p")


def _calendar_window_days(window: str) -> float:
    days = pd.Timedelta(window).total_seconds() / 86_400.0
    if days <= 0.0:
        raise ValueError("calendar windows must be positive")
    return days


def _add_calendar_regime_features(
    features: pd.DataFrame,
    *,
    close: pd.Series,
    log_return: pd.Series,
    total_notional: pd.Series,
    signed_notional: pd.Series,
    theta_to_threshold: pd.Series | None,
    calendar_windows: tuple[str, ...],
    min_periods: int,
    tail_quantile: float,
    robust_z_clip: float | None,
) -> None:
    abs_return = log_return.abs()
    signed_abs_notional = signed_notional.abs()

    for window in calendar_windows:
        label = _calendar_window_label(window)
        window_days = _calendar_window_days(window)
        rolling_return = log_return.rolling(window=window, min_periods=min_periods)
        rolling_abs_return = abs_return.rolling(window=window, min_periods=min_periods)
        rolling_notional = total_notional.rolling(window=window, min_periods=min_periods)
        rolling_signed_abs_notional = signed_abs_notional.rolling(
            window=window,
            min_periods=min_periods,
        )

        bar_count = close.rolling(window=window, min_periods=1).count()
        return_skew = rolling_return.skew()
        return_excess_kurt = rolling_return.kurt()
        jb_moment_distance = np.sqrt(return_skew.pow(2) + return_excess_kurt.pow(2) / 4.0)
        abs_return_tail = rolling_abs_return.quantile(tail_quantile)
        tail_event = (abs_return >= abs_return_tail).astype(float)
        tail_event_count = tail_event.rolling(window=window, min_periods=min_periods).sum()

        features[f"calendar_bar_count_{label}"] = bar_count
        features[f"calendar_bar_density_per_day_{label}"] = bar_count / window_days
        features[f"calendar_realized_vol_{label}"] = rolling_return.std()
        features[f"calendar_return_skew_{label}"] = return_skew
        features[f"calendar_return_excess_kurt_{label}"] = return_excess_kurt
        features[f"calendar_jb_moment_distance_{label}"] = jb_moment_distance
        features[f"calendar_abs_return_max_{label}"] = rolling_abs_return.max()
        features[f"calendar_abs_return_q{int(tail_quantile * 100):02d}_{label}"] = (
            abs_return_tail
        )
        features[f"calendar_tail_event_count_{label}"] = tail_event_count
        features[f"calendar_tail_event_share_{label}"] = _safe_divide(
            tail_event_count,
            bar_count,
        )
        features[f"calendar_log_return_robust_z_{label}"] = _rolling_iqr_zscore(
            log_return,
            window,
            min_periods=min_periods,
            clip=robust_z_clip,
        )
        features[f"calendar_abs_return_robust_z_{label}"] = _rolling_iqr_zscore(
            abs_return,
            window,
            min_periods=min_periods,
            clip=robust_z_clip,
        )
        features[f"calendar_dollar_volume_sum_{label}"] = rolling_notional.sum()
        features[f"calendar_dollar_volume_robust_z_{label}"] = _rolling_iqr_zscore(
            total_notional,
            window,
            min_periods=min_periods,
            clip=robust_z_clip,
        )
        features[f"calendar_signed_notional_robust_z_{label}"] = _rolling_iqr_zscore(
            signed_notional,
            window,
            min_periods=min_periods,
            clip=robust_z_clip,
        )
        features[f"calendar_order_flow_imbalance_{label}"] = _safe_divide(
            signed_notional.rolling(window=window, min_periods=min_periods).sum(),
            rolling_notional.sum(),
        )
        features[f"calendar_vpin_{label}"] = _safe_divide(
            rolling_signed_abs_notional.sum(),
            rolling_notional.sum(),
        )

        if theta_to_threshold is not None:
            features[f"calendar_theta_to_threshold_mean_{label}"] = theta_to_threshold.rolling(
                window=window,
                min_periods=min_periods,
            ).mean()
            features[f"calendar_theta_to_threshold_max_{label}"] = theta_to_threshold.rolling(
                window=window,
                min_periods=min_periods,
            ).max()
            features[f"calendar_theta_to_threshold_robust_z_{label}"] = _rolling_iqr_zscore(
                theta_to_threshold,
                window,
                min_periods=min_periods,
                clip=robust_z_clip,
            )


def make_pipeline_features(  # noqa: C901
    frame: pd.DataFrame,
    volatility: pd.Series | None = None,
    *,
    fast_span: int = 8,
    slow_span: int = 32,
    microstructure_window: int = 50,
    information_window: int = 64,
    calendar_windows: tuple[str, ...] | None = ("1D", "3D", "7D", "14D", "30D"),
    calendar_min_periods: int = 8,
    tail_quantile: float = 0.99,
    robust_z_clip: float | None = 5.0,
    sadf_windows: tuple[int, ...] = (32, 64, 128),
    fracdiff_d: float = DEFAULT_FEATURE_FRACDIFF_D,
    fracdiff_threshold: float = DEFAULT_FEATURE_FRACDIFF_THRESHOLD,
) -> pd.DataFrame:
    """
    Build the Stage-2 feature matrix from OHLCV plus AFML bar metadata.

    The output intentionally groups the AFML pipeline's feature families:
    fixed-width fractional differentiation, Chapter 17 structural-break
    features, Chapter 18 entropy features, and Chapter 19 microstructure /
    price-impact features. Calendar-window regime features keep the input in
    DRB/event time while describing volatility, tail, and bar-density state over
    elapsed clock time.
    """
    if "close" not in frame.columns:
        raise ValueError("frame must contain a 'close' column")
    if microstructure_window < 2:
        raise ValueError("microstructure_window must be at least 2")
    if information_window < 2:
        raise ValueError("information_window must be at least 2")
    if calendar_min_periods < 2:
        raise ValueError("calendar_min_periods must be at least 2")
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be in the interval (0, 1)")

    frame = frame.copy().sort_index()
    frame.index = _as_utc_datetime_index(frame.index)
    close = frame["close"].dropna().astype(float)
    features = make_default_features(
        close,
        volatility=volatility,
        fast_span=fast_span,
        slow_span=slow_span,
        fracdiff_d=fracdiff_d,
        fracdiff_threshold=fracdiff_threshold,
    )
    log_return = np.log(close).diff()
    features["log_return_1"] = log_return
    features["realized_vol"] = log_return.rolling(microstructure_window).std()
    features["return_skew"] = log_return.rolling(microstructure_window).skew()
    features["return_kurt"] = log_return.rolling(microstructure_window).kurt()

    if {"high", "low"}.issubset(frame.columns):
        high = frame["high"].astype(float).reindex(close.index)
        low = frame["low"].astype(float).reindex(close.index)
        features["hl_range"] = np.log(_safe_divide(high, low))
    if {"open", "close"}.issubset(frame.columns):
        open_ = frame["open"].astype(float).reindex(close.index)
        features["oc_return"] = np.log(_safe_divide(close, open_))

    volume = (
        frame["volume"].astype(float).reindex(close.index)
        if "volume" in frame.columns
        else pd.Series(np.nan, index=close.index)
    )
    if {"buy_notional", "sell_notional"}.issubset(frame.columns):
        buy_notional = frame["buy_notional"].astype(float).reindex(close.index)
        sell_notional = frame["sell_notional"].astype(float).reindex(close.index)
        total_notional = buy_notional + sell_notional
        signed_notional = buy_notional - sell_notional
    else:
        buy_notional = pd.Series(np.nan, index=close.index)
        sell_notional = pd.Series(np.nan, index=close.index)
        total_notional = (close * volume).rename("dollar_volume")
        signed_notional = (np.sign(log_return).fillna(0.0) * total_notional).rename(
            "signed_notional"
        )

    if "signed_notional" in frame.columns:
        signed_notional = frame["signed_notional"].astype(float).reindex(close.index)

    signed_notional_cumsum = signed_notional.fillna(0.0).cumsum().rename("signed_notional_cumsum")
    signed_volume = _safe_divide(signed_notional, close).rename("signed_volume")
    cvd = signed_volume.fillna(0.0).cumsum().rename("cvd")
    signed_notional_cumsum_ffd = fracdiff_ffd(
        signed_notional_cumsum,
        d=fracdiff_d,
        threshold=fracdiff_threshold,
    ).reindex(close.index)

    features["log_volume"] = np.log1p(volume)
    features["log_dollar_volume"] = np.log1p(total_notional)
    features["volume_zscore"] = (
        volume - volume.rolling(microstructure_window).mean()
    ) / volume.rolling(microstructure_window).std()
    features["dollar_volume_zscore"] = (
        total_notional - total_notional.rolling(microstructure_window).mean()
    ) / total_notional.rolling(microstructure_window).std()
    features["signed_dollar_imbalance"] = _safe_divide(signed_notional, total_notional)
    features["signed_dollar_imbalance_abs"] = features["signed_dollar_imbalance"].abs()
    buy_notional_share = (
        _safe_divide(buy_notional, total_notional)
        if {"buy_notional", "sell_notional"}.issubset(frame.columns)
        else pd.Series(np.nan, index=close.index)
    )
    sell_notional_share = (
        _safe_divide(sell_notional, total_notional)
        if {"buy_notional", "sell_notional"}.issubset(frame.columns)
        else pd.Series(np.nan, index=close.index)
    )
    features["buy_notional_share"] = buy_notional_share
    features["sell_notional_share"] = sell_notional_share
    features["notional_side_hhi"] = buy_notional_share.pow(2) + sell_notional_share.pow(2)
    features["notional_side_hhi_chg"] = features["notional_side_hhi"].diff()
    features["notional_side_hhi_zscore"] = _rolling_zscore(
        features["notional_side_hhi"],
        microstructure_window,
    )
    features["buy_sell_notional_ratio"] = _safe_divide(buy_notional, sell_notional)
    features["order_flow_imbalance"] = (
        features["signed_dollar_imbalance"]
        .rolling(
            microstructure_window,
        )
        .mean()
    )
    features["vpin"] = _safe_divide(
        signed_notional.abs().rolling(microstructure_window).sum(),
        total_notional.rolling(microstructure_window).sum(),
    )
    features["vpin_zscore"] = _rolling_zscore(features["vpin"], microstructure_window)
    features["signed_notional_cumsum_ffd"] = signed_notional_cumsum_ffd
    features["signed_notional_cumsum_ffd_lag_2"] = signed_notional_cumsum_ffd.shift(1)
    features["signed_notional_cumsum_ffd_lag_4"] = signed_notional_cumsum_ffd.shift(3)
    features["signed_notional_cumsum_ffd_ewm_fast"] = signed_notional_cumsum_ffd.ewm(
        span=fast_span,
    ).mean()
    features["signed_notional_cumsum_ffd_ewm_slow"] = signed_notional_cumsum_ffd.ewm(
        span=slow_span,
    ).mean()
    features["signed_notional_cumsum_ffd_zscore"] = _rolling_zscore(
        signed_notional_cumsum_ffd,
        microstructure_window,
    )
    features["cvd"] = cvd
    features["cvd_change"] = cvd.diff()
    features["cvd_zscore"] = _rolling_zscore(cvd, microstructure_window)
    time_trend = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    features["cvd_slope"] = _rolling_ols_slope(cvd, time_trend, microstructure_window)

    signed_sqrt_dollar_volume = (np.sign(signed_notional) * np.sqrt(signed_notional.abs())).rename(
        "signed_sqrt_dollar_volume"
    )
    price_change = close.diff()
    features["kyle_lambda"] = _rolling_ols_slope(
        price_change,
        signed_volume,
        microstructure_window,
    )
    features["amihud_lambda"] = (
        _safe_divide(log_return.abs(), total_notional)
        .rolling(
            microstructure_window,
            min_periods=max(3, microstructure_window // 2),
        )
        .mean()
    )
    features["hasbrouck_lambda"] = _rolling_ols_slope(
        log_return,
        signed_sqrt_dollar_volume,
        microstructure_window,
    )
    features["sqrt_dollar_imbalance"] = _safe_divide(
        signed_sqrt_dollar_volume,
        np.sqrt(total_notional),
    )

    if {"buy_ticks", "sell_ticks"}.issubset(frame.columns):
        buy_ticks = frame["buy_ticks"].astype(float).reindex(close.index)
        sell_ticks = frame["sell_ticks"].astype(float).reindex(close.index)
        total_ticks = buy_ticks + sell_ticks
        features["buy_tick_ratio"] = _safe_divide(buy_ticks, total_ticks)
        features["tick_imbalance"] = _safe_divide(buy_ticks - sell_ticks, total_ticks)
        sell_tick_ratio = _safe_divide(sell_ticks, total_ticks)
        features["tick_side_hhi"] = features["buy_tick_ratio"].pow(2) + sell_tick_ratio.pow(2)
        features["tick_side_hhi_zscore"] = _rolling_zscore(
            features["tick_side_hhi"],
            microstructure_window,
        )
    if "ticks" in frame.columns:
        ticks = frame["ticks"].astype(float).reindex(close.index)
        features["log_ticks"] = np.log1p(ticks)
        features["tick_zscore"] = (
            ticks - ticks.rolling(microstructure_window).mean()
        ) / ticks.rolling(microstructure_window).std()
        features["notional_per_tick"] = _safe_divide(total_notional, ticks)
        features["notional_per_tick_zscore"] = _rolling_zscore(
            features["notional_per_tick"],
            microstructure_window,
        )

    bid_depth = _first_numeric_column(
        frame,
        ("bid_depth", "bid_size", "bid_qty", "bid_volume", "best_bid_size"),
        close.index,
    )
    ask_depth = _first_numeric_column(
        frame,
        ("ask_depth", "ask_size", "ask_qty", "ask_volume", "best_ask_size"),
        close.index,
    )
    if bid_depth is not None and ask_depth is not None:
        total_depth = bid_depth + ask_depth
        features["lob_depth_imbalance"] = _safe_divide(bid_depth - ask_depth, total_depth)
        features["lob_depth_ratio"] = _safe_divide(bid_depth, ask_depth)
        features["lob_depth_hhi"] = (
            _safe_divide(bid_depth, total_depth).pow(2)
            + _safe_divide(ask_depth, total_depth).pow(2)
        )
        features["lob_depth_imbalance_zscore"] = _rolling_zscore(
            features["lob_depth_imbalance"],
            microstructure_window,
        )

    bid_price = _first_numeric_column(frame, ("bid", "bid_price", "best_bid"), close.index)
    ask_price = _first_numeric_column(frame, ("ask", "ask_price", "best_ask"), close.index)
    if bid_price is not None and ask_price is not None:
        quoted_spread = (ask_price - bid_price).rename("quoted_spread")
        relative_spread = _safe_divide(quoted_spread, close).rename("relative_spread")
        features["quoted_spread"] = quoted_spread
        features["relative_spread"] = relative_spread
        features["spread_zscore"] = _rolling_zscore(relative_spread, microstructure_window)
        features["spread_elasticity"] = _safe_divide(
            relative_spread.pct_change().replace([np.inf, -np.inf], np.nan),
            log_return.abs(),
        )
    elif {"high", "low"}.issubset(frame.columns):
        bar_spread_proxy = _safe_divide(high - low, close).rename("bar_spread_proxy")
        features["bar_spread_proxy"] = bar_spread_proxy
        features["bar_spread_elasticity_proxy"] = _safe_divide(
            bar_spread_proxy.pct_change().replace([np.inf, -np.inf], np.nan),
            log_return.abs(),
        )

    for column in ("theta", "threshold", "threshold_scale", "expected_ticks"):
        if column in frame.columns:
            value = frame[column].astype(float).reindex(close.index)
            features[column] = value
            features[f"{column}_chg"] = value.pct_change().replace([np.inf, -np.inf], np.nan)

    theta_to_threshold: pd.Series | None = None
    if {"theta", "threshold"}.issubset(frame.columns):
        theta = frame["theta"].astype(float).reindex(close.index)
        threshold = frame["threshold"].astype(float).reindex(close.index)
        theta_to_threshold = _safe_divide(theta.abs(), threshold)
        features["theta_to_threshold"] = theta_to_threshold
        features["theta_to_threshold_chg"] = theta_to_threshold.pct_change().replace(
            [np.inf, -np.inf],
            np.nan,
        )
        features["theta_to_threshold_zscore"] = _rolling_zscore(
            theta_to_threshold,
            microstructure_window,
        )
        features["theta_to_threshold_robust_zscore"] = _rolling_iqr_zscore(
            theta_to_threshold,
            microstructure_window,
            min_periods=max(3, microstructure_window // 2),
            clip=robust_z_clip,
        )

    if calendar_windows:
        _add_calendar_regime_features(
            features,
            close=close,
            log_return=log_return,
            total_notional=total_notional,
            signed_notional=signed_notional,
            theta_to_threshold=theta_to_threshold,
            calendar_windows=tuple(calendar_windows),
            min_periods=calendar_min_periods,
            tail_quantile=tail_quantile,
            robust_z_clip=robust_z_clip,
        )
        features = features.copy()

    features["shannon_entropy"] = rolling_binary_entropy(log_return, window=information_window)
    features["lz_complexity"] = rolling_lz_complexity(log_return, window=information_window)
    features["entropy_zscore"] = _rolling_zscore(features["shannon_entropy"], information_window)
    features["lz_zscore"] = _rolling_zscore(features["lz_complexity"], information_window)
    features["flow_shannon_entropy"] = rolling_binary_entropy(
        signed_notional,
        window=information_window,
    )
    features["flow_lz_complexity"] = rolling_lz_complexity(
        signed_notional,
        window=information_window,
    )

    log_close = np.log(close)
    sadf_stats = [
        rolling_adf_t_stat(log_close, window=window).reindex(close.index) for window in sadf_windows
    ]
    for stat in sadf_stats:
        features[stat.name] = stat
    features["sadf"] = pd.concat(sadf_stats, axis=1).max(axis=1)
    features["sadf_change"] = features["sadf"].diff()
    features["sadf_zscore"] = _rolling_zscore(features["sadf"], information_window)
    return features.replace([np.inf, -np.inf], np.nan)


# -------------------------------------------------------------------------------------------------
# Pipeline composition and Nautilus I/O
# -------------------------------------------------------------------------------------------------


def _prepare_features(
    close: pd.Series,
    features: pd.DataFrame | None,
    volatility: pd.Series,
    *,
    feature_fracdiff_d: float,
    feature_fracdiff_threshold: float,
) -> pd.DataFrame:
    if features is None:
        return make_default_features(
            close,
            volatility=volatility,
            fracdiff_d=feature_fracdiff_d,
            fracdiff_threshold=feature_fracdiff_threshold,
        )
    features = features.copy().sort_index()
    features.index = _as_utc_datetime_index(features.index)
    return features


def _resolve_cusum_threshold(
    close: pd.Series,
    volatility: pd.Series,
    *,
    cusum_threshold: float | pd.Series | None,
    cusum_threshold_mult: float,
) -> pd.Series:
    if cusum_threshold is None:
        if cusum_threshold_mult <= 0.0:
            raise ValueError("cusum_threshold_mult must be positive")
        return (volatility.reindex(close.index).ffill() * cusum_threshold_mult).rename(
            "cusum_threshold"
        )
    if isinstance(cusum_threshold, pd.Series):
        resolved = cusum_threshold.astype(float).sort_index()
        resolved.index = _as_utc_datetime_index(resolved.index)
        return resolved.reindex(close.index).ffill().rename("cusum_threshold")

    resolved_value = float(cusum_threshold)
    if resolved_value <= 0.0:
        raise ValueError("cusum_threshold must be positive")
    return pd.Series(resolved_value, index=close.index, name="cusum_threshold")


def build_afml_dataset(
    close: pd.Series,
    *,
    features: pd.DataFrame | None = None,
    volatility_span: int = 100,
    cusum_threshold_mult: float = 1.0,
    cusum_threshold: float | pd.Series | None = None,
    pt_sl: tuple[float, float] | dict[str, float] = (1.0, 1.0),
    target: pd.Series | None = None,
    min_ret: float = 0.0,
    vertical_barrier_days: float | None = 1,
    vertical_barrier_bars: int | None = None,
    zero_on_vertical: bool = False,
    meta_label: bool = False,
    drop_neutral: bool = True,
    rare_label_min_pct: float | None = None,
    oldest_weight: float = 1.0,
    feature_fracdiff_d: float = DEFAULT_FEATURE_FRACDIFF_D,
    feature_fracdiff_threshold: float = DEFAULT_FEATURE_FRACDIFF_THRESHOLD,
    return_type: str = "simple",
) -> AfmlDataset:
    """
    Build features, events, labels, and sample weights for model training.
    """
    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    if close.shape[0] < 10:
        raise ValueError("At least 10 close observations are required")
    if target is None:
        volatility = get_daily_vol(close, span=volatility_span)
    else:
        volatility = target.dropna().astype(float).sort_index()
        volatility.index = _as_utc_datetime_index(volatility.index)
        volatility = volatility.rename("trgt")
    resolved_cusum_threshold = _resolve_cusum_threshold(
        close,
        volatility,
        cusum_threshold=cusum_threshold,
        cusum_threshold_mult=cusum_threshold_mult,
    )
    t_events = symmetric_cusum_filter(np.log(close), threshold=resolved_cusum_threshold)
    if t_events.empty:
        raise ValueError("CUSUM filter produced no events; lower cusum_threshold_mult")

    vertical_barriers = add_vertical_barrier(
        t_events,
        close,
        num_days=vertical_barrier_days,
        num_bars=vertical_barrier_bars,
    )
    events = get_triple_barrier_events(
        close,
        t_events,
        pt_sl,
        volatility,
        min_ret=min_ret,
        t1=vertical_barriers,
        return_type=return_type,
    )
    labels = get_bins(
        events,
        close,
        zero_on_vertical=zero_on_vertical,
        meta_label=meta_label,
        return_type=return_type,
    )
    if drop_neutral and not meta_label:
        labels = drop_neutral_labels(labels)
    if rare_label_min_pct is not None:
        labels = drop_rare_labels(labels, min_pct=rare_label_min_pct)

    if labels.empty:
        raise ValueError("Triple-barrier labeling produced no usable labels")

    features = _prepare_features(
        close,
        features,
        volatility,
        feature_fracdiff_d=feature_fracdiff_d,
        feature_fracdiff_threshold=feature_fracdiff_threshold,
    )

    aligned_features = features.reindex(labels.index)
    valid = aligned_features.notna().all(axis=1)
    labels = labels.loc[valid]
    aligned_features = aligned_features.loc[valid]
    events = events.loc[labels.index]
    if labels.empty:
        raise ValueError("No labels remain after feature alignment")

    num_co_events = num_concurrent_events(close.index, events["t1"])
    uniqueness = average_uniqueness(events["t1"], num_co_events).reindex(labels.index)
    return_weights = sample_weights_by_return(events["t1"], num_co_events, close).reindex(
        labels.index
    )
    decay = time_decay_weights(uniqueness, oldest_weight=oldest_weight).reindex(labels.index)
    sample_weight = (return_weights * decay).fillna(0.0)
    if sample_weight.sum() <= 0.0:
        sample_weight = uniqueness.fillna(1.0)
    sample_weight = _normalize_sample_weights(sample_weight).rename("sample_weight")

    return AfmlDataset(
        close=close,
        features=aligned_features,
        events=events,
        labels=labels,
        sample_weight=sample_weight,
        volatility=volatility,
        t_events=t_events,
        cusum_threshold=resolved_cusum_threshold,
    )


def build_afml_meta_dataset(
    close: pd.Series,
    side: pd.Series,
    *,
    features: pd.DataFrame | None = None,
    volatility_span: int = 100,
    cusum_threshold_mult: float = 1.0,
    pt_sl: tuple[float, float] | dict[str, float] = (1.0, 1.0),
    target: pd.Series | None = None,
    min_ret: float = 0.0,
    meta_label_min_ret: float = 0.0,
    vertical_barrier_days: float | None = 1,
    vertical_barrier_bars: int | None = None,
    oldest_weight: float = 1.0,
    feature_fracdiff_d: float = DEFAULT_FEATURE_FRACDIFF_D,
    feature_fracdiff_threshold: float = DEFAULT_FEATURE_FRACDIFF_THRESHOLD,
    return_type: str = "simple",
) -> AfmlDataset:
    """
    Build a meta-labeling dataset from externally supplied bet sides.

    ``side`` is the primary model's {-1, 1} decision. Labels are {0, 1}, where
    1 means the side-adjusted return exceeded ``meta_label_min_ret`` and 0
    means pass. Use ``meta_label_min_ret`` to model taker fees and slippage.
    """
    close = close.dropna().astype(float).sort_index()
    close.index = _as_utc_datetime_index(close.index)
    side = side.dropna().astype(int).sort_index()
    side.index = _as_utc_datetime_index(side.index)
    side = side[side != 0]
    if close.shape[0] < 10:
        raise ValueError("At least 10 close observations are required")
    if cusum_threshold_mult <= 0.0:
        raise ValueError("cusum_threshold_mult must be positive")

    if target is None:
        volatility = get_daily_vol(close, span=volatility_span)
    else:
        volatility = target.dropna().astype(float).sort_index()
        volatility.index = _as_utc_datetime_index(volatility.index)
        volatility = volatility.rename("trgt")
    t_events = pd.DatetimeIndex(side.index.intersection(close.index))
    if t_events.empty:
        raise ValueError("No sided events overlap the close series")

    vertical_barriers = add_vertical_barrier(
        t_events,
        close,
        num_days=vertical_barrier_days,
        num_bars=vertical_barrier_bars,
    )
    events = get_triple_barrier_events(
        close,
        t_events,
        pt_sl,
        volatility,
        min_ret=min_ret,
        t1=vertical_barriers,
        side=side,
        return_type=return_type,
    )
    labels = get_bins(
        events,
        close,
        meta_label=True,
        meta_label_min_ret=meta_label_min_ret,
        return_type=return_type,
    )
    if labels.empty:
        raise ValueError("Meta-labeling produced no usable labels")

    features = _prepare_features(
        close,
        features,
        volatility,
        feature_fracdiff_d=feature_fracdiff_d,
        feature_fracdiff_threshold=feature_fracdiff_threshold,
    )

    aligned_features = features.reindex(labels.index)
    valid = aligned_features.notna().all(axis=1)
    labels = labels.loc[valid]
    aligned_features = aligned_features.loc[valid]
    events = events.loc[labels.index]
    if labels.empty:
        raise ValueError("No meta-labels remain after feature alignment")

    num_co_events = num_concurrent_events(close.index, events["t1"])
    uniqueness = average_uniqueness(events["t1"], num_co_events).reindex(labels.index)
    return_weights = sample_weights_by_return(events["t1"], num_co_events, close).reindex(
        labels.index
    )
    decay = time_decay_weights(uniqueness, oldest_weight=oldest_weight).reindex(labels.index)
    sample_weight = (return_weights * decay).fillna(0.0)
    if sample_weight.sum() <= 0.0:
        sample_weight = uniqueness.fillna(1.0)
    sample_weight = _normalize_sample_weights(sample_weight).rename("sample_weight")

    return AfmlDataset(
        close=close,
        features=aligned_features,
        events=events,
        labels=labels,
        sample_weight=sample_weight,
        volatility=volatility,
        t_events=t_events,
        cusum_threshold=None,
    )


def save_model_artifact(model: Any, path: str | Path) -> Path:
    """
    Save a fitted model using joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Saving this model requires joblib") from exc

    if path.suffix.lower() == ".json":
        path = path.with_suffix(".joblib")
    joblib.dump(model, path)
    return path


def _split_dataset_indices(
    dataset: AfmlDataset,
    *,
    train_fraction: float,
    train_start: TimeLike,
    train_end: TimeLike,
    test_start: TimeLike,
    test_end: TimeLike,
    require_label_end_within_split: bool,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    X = dataset.X
    event_end = pd.Series(
        pd.to_datetime(dataset.events.loc[X.index, "t1"], utc=True),
        index=X.index,
    )
    explicit_dates = any(
        value is not None for value in (train_start, train_end, test_start, test_end)
    )
    if explicit_dates:
        train_mask = _date_mask(pd.DatetimeIndex(X.index), train_start, train_end)
        test_mask = _date_mask(pd.DatetimeIndex(X.index), test_start, test_end)
        if require_label_end_within_split:
            train_end_ts = _time_bound(train_end, is_end=True)
            test_end_ts = _time_bound(test_end, is_end=True)
            if train_end_ts is not None:
                train_mask &= event_end <= train_end_ts
            if test_end_ts is not None:
                test_mask &= event_end <= test_end_ts
        train_index = pd.DatetimeIndex(X.index[train_mask])
        test_index = pd.DatetimeIndex(X.index[test_mask])
    else:
        split = int(X.shape[0] * train_fraction)
        if split < 1 or split >= X.shape[0]:
            raise ValueError("Train/test split leaves one side empty")
        train_index = pd.DatetimeIndex(X.index[:split])
        test_index = pd.DatetimeIndex(X.index[split:])

    if train_index.empty or test_index.empty:
        raise ValueError("Train/test split leaves one side empty")
    return train_index, test_index


def _fit_classifier(model: Any, dataset: AfmlDataset, train_index: pd.DatetimeIndex) -> Any:
    X = dataset.X
    y = dataset.y
    if getattr(model, "requires_event_spans", False):
        model.fit(
            X.loc[train_index],
            y.loc[train_index],
            sample_weight=dataset.sample_weight.loc[train_index],
            t1=dataset.events.loc[train_index, "t1"],
            bar_index=dataset.close.index,
        )
    else:
        model.fit(
            X.loc[train_index],
            y.loc[train_index],
            sample_weight=dataset.sample_weight.loc[train_index],
        )
    return model


def _replace_dataset_features(dataset: AfmlDataset, features: pd.DataFrame) -> AfmlDataset:
    features = features.reindex(dataset.features.index)
    return AfmlDataset(
        close=dataset.close,
        features=features,
        events=dataset.events,
        labels=dataset.labels,
        sample_weight=dataset.sample_weight,
        volatility=dataset.volatility,
        t_events=dataset.t_events,
    )


def _fit_pca_feature_transformer(
    features: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    *,
    n_components: float,
    random_state: int | None = None,
) -> Any:
    try:
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("PCA feature transformation requires scikit-learn") from exc

    train_features = features.loc[train_index].replace([np.inf, -np.inf], np.nan).dropna()
    if train_features.empty:
        raise ValueError("Cannot fit PCA on an empty training feature matrix")

    if isinstance(n_components, int):
        max_components = min(train_features.shape[0], train_features.shape[1])
        n_components = min(n_components, max_components)
        if n_components < 1:
            raise ValueError("pca_components must leave at least one component")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
        ],
    ).fit(train_features)


def _transform_pca_features(transformer: Any, features: pd.DataFrame) -> pd.DataFrame:
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    transformed = transformer.transform(features)
    columns = [f"pca_{idx + 1:03d}" for idx in range(transformed.shape[1])]
    return pd.DataFrame(transformed, index=features.index, columns=columns)


def _pca_dataset(
    dataset: AfmlDataset,
    train_index: pd.DatetimeIndex,
    *,
    pca_components: float | None,
    pca_random_state: int | None = None,
) -> tuple[AfmlDataset, Any | None]:
    if pca_components is None:
        return dataset, None

    transformer = _fit_pca_feature_transformer(
        dataset.X,
        train_index,
        n_components=pca_components,
        random_state=pca_random_state,
    )
    transformed = _transform_pca_features(transformer, dataset.features)
    return _replace_dataset_features(dataset, transformed), transformer


def project_pca_importance_to_features(
    component_importance: pd.DataFrame,
    transformer: Any,
    original_features: list[str] | pd.Index,
    *,
    method: str | None = None,
) -> pd.DataFrame:
    """
    Back-project PCA component importances onto original feature names.
    """
    if component_importance.empty:
        return pd.DataFrame(columns=["method", "feature", "importance", "std"])
    if "importance" not in component_importance or "feature" not in component_importance:
        raise ValueError("component_importance must contain 'feature' and 'importance'")

    pca = transformer.named_steps["pca"] if hasattr(transformer, "named_steps") else transformer
    components = np.asarray(pca.components_, dtype=float)
    feature_names = [str(feature) for feature in original_features]
    if components.shape[1] != len(feature_names):
        raise ValueError("PCA component width does not match original feature count")

    component_names = [f"pca_{idx + 1:03d}" for idx in range(components.shape[0])]
    importance = (
        component_importance.set_index("feature")["importance"]
        .reindex(component_names)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    weights = np.abs(components).T @ importance
    total = float(weights.sum())
    if total > 0.0:
        weights = weights / total

    output_method = method or str(component_importance.iloc[0].get("method", "PCA_BACKPROJECTED"))
    return pd.DataFrame(
        {
            "method": output_method,
            "feature": feature_names,
            "importance": weights,
            "std": 0.0,
        },
    ).sort_values("importance", ascending=False, ignore_index=True)


def pca_mdi_feature_importance(
    model: Any,
    transformer: Any,
    original_features: list[str] | pd.Index,
) -> pd.DataFrame:
    component_importance = mdi_feature_importance(model)
    return project_pca_importance_to_features(
        component_importance,
        transformer,
        original_features,
        method="MDI_PCA_BACKPROJECTED",
    )


def pca_mda_feature_importance(
    model: Any,
    transformer: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: pd.Series | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    transformed = _transform_pca_features(transformer, X)
    component_importance = mda_feature_importance(
        model,
        transformed,
        y,
        sample_weight=sample_weight,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return project_pca_importance_to_features(
        component_importance,
        transformer,
        X.columns,
        method="MDA_PCA_BACKPROJECTED",
    )


def pca_sfi_feature_importance(
    dataset: AfmlDataset,
    *,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    model_factory: Any,
    pca_components: float,
    pca_random_state: int | None = None,
) -> pd.DataFrame:
    pca_dataset, transformer = _pca_dataset(
        dataset,
        train_index,
        pca_components=pca_components,
        pca_random_state=pca_random_state,
    )
    if transformer is None:
        raise ValueError("pca_components must not be None")
    component_importance = sfi_feature_importance(
        pca_dataset,
        train_index=train_index,
        test_index=test_index,
        model_factory=model_factory,
    )
    return project_pca_importance_to_features(
        component_importance,
        transformer,
        dataset.X.columns,
        method="SFI_PCA_BACKPROJECTED",
    )


def _binary_precision_recall(
    truth: pd.Series,
    prediction: pd.Series,
) -> tuple[float, float]:
    truth = truth.astype(int)
    prediction = prediction.reindex(truth.index).fillna(0).astype(int)
    true_positive = int(((truth == 1) & (prediction == 1)).sum())
    false_positive = int(((truth == 0) & (prediction == 1)).sum())
    false_negative = int(((truth == 1) & (prediction == 0)).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return float(precision), float(recall)


def primary_side_frame(
    model: Any,
    features: pd.DataFrame,
    *,
    probability_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Convert a side model into {-1, 1} candidates plus confidence features.
    """
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    proba = model.predict_proba(features)
    predictions = pd.Series(proba.idxmax(axis=1).astype(int), index=proba.index)
    confidence = proba.max(axis=1)
    side = predictions.where(confidence >= probability_threshold, 0).astype(int)

    frame = pd.DataFrame(
        {
            "primary_side": side,
            "primary_prediction": predictions,
            "primary_confidence": confidence,
        },
        index=features.index,
    )
    for column in proba.columns:
        frame[f"primary_prob_{int(column)}"] = proba[column]
    if -1 in proba.columns and 1 in proba.columns:
        frame["primary_prob_margin"] = proba[1] - proba[-1]
    return frame


def make_meta_features(features: pd.DataFrame, primary_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add primary model confidence columns for the meta model.
    """
    primary_columns = [
        column
        for column in primary_frame.columns
        if column.startswith("primary_prob_") or column == "primary_confidence"
    ]
    return features.join(primary_frame[primary_columns], how="inner")


def make_signal_frame(
    model: Any,
    features: pd.DataFrame,
    *,
    probability_threshold: float = 0.55,
    active_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Convert classifier probabilities to {-1, 0, 1} trading signals.
    """
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1]")

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    proba = model.predict_proba(features)
    predictions = pd.Series(proba.idxmax(axis=1).astype(int), index=proba.index)
    confidence = proba.max(axis=1)
    signal = predictions.where(confidence >= probability_threshold, 0).astype(int)

    frame = pd.DataFrame(
        {
            "signal": signal,
            "prediction": predictions,
            "confidence": confidence,
        },
        index=features.index,
    )
    for column in proba.columns:
        frame[f"prob_{int(column)}"] = proba[column]

    if active_index is not None:
        active_index = _as_utc_datetime_index(active_index)
        inactive = ~frame.index.isin(active_index)
        frame.loc[inactive, "signal"] = 0
    return frame


def fit_afml_model(
    dataset: AfmlDataset,
    *,
    train_fraction: float = 0.7,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    test_start: TimeLike = None,
    test_end: TimeLike = None,
    require_label_end_within_split: bool = True,
    probability_threshold: float = 0.55,
    model: Any | None = None,
    pca_components: float | None = None,
    pca_random_state: int | None = None,
) -> AfmlFitResult:
    """
    Fit a weighted classifier using chronological or explicit date splitting.
    """
    train_index, test_index = _split_dataset_indices(
        dataset,
        train_fraction=train_fraction,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        require_label_end_within_split=require_label_end_within_split,
    )
    dataset, feature_transformer = _pca_dataset(
        dataset,
        train_index,
        pca_components=pca_components,
        pca_random_state=pca_random_state,
    )
    X = dataset.X
    y = dataset.y
    model = model or SequentialBootstrapBaggingClassifier()
    model = _fit_classifier(model, dataset, train_index)

    train_pred = model.predict(X.loc[train_index])
    test_pred = model.predict(X.loc[test_index])
    train_accuracy = float((train_pred == y.loc[train_index]).mean())
    test_accuracy = float((test_pred == y.loc[test_index]).mean())
    signals = make_signal_frame(
        model,
        X,
        probability_threshold=probability_threshold,
        active_index=test_index,
    )

    return AfmlFitResult(
        model=model,
        train_index=train_index,
        test_index=test_index,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        signals=signals,
        feature_transformer=feature_transformer,
    )


def make_meta_signal_frame(
    primary_model: Any,
    meta_model: Any,
    features: pd.DataFrame,
    *,
    primary_features: pd.DataFrame | None = None,
    meta_feature_transformer: Any | None = None,
    primary_probability_threshold: float = 0.0,
    meta_probability_threshold: float = 0.6,
    active_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Combine primary sides with meta-model probabilities into trading signals.
    """
    if not 0.0 <= primary_probability_threshold <= 1.0:
        raise ValueError("primary_probability_threshold must be in [0, 1]")
    if not 0.0 <= meta_probability_threshold <= 1.0:
        raise ValueError("meta_probability_threshold must be in [0, 1]")

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    primary_features = features if primary_features is None else primary_features
    primary_features = primary_features.reindex(features.index).dropna()
    primary = primary_side_frame(
        primary_model,
        primary_features,
        probability_threshold=primary_probability_threshold,
    )
    candidate_index = primary.index[primary["primary_side"] != 0]
    frame = pd.DataFrame(
        {
            "signal": 0,
            "prediction": primary["primary_prediction"],
            "confidence": 0.0,
            "primary_side": primary["primary_side"],
            "primary_confidence": primary["primary_confidence"],
            "meta_probability": 0.0,
            "meta_prediction": 0,
        },
        index=primary.index,
    )
    for column in primary.columns:
        if column.startswith("primary_prob_"):
            frame[column] = primary[column]

    if len(candidate_index) > 0:
        meta_features = make_meta_features(
            features.loc[candidate_index], primary.loc[candidate_index]
        )
        if meta_feature_transformer is not None:
            meta_features = _transform_pca_features(meta_feature_transformer, meta_features)
        meta_proba = meta_model.predict_proba(meta_features)
        if 1 in meta_proba.columns:
            trade_probability = meta_proba[1]
        else:
            trade_probability = pd.Series(0.0, index=meta_proba.index)
        meta_prediction = (trade_probability >= meta_probability_threshold).astype(int)
        frame.loc[meta_features.index, "meta_probability"] = trade_probability
        frame.loc[meta_features.index, "meta_prediction"] = meta_prediction
        frame.loc[meta_features.index, "confidence"] = trade_probability
        frame.loc[meta_features.index, "signal"] = (
            primary.loc[meta_features.index, "primary_side"] * meta_prediction
        ).astype(int)

    if active_index is not None:
        active_index = _as_utc_datetime_index(active_index)
        inactive = ~frame.index.isin(active_index)
        frame.loc[inactive, "signal"] = 0
    return frame


def fit_afml_meta_model(
    primary_dataset: AfmlDataset,
    *,
    train_fraction: float = 0.7,
    train_start: TimeLike = None,
    train_end: TimeLike = None,
    test_start: TimeLike = None,
    test_end: TimeLike = None,
    require_label_end_within_split: bool = True,
    primary_probability_threshold: float = 0.0,
    meta_probability_threshold: float = 0.6,
    primary_model: Any | None = None,
    meta_model: Any | None = None,
    volatility_span: int = 100,
    pt_sl: tuple[float, float] | dict[str, float] = (1.0, 1.0),
    target: pd.Series | None = None,
    min_ret: float = 0.0,
    meta_label_min_ret: float = 0.0,
    vertical_barrier_days: float | None = 1,
    vertical_barrier_bars: int | None = None,
    oldest_weight: float = 1.0,
    pca_components: float | None = None,
    pca_random_state: int | None = None,
) -> AfmlMetaFitResult:
    """
    Fit AFML meta-labeling: primary side model plus secondary pass/trade model.

    The primary model is intentionally thresholded for recall. The meta model then
    filters the primary model's opportunities for precision.
    """
    raw_X = primary_dataset.X
    y = primary_dataset.y
    primary_train_index, primary_test_index = _split_dataset_indices(
        primary_dataset,
        train_fraction=train_fraction,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        require_label_end_within_split=require_label_end_within_split,
    )
    primary_model_dataset, primary_feature_transformer = _pca_dataset(
        primary_dataset,
        primary_train_index,
        pca_components=pca_components,
        pca_random_state=pca_random_state,
    )
    primary_X = primary_model_dataset.X
    primary_model = primary_model or SequentialBootstrapBaggingClassifier()
    primary_model = _fit_classifier(primary_model, primary_model_dataset, primary_train_index)

    primary_frame = primary_side_frame(
        primary_model,
        primary_X,
        probability_threshold=primary_probability_threshold,
    )
    side = primary_frame["primary_side"]
    meta_features = make_meta_features(raw_X, primary_frame)
    meta_dataset = build_afml_meta_dataset(
        primary_dataset.close,
        side,
        features=meta_features,
        volatility_span=volatility_span,
        cusum_threshold_mult=1.0,
        pt_sl=pt_sl,
        target=target if target is not None else primary_dataset.volatility,
        min_ret=min_ret,
        meta_label_min_ret=meta_label_min_ret,
        vertical_barrier_days=vertical_barrier_days,
        vertical_barrier_bars=vertical_barrier_bars,
        oldest_weight=oldest_weight,
    )
    meta_train_index, meta_test_index = _split_dataset_indices(
        meta_dataset,
        train_fraction=train_fraction,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        require_label_end_within_split=require_label_end_within_split,
    )
    meta_model_dataset, meta_feature_transformer = _pca_dataset(
        meta_dataset,
        meta_train_index,
        pca_components=pca_components,
        pca_random_state=pca_random_state,
    )
    meta_model = meta_model or SequentialBootstrapBaggingClassifier()
    meta_model = _fit_classifier(meta_model, meta_model_dataset, meta_train_index)

    primary_train_pred = primary_model.predict(primary_X.loc[primary_train_index])
    primary_test_pred = primary_model.predict(primary_X.loc[primary_test_index])
    primary_train_accuracy = float((primary_train_pred == y.loc[primary_train_index]).mean())
    primary_test_accuracy = float((primary_test_pred == y.loc[primary_test_index]).mean())

    meta_train_proba = meta_model.predict_proba(meta_model_dataset.X.loc[meta_train_index])
    meta_test_proba = meta_model.predict_proba(meta_model_dataset.X.loc[meta_test_index])
    meta_train_pred = (meta_train_proba[1] >= meta_probability_threshold).astype(int)
    meta_test_pred = (meta_test_proba[1] >= meta_probability_threshold).astype(int)
    meta_train_precision, meta_train_recall = _binary_precision_recall(
        meta_dataset.y.loc[meta_train_index],
        meta_train_pred,
    )
    meta_test_precision, meta_test_recall = _binary_precision_recall(
        meta_dataset.y.loc[meta_test_index],
        meta_test_pred,
    )
    signals = make_meta_signal_frame(
        primary_model,
        meta_model,
        raw_X,
        primary_features=primary_X,
        meta_feature_transformer=meta_feature_transformer,
        primary_probability_threshold=primary_probability_threshold,
        meta_probability_threshold=meta_probability_threshold,
        active_index=meta_test_index,
    )

    return AfmlMetaFitResult(
        primary_model=primary_model,
        meta_model=meta_model,
        primary_dataset=primary_model_dataset,
        meta_dataset=meta_model_dataset,
        primary_train_index=primary_train_index,
        primary_test_index=primary_test_index,
        train_index=meta_train_index,
        test_index=meta_test_index,
        primary_train_accuracy=primary_train_accuracy,
        primary_test_accuracy=primary_test_accuracy,
        meta_train_precision=meta_train_precision,
        meta_train_recall=meta_train_recall,
        meta_test_precision=meta_test_precision,
        meta_test_recall=meta_test_recall,
        signals=signals,
        primary_feature_transformer=primary_feature_transformer,
        meta_feature_transformer=meta_feature_transformer,
    )


def write_signal_csv(signals: pd.DataFrame, path: str | Path) -> Path:
    """
    Write signal DataFrame for ``AfmlSignalStrategy`` consumption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = signals.copy()
    out.index = _as_utc_datetime_index(out.index)
    out.insert(0, "timestamp", out.index.astype(str))
    out.insert(1, "timestamp_ns", [int(ts.value) for ts in out.index])
    out.to_csv(path, index=False)
    return path
