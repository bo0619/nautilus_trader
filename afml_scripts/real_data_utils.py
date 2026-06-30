from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from afml_strategies.config_loader import resolve_repo_path
from afml_strategies.config_loader import section
from afml_strategies.config_loader import string_tuple


def normalize_symbol(value: str) -> str:
    return value.strip().upper().replace("/", "").replace("-", "").replace(".P", "")


def configured_symbols(config: dict[str, Any], cli_symbols: list[str] | None) -> list[str]:
    if cli_symbols:
        return [normalize_symbol(symbol) for symbol in cli_symbols]
    real_config = section(config, "real_data")
    values = string_tuple(real_config.get("symbols"))
    if not values:
        raise ValueError("real_data.symbols must contain at least one symbol")
    return [normalize_symbol(symbol) for symbol in values]


def latest_input_csv(symbol: str, real_config: dict[str, Any]) -> Path | None:
    output_dir = resolve_repo_path(
        real_config.get("output_dir", "binance_scripts/data/afml_bars"),
    )
    symbol_dir = output_dir / normalize_symbol(symbol)
    if not symbol_dir.exists():
        return None
    candidates = [path for path in symbol_dir.glob("*_DRB.csv") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _validate_input_symbol(path: Path, symbol: str) -> None:
    expected = normalize_symbol(symbol)
    filename_symbol = normalize_symbol(path.name.split("_", maxsplit=1)[0])
    if filename_symbol != expected:
        raise ValueError(
            f"Configured symbol {expected} does not match AFML input CSV {path.name}",
        )


def input_csv_for_symbol(
    symbol: str,
    config: dict[str, Any],
    cli_input_csv: str | None,
    total_symbols: int,
) -> Path:
    if cli_input_csv is not None:
        if total_symbols != 1:
            raise ValueError("--input-csv can only be used with one --symbols value.")
        path = resolve_repo_path(cli_input_csv)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        _validate_input_symbol(path, symbol)
        return path

    real_config = section(config, "real_data")
    input_csvs = real_config.get("input_csvs", {})
    if isinstance(input_csvs, dict) and symbol in input_csvs:
        path = resolve_repo_path(input_csvs[symbol])
        if not path.exists():
            raise FileNotFoundError(f"Configured input CSV for {symbol} does not exist: {path}")
        _validate_input_symbol(path, symbol)
        return path

    latest = latest_input_csv(symbol, real_config)
    if latest is None:
        raise FileNotFoundError(f"No AFML DRB CSV found for {symbol}.")
    _validate_input_symbol(latest, symbol)
    return latest


def read_real_bars(path: Path, price_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if price_column not in frame.columns:
        raise ValueError(f"Missing price column {price_column!r} in {path}.")
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    frame = frame.loc[frame[price_column] > 0.0].reset_index(drop=True)
    if len(frame) < 1_000:
        raise ValueError(f"Need at least 1,000 real bars for AFML training: {path}")
    return frame


def kalman_local_level(
    frame: pd.DataFrame,
    price_column: str,
    q_over_r: float,
) -> pd.Series:
    if q_over_r <= 0.0:
        raise ValueError("Kalman q_over_r must be positive.")

    observations = np.asarray(frame[price_column], dtype=float)
    state = np.full(len(observations), np.nan, dtype=float)
    finite_index = np.flatnonzero(np.isfinite(observations))
    if finite_index.size == 0:
        return pd.Series(state, index=frame.index)

    first = int(finite_index[0])
    estimate = float(observations[first])
    covariance = 1.0
    process_variance = float(q_over_r)
    observation_variance = 1.0

    for idx in range(first, len(observations)):
        observation = observations[idx]
        covariance += process_variance
        if math.isfinite(float(observation)):
            gain = covariance / (covariance + observation_variance)
            estimate += gain * (float(observation) - estimate)
            covariance = (1.0 - gain) * covariance
            state[idx] = estimate

    return pd.Series(state, index=frame.index)


def _rolling_slope_r2(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
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


def _rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=1)
    return ((series - mean) / std.replace(0.0, np.nan)).to_numpy(dtype=float)


def _signed_notional(frame: pd.DataFrame) -> np.ndarray:
    if "signed_notional" in frame.columns:
        return (
            pd.to_numeric(frame["signed_notional"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    if {"buy_notional", "sell_notional"}.issubset(frame.columns):
        buy = pd.to_numeric(frame["buy_notional"], errors="coerce").fillna(0.0)
        sell = pd.to_numeric(frame["sell_notional"], errors="coerce").fillna(0.0)
        return (buy - sell).to_numpy(dtype=float)
    if {"buy_ticks", "sell_ticks"}.issubset(frame.columns):
        buy = pd.to_numeric(frame["buy_ticks"], errors="coerce").fillna(0.0)
        sell = pd.to_numeric(frame["sell_ticks"], errors="coerce").fillna(0.0)
        return (buy - sell).to_numpy(dtype=float)
    raise ValueError(
        "CVD requires signed_notional, buy/sell_notional, or buy/sell_ticks columns.",
    )


def event_state_features(
    frame: pd.DataFrame,
    price_column: str,
    fair_value_config: dict[str, Any],
    event_config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    close = pd.to_numeric(frame[price_column], errors="coerce").to_numpy(dtype=float)
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

    micro_slope, _ = _rolling_slope_r2(log_trend, micro_window)
    _, trend_r2 = _rolling_slope_r2(log_trend, trend_window)

    cvd_window = int(event_config.get("cvd_z_window_bars", 200))
    cvd_increment = _signed_notional(frame)
    cvd = np.cumsum(cvd_increment)
    cvd_z = _rolling_zscore(cvd, cvd_window)
    signed_notional_z = _rolling_zscore(cvd_increment, cvd_window)
    residual_z = _rolling_zscore(residual, cvd_window)
    returns = pd.Series(log_close).diff()
    realized_vol_20 = (
        returns.rolling(window=20, min_periods=20).std(ddof=1).to_numpy(dtype=float)
    )

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


def round_trip_cost_from_execution_config(config: dict[str, Any]) -> float:
    required = {
        "slippage_per_side_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "fee_liquidity",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"execution_costs missing explicit cost fields: {missing}")
    slippage = float(config["slippage_per_side_rate"])
    liquidity = str(config["fee_liquidity"])
    if liquidity in {"maker", "taker"}:
        fee = float(config[f"{liquidity}_fee_rate"]) * 2.0
    elif liquidity in {"maker_taker", "maker_entry_taker_exit"}:
        fee = float(config["maker_fee_rate"]) + float(config["taker_fee_rate"])
    else:
        raise ValueError(
            "execution_costs.fee_liquidity must be 'maker', 'taker', "
            "'maker_taker', or 'maker_entry_taker_exit'",
        )
    return 2.0 * slippage + fee
