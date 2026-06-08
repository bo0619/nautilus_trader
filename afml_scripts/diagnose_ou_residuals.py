from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_strategies.config_loader import load_afml_data_config  # noqa: E402
from afml_strategies.config_loader import resolve_repo_path  # noqa: E402
from afml_strategies.config_loader import section  # noqa: E402
from afml_strategies.config_loader import string_tuple  # noqa: E402


APPROX_ADF_CRITICAL_VALUES = {
    "n": {"1%": -2.58, "5%": -1.95, "10%": -1.62},
    "c": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
}


@dataclass(frozen=True)
class AdfResult:
    statistic: float
    p_value: float | None
    used_lag: int
    n_observations: int
    critical_values: dict[str, float]
    method: str


@dataclass(frozen=True)
class HalfLifeResult:
    phi: float
    intercept: float
    mean: float
    kappa: float
    half_life_bars: float
    innovation_std: float
    ou_sigma: float


def finite_series(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_symbol(value: str) -> str:
    return value.split(".", maxsplit=1)[0].replace("-PERP", "").replace("_PERP", "")


def latest_input_csv(config: dict[str, Any], symbol: str | None = None) -> Path | None:
    synthetic_config = section(config, "synthetic_data")
    real_config = section(config, "real_data")
    symbol = normalize_symbol(symbol or str(synthetic_config.get("symbol", "BTCUSDT.P")))
    output_dir = resolve_repo_path(real_config.get("output_dir", "binance_scripts/data/afml_bars"))
    symbol_dir = output_dir / symbol
    if not symbol_dir.exists():
        return None
    candidates = [
        path
        for path in symbol_dir.glob("*_DRB.csv")
        if path.is_file() and not path.name.endswith("_summary.csv")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def configured_symbols(config: dict[str, Any], cli_symbols: list[str] | None) -> list[str]:
    if cli_symbols:
        return [normalize_symbol(symbol) for symbol in cli_symbols]
    synthetic_config = section(config, "synthetic_data")
    real_config = section(config, "real_data")
    values = string_tuple(synthetic_config.get("symbols"))
    if not values:
        values = string_tuple(real_config.get("symbols"))
    if not values:
        values = (str(synthetic_config.get("symbol", "BTCUSDT.P")),)
    return [normalize_symbol(symbol) for symbol in values]


def resolve_input_csv(config: dict[str, Any], cli_value: str | None, auto_latest: bool) -> Path:
    if cli_value:
        path = resolve_repo_path(cli_value)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        return path

    synthetic_config = section(config, "synthetic_data")
    configured = synthetic_config.get("input_csv")
    if configured:
        path = resolve_repo_path(configured)
        if path.exists():
            return path
        if not auto_latest:
            raise FileNotFoundError(f"Configured input CSV does not exist: {path}")

    latest = latest_input_csv(config)
    if latest is None:
        raise FileNotFoundError(
            "No input CSV found. Pass --input-csv or set synthetic_data.input_csv.",
        )
    return latest


def input_csv_for_symbol(
    symbol: str,
    config: dict[str, Any],
    cli_value: str | None,
    total_symbols: int,
    auto_latest: bool,
) -> Path:
    if cli_value is not None:
        if total_symbols != 1:
            raise ValueError("--input-csv can only be used with one --symbols value.")
        path = resolve_repo_path(cli_value)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        return path

    synthetic_config = section(config, "synthetic_data")
    input_csvs = synthetic_config.get("input_csvs", {})
    if isinstance(input_csvs, dict) and symbol in input_csvs:
        path = resolve_repo_path(input_csvs[symbol])
        if path.exists():
            return path
        if not auto_latest:
            raise FileNotFoundError(f"Configured input CSV for {symbol} does not exist: {path}")

    latest = latest_input_csv(config, symbol=symbol)
    if latest is None:
        raise FileNotFoundError(f"No input CSV found for {symbol}.")
    return latest


def read_bars(path: Path, price_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {price_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    for column in ("open", "high", "low", "close", "volume", "buy_notional", "sell_notional"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def apply_input_space(df: pd.DataFrame, price_column: str, input_space: str) -> pd.DataFrame:
    if input_space == "price":
        return df
    if input_space == "log_price":
        out = df.copy()
        price = pd.to_numeric(out[price_column], errors="coerce")
        if (price <= 0.0).any():
            raise ValueError("log_price diagnostics require strictly positive prices.")
        out[price_column] = np.log(price)
        return out
    raise ValueError(f"Unsupported input_space: {input_space}")


def bar_timestamps(df: pd.DataFrame) -> pd.Series:
    if "ts_event" in df.columns:
        timestamps = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    elif "ts_event_ns" in df.columns:
        timestamps = pd.to_datetime(df["ts_event_ns"], unit="ns", utc=True, errors="coerce")
    else:
        raise ValueError("Rolling diagnostics require 'ts_event' or 'ts_event_ns' column.")
    if timestamps.notna().sum() == 0:
        raise ValueError("Could not parse any bar timestamps for rolling diagnostics.")
    return timestamps


def rolling_window_bounds(
    timestamps: pd.Series,
    *,
    window_days: float,
    step_days: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if window_days <= 0.0:
        raise ValueError("rolling_window_days must be positive.")
    if step_days <= 0.0:
        raise ValueError("rolling_step_days must be positive.")
    valid = timestamps.dropna()
    if valid.empty:
        return []
    first = valid.min()
    last = valid.max()
    window = pd.Timedelta(days=float(window_days))
    step = pd.Timedelta(days=float(step_days))
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = first
    while start + window <= last:
        bounds.append((start, start + window))
        start = start + step
    final_start = last - window
    if final_start > first and (not bounds or final_start > bounds[-1][0]):
        bounds.append((final_start, last))
    return bounds


def bar_notional(df: pd.DataFrame, price_column: str) -> pd.Series:
    if {"buy_notional", "sell_notional"}.issubset(df.columns):
        return df["buy_notional"].abs() + df["sell_notional"].abs()
    if "volume" not in df.columns:
        raise ValueError("Rolling VWAP needs either buy/sell notional columns or volume.")
    return df[price_column] * df["volume"]


def rolling_vwap(df: pd.DataFrame, price_column: str, window: int) -> pd.Series:
    if window < 2:
        raise ValueError("VWAP window must be >= 2.")
    if "volume" not in df.columns:
        raise ValueError("Rolling VWAP needs a volume column.")
    notional = bar_notional(df, price_column)
    volume = df["volume"].replace(0.0, np.nan)
    return notional.rolling(window=window, min_periods=window).sum() / volume.rolling(
        window=window,
        min_periods=window,
    ).sum()


def ewma_trend(df: pd.DataFrame, price_column: str, span: int) -> pd.Series:
    if span < 2:
        raise ValueError("EWMA span must be >= 2.")
    return df[price_column].ewm(span=span, adjust=False, min_periods=span).mean()


def kalman_local_level(df: pd.DataFrame, price_column: str, q_over_r: float) -> pd.Series:
    if q_over_r <= 0.0:
        raise ValueError("Kalman q_over_r must be positive.")

    observations = np.asarray(df[price_column], dtype=float)
    state = np.empty(len(observations), dtype=float)
    state[:] = np.nan

    finite_index = np.flatnonzero(np.isfinite(observations))
    if finite_index.size == 0:
        return pd.Series(state, index=df.index)

    first = int(finite_index[0])
    x = float(observations[first])
    p = 1.0
    q = float(q_over_r)
    r = 1.0

    for idx in range(first, len(observations)):
        observation = observations[idx]
        p += q
        if math.isfinite(float(observation)):
            gain = p / (p + r)
            x += gain * (float(observation) - x)
            p = (1.0 - gain) * p
            state[idx] = x
        else:
            state[idx] = np.nan

    return pd.Series(state, index=df.index)


def select_adf_max_lag(n_observations: int, configured: int | None) -> int:
    if configured is not None:
        return max(0, int(configured))
    if n_observations < 100:
        return 0
    lag = math.floor(12.0 * (n_observations / 100.0) ** 0.25)
    return max(0, min(lag, 32))


def fit_adf_ols(x: np.ndarray, lag: int, regression: str) -> tuple[float, float, int, float]:
    diff = np.diff(x)
    if len(diff) <= lag + 2:
        raise ValueError("Series is too short for requested ADF lag.")

    y = diff[lag:]
    y_lag = x[lag:-1]
    columns = []
    if regression == "c":
        columns.append(np.ones_like(y))
    elif regression != "n":
        raise ValueError("Only ADF regression modes 'c' and 'n' are supported.")
    columns.append(y_lag)
    for offset in range(1, lag + 1):
        columns.append(diff[lag - offset : -offset])
    design = np.column_stack(columns)

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    sigma2 = float((residual @ residual) / dof)
    xtx_inv = np.linalg.pinv(design.T @ design)
    y_lag_coef_index = 1 if regression == "c" else 0
    std_err = math.sqrt(max(sigma2 * float(xtx_inv[y_lag_coef_index, y_lag_coef_index]), 0.0))
    statistic = float(beta[y_lag_coef_index] / std_err) if std_err > 0.0 else float("nan")
    rss = float(residual @ residual)
    aic = len(y) * math.log(max(rss / len(y), np.finfo(float).tiny)) + 2.0 * design.shape[1]
    return statistic, aic, len(y), rss


def fallback_adf(x: np.ndarray, max_lag: int, regression: str) -> AdfResult:
    max_lag = min(max_lag, max(len(x) // 2 - 2, 0))
    best: tuple[float, int, int, float] | None = None
    for lag in range(max_lag + 1):
        statistic, aic, n_observations, _ = fit_adf_ols(x, lag, regression)
        if best is None or aic < best[0]:
            best = (aic, lag, n_observations, statistic)
    if best is None:
        raise ValueError("Could not fit fallback ADF regression.")
    _, used_lag, n_observations, statistic = best
    return AdfResult(
        statistic=statistic,
        p_value=None,
        used_lag=used_lag,
        n_observations=n_observations,
        critical_values=APPROX_ADF_CRITICAL_VALUES[regression],
        method="ols_aic_approx_critical_values",
    )


def adf_test(x: np.ndarray, max_lag: int, regression: str) -> AdfResult:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return fallback_adf(x, max_lag=max_lag, regression=regression)

    statistic, p_value, used_lag, n_observations, critical_values, _ = adfuller(
        x,
        maxlag=max_lag,
        regression=regression,
        autolag="AIC",
    )
    return AdfResult(
        statistic=float(statistic),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_observations=int(n_observations),
        critical_values={key: float(value) for key, value in critical_values.items()},
        method="statsmodels_adfuller_aic",
    )


def hurst_exponent(x: np.ndarray, lag_min: int, lag_max: int) -> tuple[float, int, int]:
    lag_min = max(2, int(lag_min))
    lag_max = min(int(lag_max), max(len(x) // 2, lag_min))
    lags: list[int] = []
    tau: list[float] = []
    for lag in range(lag_min, lag_max + 1):
        diff = x[lag:] - x[:-lag]
        scale = float(np.std(diff, ddof=1))
        if math.isfinite(scale) and scale > 0.0:
            lags.append(lag)
            tau.append(scale)
    if len(lags) < 2:
        return float("nan"), lag_min, lag_max
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(slope), int(lags[0]), int(lags[-1])


def estimate_half_life(x: np.ndarray) -> HalfLifeResult:
    y = x[1:]
    y_lag = x[:-1]
    design = np.column_stack([np.ones_like(y), y_lag])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(beta[0])
    phi = float(beta[1])
    residual = y - design @ beta
    innovation_std = float(np.std(residual, ddof=1))

    if 0.0 < phi < 1.0:
        kappa = -math.log(phi)
        half_life = math.log(2.0) / kappa
        mean = intercept / (1.0 - phi)
        ou_sigma = innovation_std * math.sqrt(2.0 * kappa / max(1.0 - phi * phi, 1e-12))
    else:
        kappa = float("nan")
        half_life = float("nan")
        mean = float("nan")
        ou_sigma = float("nan")

    return HalfLifeResult(
        phi=phi,
        intercept=intercept,
        mean=mean,
        kappa=kappa,
        half_life_bars=half_life,
        innovation_std=innovation_std,
        ou_sigma=ou_sigma,
    )


def residual_frame(
    df: pd.DataFrame,
    price_column: str,
    fair_value: pd.Series,
    fair_value_shift: int,
) -> pd.DataFrame:
    shifted = fair_value.shift(fair_value_shift) if fair_value_shift else fair_value
    result = pd.DataFrame(
        {
            "price": df[price_column],
            "fair_value": shifted,
            "residual": df[price_column] - shifted,
        },
        index=df.index,
    )
    if "ts_event" in df.columns:
        result.insert(0, "ts_event", df["ts_event"])
    if "ts_event_ns" in df.columns:
        result.insert(1 if "ts_event" in result.columns else 0, "ts_event_ns", df["ts_event_ns"])
    return result.dropna(subset=["price", "fair_value", "residual"])


def evaluate_residuals(
    residuals: np.ndarray,
    diagnostics_config: dict[str, Any],
    adf_regression: str,
) -> dict[str, Any]:
    x = finite_series(residuals)
    x = x - float(np.mean(x))
    if len(x) < int(diagnostics_config.get("min_observations", 100)):
        raise ValueError(f"Not enough finite residual observations: {len(x)}")

    max_adf_lag = diagnostics_config.get("max_adf_lag")
    selected_max_lag = select_adf_max_lag(
        len(x),
        int(max_adf_lag) if max_adf_lag is not None else None,
    )
    adf = adf_test(x, max_lag=selected_max_lag, regression=adf_regression)
    hurst, hurst_lag_min, hurst_lag_max = hurst_exponent(
        x,
        int(diagnostics_config.get("hurst_lag_min", 2)),
        int(diagnostics_config.get("hurst_lag_max", 64)),
    )
    half_life = estimate_half_life(x)

    adf_alpha = float(diagnostics_config.get("adf_alpha", 0.05))
    adf_critical_key = str(diagnostics_config.get("adf_critical_key", "5%"))
    if adf.p_value is not None:
        adf_pass = adf.p_value < adf_alpha
    else:
        critical_value = adf.critical_values.get(adf_critical_key, adf.critical_values["5%"])
        adf_pass = adf.statistic < critical_value

    hurst_threshold = float(diagnostics_config.get("hurst_threshold", 0.5))
    hurst_pass = math.isfinite(hurst) and hurst < hurst_threshold

    min_half_life = float(diagnostics_config.get("min_half_life_bars", 1.0))
    max_half_life = float(diagnostics_config.get("max_half_life_bars", 150.0))
    half_life_pass = (
        math.isfinite(half_life.half_life_bars)
        and min_half_life <= half_life.half_life_bars <= max_half_life
    )

    return {
        "n_observations": len(x),
        "residual_mean": float(np.mean(x)),
        "residual_std": float(np.std(x, ddof=1)),
        "adf": {
            "statistic": adf.statistic,
            "p_value": adf.p_value,
            "used_lag": adf.used_lag,
            "max_lag": selected_max_lag,
            "n_observations": adf.n_observations,
            "critical_values": adf.critical_values,
            "method": adf.method,
            "pass": bool(adf_pass),
        },
        "hurst": {
            "value": hurst,
            "lag_min": hurst_lag_min,
            "lag_max": hurst_lag_max,
            "threshold": hurst_threshold,
            "pass": bool(hurst_pass),
        },
        "half_life": {
            "phi": half_life.phi,
            "intercept": half_life.intercept,
            "mean": half_life.mean,
            "kappa": half_life.kappa,
            "half_life_bars": half_life.half_life_bars,
            "innovation_std": half_life.innovation_std,
            "ou_sigma": half_life.ou_sigma,
            "min_half_life_bars": min_half_life,
            "max_half_life_bars": max_half_life,
            "pass": bool(half_life_pass),
        },
        "pass_count": int(adf_pass) + int(hurst_pass) + int(half_life_pass),
        "pass": bool(adf_pass and hurst_pass and half_life_pass),
    }


def candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if not args.only_kalman:
        for window in args.vwap_windows:
            specs.append({"kind": "rolling_vwap", "name": f"rolling_vwap_{window}", "window": window})
        for span in args.ewma_spans:
            specs.append({"kind": "ewma_trend", "name": f"ewma_trend_{span}", "span": span})
    for q_over_r in args.kalman_q_over_r:
        label = str(q_over_r).replace(".", "p")
        specs.append(
            {
                "kind": "kalman_local_level",
                "name": f"kalman_local_level_qr_{label}",
                "q_over_r": q_over_r,
            },
        )
    return specs


def build_fair_value(df: pd.DataFrame, price_column: str, spec: dict[str, Any]) -> pd.Series:
    kind = spec["kind"]
    if kind == "rolling_vwap":
        return rolling_vwap(df, price_column=price_column, window=int(spec["window"]))
    if kind == "ewma_trend":
        return ewma_trend(df, price_column=price_column, span=int(spec["span"]))
    if kind == "kalman_local_level":
        return kalman_local_level(df, price_column=price_column, q_over_r=float(spec["q_over_r"]))
    raise ValueError(f"Unsupported fair value kind: {kind}")


def evaluate_specs(
    df: pd.DataFrame,
    *,
    specs: list[dict[str, Any]],
    diagnostics_config: dict[str, Any],
    price_column: str,
    fair_value_shift: int,
    adf_regression: str,
    symbol: str | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    results: list[dict[str, Any]] = []
    residuals_by_name: dict[str, pd.DataFrame] = {}
    for spec in specs:
        fair_value = build_fair_value(df, price_column=price_column, spec=spec)
        residuals = residual_frame(
            df,
            price_column=price_column,
            fair_value=fair_value,
            fair_value_shift=fair_value_shift,
        )
        diagnostics = evaluate_residuals(
            residuals["residual"].to_numpy(dtype=float),
            diagnostics_config=diagnostics_config,
            adf_regression=adf_regression,
        )
        params = {key: value for key, value in spec.items() if key not in {"kind", "name"}}
        result = {
            "symbol": symbol,
            "window_start": window_start.isoformat() if window_start is not None else None,
            "window_end": window_end.isoformat() if window_end is not None else None,
            "window_days": (
                float((window_end - window_start) / pd.Timedelta(days=1))
                if window_start is not None and window_end is not None
                else None
            ),
            "name": spec["name"],
            "kind": spec["kind"],
            "params": params,
            "diagnostics": diagnostics,
        }
        results.append(result)
        residuals_by_name[spec["name"]] = residuals
    return results, residuals_by_name


def flatten_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result["diagnostics"]
    return {
        "symbol": result.get("symbol"),
        "window_start": result.get("window_start"),
        "window_end": result.get("window_end"),
        "window_days": result.get("window_days"),
        "name": result["name"],
        "kind": result["kind"],
        "params": json.dumps(result["params"], sort_keys=True),
        "pass": diagnostics["pass"],
        "pass_count": diagnostics["pass_count"],
        "n_observations": diagnostics["n_observations"],
        "adf_statistic": diagnostics["adf"]["statistic"],
        "adf_p_value": diagnostics["adf"]["p_value"],
        "adf_used_lag": diagnostics["adf"]["used_lag"],
        "adf_pass": diagnostics["adf"]["pass"],
        "hurst": diagnostics["hurst"]["value"],
        "hurst_pass": diagnostics["hurst"]["pass"],
        "half_life_bars": diagnostics["half_life"]["half_life_bars"],
        "phi": diagnostics["half_life"]["phi"],
        "kappa": diagnostics["half_life"]["kappa"],
        "ou_mean": diagnostics["half_life"]["mean"],
        "ou_sigma": diagnostics["half_life"]["ou_sigma"],
        "half_life_pass": diagnostics["half_life"]["pass"],
        "residual_std": diagnostics["residual_std"],
    }


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_key(result: dict[str, Any]) -> tuple[bool, int, float, float, float]:
        diagnostics = result["diagnostics"]
        half_life = diagnostics["half_life"]["half_life_bars"]
        half_life_rank = half_life if math.isfinite(half_life) else float("-inf")
        hurst = diagnostics["hurst"]["value"]
        hurst_threshold = diagnostics["hurst"]["threshold"]
        hurst_margin = hurst_threshold - hurst if math.isfinite(hurst) else float("-inf")
        adf_statistic = diagnostics["adf"]["statistic"]
        adf_strength = -adf_statistic if math.isfinite(adf_statistic) else float("-inf")
        return (
            diagnostics["pass"],
            diagnostics["pass_count"],
            half_life_rank,
            hurst_margin,
            adf_strength,
        )

    return sorted(
        results,
        key=rank_key,
        reverse=True,
    )


def write_residual_series(
    output_dir: Path,
    residuals_by_name: dict[str, pd.DataFrame],
    limit: int | None,
) -> None:
    series_dir = output_dir / "residual_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in residuals_by_name.items():
        if limit is not None and limit > 0:
            frame = frame.tail(limit)
        frame.to_csv(series_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether AFML bar residuals are suitable for OU residual synthetic data."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to AFML data config JSON.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--input-csv", default=None, help="AFML bars CSV. Defaults to config.")
    parser.add_argument(
        "--no-auto-latest-input",
        action="store_true",
        help="Fail if config input_csv is missing instead of using the latest symbol CSV.",
    )
    parser.add_argument("--output-dir", default=None, help="Diagnostics output directory.")
    parser.add_argument("--price-column", default="close")
    parser.add_argument(
        "--input-space",
        choices=["price", "log_price"],
        default=None,
        help="Transform price column before fair-value/residual diagnostics. Defaults to config.",
    )
    parser.add_argument("--fair-value-shift", type=int, default=0)
    parser.add_argument("--vwap-windows", type=int, nargs="+", default=[20, 50, 100, 250])
    parser.add_argument("--ewma-spans", type=int, nargs="+", default=[20, 50, 150, 250])
    parser.add_argument(
        "--kalman-q-over-r",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--only-kalman", action="store_true", help="Skip VWAP/EWMA candidates.")
    parser.add_argument(
        "--rolling-window-days",
        type=float,
        default=None,
        help="Run diagnostics on rolling calendar windows of this length.",
    )
    parser.add_argument(
        "--rolling-step-days",
        type=float,
        default=30.0,
        help="Calendar-day step between rolling diagnostic windows.",
    )
    parser.add_argument("--max-adf-lag", type=int, default=None)
    parser.add_argument("--adf-regression", choices=["c", "n"], default="c")
    parser.add_argument("--adf-alpha", type=float, default=None)
    parser.add_argument("--hurst-lag-min", type=int, default=None)
    parser.add_argument("--hurst-lag-max", type=int, default=None)
    parser.add_argument("--hurst-threshold", type=float, default=None)
    parser.add_argument("--min-half-life-bars", type=float, default=None)
    parser.add_argument("--max-half-life-bars", type=float, default=None)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--write-residual-series", action="store_true")
    parser.add_argument("--residual-series-tail", type=int, default=5000)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def diagnostics_config_from_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    synthetic_config = section(config, "synthetic_data")
    diagnostics_config = dict(section(synthetic_config, "diagnostics"))
    overrides = {
        "max_adf_lag": args.max_adf_lag,
        "adf_alpha": args.adf_alpha,
        "hurst_lag_min": args.hurst_lag_min,
        "hurst_lag_max": args.hurst_lag_max,
        "hurst_threshold": args.hurst_threshold,
        "min_half_life_bars": args.min_half_life_bars,
        "max_half_life_bars": args.max_half_life_bars,
        "min_observations": args.min_observations,
    }
    diagnostics_config.update({key: value for key, value in overrides.items() if value is not None})
    diagnostics_config.setdefault("min_observations", args.min_observations)
    return diagnostics_config


def main() -> None:  # noqa: C901
    args = parse_args()
    config = load_afml_data_config(args.config)
    synthetic_config = section(config, "synthetic_data")
    fair_value_config = section(synthetic_config, "fair_value")
    input_space = args.input_space or str(fair_value_config.get("input_space", "price"))
    diagnostics_config = diagnostics_config_from_args(config, args)
    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir
        else resolve_repo_path(synthetic_config.get("output_dir", "afml_scripts/output/ou_synthetic"))
        / "diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs(args)

    if args.rolling_window_days is not None or args.symbols is not None:
        symbols = configured_symbols(config, args.symbols)
        all_results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        inputs: dict[str, str] = {}
        for symbol in symbols:
            input_csv = input_csv_for_symbol(
                symbol,
                config,
                args.input_csv,
                total_symbols=len(symbols),
                auto_latest=not args.no_auto_latest_input,
            )
            inputs[symbol] = str(input_csv)
            raw_df = read_bars(input_csv, price_column=args.price_column)
            timestamps = bar_timestamps(raw_df)
            df = apply_input_space(raw_df, args.price_column, input_space)
            if args.rolling_window_days is None:
                bounds = [(timestamps.min(), timestamps.max())]
            else:
                bounds = rolling_window_bounds(
                    timestamps,
                    window_days=args.rolling_window_days,
                    step_days=args.rolling_step_days,
                )
            for window_start, window_end in bounds:
                mask = (timestamps >= window_start) & (timestamps <= window_end)
                window_df = df.loc[mask].reset_index(drop=True)
                try:
                    window_results, _ = evaluate_specs(
                        window_df,
                        specs=specs,
                        diagnostics_config=diagnostics_config,
                        price_column=args.price_column,
                        fair_value_shift=args.fair_value_shift,
                        adf_regression=args.adf_regression,
                        symbol=symbol,
                        window_start=window_start,
                        window_end=window_end,
                    )
                except ValueError as exc:
                    failures.append(
                        {
                            "symbol": symbol,
                            "window_start": window_start.isoformat(),
                            "window_end": window_end.isoformat(),
                            "n_bars": len(window_df),
                            "error": str(exc),
                        },
                    )
                    continue
                all_results.extend(window_results)

        rows = [flatten_summary_row(result) for result in all_results]
        summary_csv = output_dir / "ou_residual_rolling_diagnostics.csv"
        rows_frame = pd.DataFrame(rows)
        rows_frame.to_csv(summary_csv, index=False)
        summary_json = output_dir / "ou_residual_rolling_diagnostics_summary.json"
        symbol_summary = []
        if not rows_frame.empty:
            for symbol, group in rows_frame.groupby("symbol", sort=True):
                symbol_summary.append(
                    {
                        "symbol": symbol,
                        "n_windows": len(group),
                        "pass_rate": float(group["pass"].mean()),
                        "adf_pass_rate": float(group["adf_pass"].mean()),
                        "hurst_pass_rate": float(group["hurst_pass"].mean()),
                        "half_life_pass_rate": float(group["half_life_pass"].mean()),
                        "median_adf_statistic": float(group["adf_statistic"].median()),
                        "median_hurst": float(group["hurst"].median()),
                        "median_half_life_bars": float(group["half_life_bars"].median()),
                    },
                )
        payload = {
            "inputs": inputs,
            "output_dir": str(output_dir),
            "price_column": args.price_column,
            "input_space": input_space,
            "fair_value_shift": args.fair_value_shift,
            "rolling_window_days": args.rolling_window_days,
            "rolling_step_days": args.rolling_step_days,
            "rank_rule": "rolling windows pass ADF, Hurst, and OU half-life checks",
            "diagnostics_config": diagnostics_config,
            "specs": specs,
            "symbol_summary": symbol_summary,
            "failures": failures,
            "results": all_results,
        }
        summary_json.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")

        print(f"symbols={', '.join(symbols)}")
        print(f"wrote summary -> {summary_json}")
        print(f"wrote table   -> {summary_csv}")
        if failures:
            print(f"skipped windows -> {len(failures)}")
        print()
        print("rolling OU diagnostics by symbol:")
        for item in symbol_summary:
            print(
                f"{item['symbol']}: windows={item['n_windows']} "
                f"pass_rate={item['pass_rate']:.2%} "
                f"adf={item['adf_pass_rate']:.2%} "
                f"hurst={item['hurst_pass_rate']:.2%} "
                f"half_life={item['half_life_pass_rate']:.2%} "
                f"median_half_life={item['median_half_life_bars']:.2f}",
            )
        return

    input_csv = resolve_input_csv(
        config,
        args.input_csv,
        auto_latest=not args.no_auto_latest_input,
    )
    df = read_bars(input_csv, price_column=args.price_column)
    df = apply_input_space(df, args.price_column, input_space)
    results, residuals_by_name = evaluate_specs(
        df,
        specs=specs,
        diagnostics_config=diagnostics_config,
        price_column=args.price_column,
        fair_value_shift=args.fair_value_shift,
        adf_regression=args.adf_regression,
        symbol=normalize_symbol(str(synthetic_config.get("symbol", "BTCUSDT.P"))),
    )

    sorted_results = sort_results(results)
    rows = [flatten_summary_row(result) for result in sorted_results]
    summary_csv = output_dir / "ou_residual_diagnostics.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    summary_json = output_dir / "ou_residual_diagnostics_summary.json"
    payload = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "price_column": args.price_column,
        "input_space": input_space,
        "fair_value_shift": args.fair_value_shift,
        "n_bars": len(df),
        "rank_rule": "pass all checks, then prefer longer valid half-life residuals",
        "diagnostics_config": diagnostics_config,
        "results": sorted_results,
        "best": sorted_results[0] if sorted_results else None,
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")

    if args.write_residual_series:
        write_residual_series(output_dir, residuals_by_name, limit=args.residual_series_tail)

    print(f"input_csv={input_csv}")
    print(f"wrote summary -> {summary_json}")
    print(f"wrote table   -> {summary_csv}")
    print()
    print("top residual OU diagnostics:")
    for row in rows[: args.top]:
        adf_p = row["adf_p_value"]
        adf_p_text = "n/a" if adf_p is None else f"{adf_p:.4g}"
        half_life = safe_float(row["half_life_bars"])
        half_life_text = "n/a" if half_life is None else f"{half_life:.2f}"
        print(
            f"{row['name']}: pass={row['pass']} pass_count={row['pass_count']}/3 "
            f"adf={row['adf_statistic']:.3f} p={adf_p_text} "
            f"hurst={row['hurst']:.3f} half_life={half_life_text}",
        )


if __name__ == "__main__":
    main()
