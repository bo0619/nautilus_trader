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
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_scripts.diagnose_ou_residuals import kalman_local_level  # noqa: E402
from afml_strategies.config_loader import load_afml_data_config  # noqa: E402
from afml_strategies.config_loader import resolve_repo_path  # noqa: E402
from afml_strategies.config_loader import section  # noqa: E402
from afml_strategies.config_loader import string_tuple  # noqa: E402


@dataclass(frozen=True)
class JointTrendOuModel:
    trend_drift: float
    trend_std: float
    ou_intercept: float
    ou_phi: float
    ou_mean: float
    ou_kappa: float
    ou_half_life_bars: float
    ou_innovation_std: float
    ou_sigma: float
    innovation_covariance: np.ndarray
    observed_log_return_volatility: float
    observed_ewma_1bar_log_return_volatility: float
    observed_horizon_volatility: float
    last_log_trend: float
    last_log_residual: float


def normalize_symbol(value: str) -> str:
    return value.split(".", maxsplit=1)[0].replace("-PERP", "").replace("_PERP", "")


def q_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


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


def deterministic_symbol_seed(seed: int, symbol: str) -> int:
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(symbol))
    return (int(seed) + offset) % (2**32 - 1)


def grid_values(config: dict[str, Any]) -> np.ndarray:
    start = float(config["start"])
    stop = float(config["stop"])
    step = float(config["step"])
    count = round((stop - start) / step) + 1
    return start + step * np.arange(count, dtype=float)


def latest_input_csv(symbol: str, real_config: dict[str, Any]) -> Path | None:
    output_dir = resolve_repo_path(real_config.get("output_dir", "binance_scripts/data/afml_bars"))
    symbol_dir = output_dir / symbol
    if not symbol_dir.exists():
        return None
    candidates = [path for path in symbol_dir.glob("*_DRB.csv") if path.is_file()]
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
        return path

    synthetic_config = section(config, "synthetic_data")
    real_config = section(config, "real_data")
    input_csvs = synthetic_config.get("input_csvs", {})
    if isinstance(input_csvs, dict) and symbol in input_csvs:
        path = resolve_repo_path(input_csvs[symbol])
        if path.exists():
            return path
        raise FileNotFoundError(f"Configured input CSV for {symbol} does not exist: {path}")

    latest = latest_input_csv(symbol, real_config)
    if latest is None:
        raise FileNotFoundError(f"No AFML DRB CSV found for {symbol}.")
    return latest


def load_strategy_config(config: dict[str, Any], cli_path: str | None = None) -> dict[str, Any]:
    synthetic_config = section(config, "synthetic_data")
    strategy_path_value = cli_path or synthetic_config.get("strategy_config")
    if not strategy_path_value:
        legacy = {
            "barrier_optimization": section(synthetic_config, "barrier_optimization"),
        }
        return legacy

    strategy_path = resolve_repo_path(strategy_path_value)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy config does not exist: {strategy_path}")
    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Strategy config root must be a JSON object: {strategy_path}")
    payload["_strategy_config_path"] = str(strategy_path)
    return payload


def finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def read_real_bars(path: Path, price_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if price_column not in df.columns:
        raise ValueError(f"Missing price column {price_column!r} in {path}.")
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")
    df = df.loc[df[price_column] > 0.0].reset_index(drop=True)
    if len(df) < 1_000:
        raise ValueError(f"Need at least 1,000 real bars to calibrate Monte Carlo model: {path}")
    return df


def bar_timestamps(df: pd.DataFrame) -> pd.Series:
    if "ts_event" in df.columns:
        timestamps = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    elif "ts_event_ns" in df.columns:
        timestamps = pd.to_datetime(df["ts_event_ns"], unit="ns", utc=True, errors="coerce")
    else:
        raise ValueError("Rolling generation requires 'ts_event' or 'ts_event_ns' column.")
    if timestamps.notna().sum() == 0:
        raise ValueError("Could not parse any bar timestamps for rolling generation.")
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
        start += step
    final_start = last - window
    if final_start > first and (not bounds or final_start > bounds[-1][0]):
        bounds.append((final_start, last))
    return bounds


def timestamp_label(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def horizon_log_return_volatility(log_price: np.ndarray, horizon: int) -> float:
    horizon = max(1, min(int(horizon), len(log_price) - 1))
    returns = log_price[horizon:] - log_price[:-horizon]
    returns = finite_array(returns)
    if len(returns) < 2:
        return float("nan")
    return float(np.std(returns, ddof=1))


def ewma_1bar_log_return_volatility(log_price: np.ndarray, span: int = 80) -> float:
    if span < 1:
        raise ValueError("span must be positive")
    returns = pd.Series(np.diff(log_price))
    if returns.shape[0] < 2:
        return float("nan")
    volatility = returns.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).std()
    value = volatility.dropna()
    if value.empty:
        return float(np.std(finite_array(returns.to_numpy(dtype=float)), ddof=1))
    return float(value.iloc[-1])


def fit_joint_trend_ou_model(
    df: pd.DataFrame,
    price_column: str,
    fair_value_config: dict[str, Any],
    trend_config: dict[str, Any],
    ou_config: dict[str, Any],
    barrier_config: dict[str, Any],
) -> JointTrendOuModel:
    log_price = np.log(df[price_column].to_numpy(dtype=float))
    trend_input = pd.DataFrame({"log_price": log_price})
    log_trend = kalman_local_level(
        trend_input,
        price_column="log_price",
        q_over_r=float(fair_value_config.get("q_over_r", 0.001)),
    ).to_numpy(dtype=float)
    valid = np.isfinite(log_price) & np.isfinite(log_trend)
    log_price = log_price[valid]
    log_trend = log_trend[valid]
    log_residual = log_price - log_trend

    trend_increment = np.diff(log_trend)
    residual_next = log_residual[1:]
    residual_lag = log_residual[:-1]
    design = np.column_stack([np.ones_like(residual_lag), residual_lag])
    beta, *_ = np.linalg.lstsq(design, residual_next, rcond=None)
    ou_intercept = float(beta[0])
    ou_phi = float(beta[1])
    if not 0.0 < ou_phi < 1.0:
        raise ValueError(f"OU residual phi must be in (0, 1), got {ou_phi}.")

    ou_innovation = residual_next - design @ beta
    trend_drift = float(np.mean(trend_increment)) * float(trend_config.get("drift_shrinkage", 1.0))
    trend_centered = trend_increment - float(np.mean(trend_increment))
    residual_centered = ou_innovation - float(np.mean(ou_innovation))
    trend_scale = float(trend_config.get("volatility_scale", 1.0))
    residual_scale = float(ou_config.get("residual_volatility_scale", 1.0))
    innovations = np.column_stack([trend_centered * trend_scale, residual_centered * residual_scale])
    covariance = np.cov(innovations, rowvar=False)
    covariance = nearest_positive_semidefinite(covariance)

    ou_mean = ou_intercept / (1.0 - ou_phi)
    ou_kappa = -math.log(ou_phi)
    ou_half_life_bars = math.log(2.0) / ou_kappa
    ou_innovation_std = float(np.std(ou_innovation, ddof=1)) * residual_scale
    ou_sigma = ou_innovation_std * math.sqrt(2.0 * ou_kappa / max(1.0 - ou_phi * ou_phi, 1e-12))
    log_return = np.diff(log_price)
    volatility_horizon = int(barrier_config.get("volatility_horizon_bars", 100))
    volatility_ewma_span = int(barrier_config.get("volatility_ewma_span", 80))

    return JointTrendOuModel(
        trend_drift=trend_drift,
        trend_std=float(np.std(trend_increment, ddof=1)) * trend_scale,
        ou_intercept=ou_intercept,
        ou_phi=ou_phi,
        ou_mean=ou_mean,
        ou_kappa=ou_kappa,
        ou_half_life_bars=ou_half_life_bars,
        ou_innovation_std=ou_innovation_std,
        ou_sigma=ou_sigma,
        innovation_covariance=covariance,
        observed_log_return_volatility=float(np.std(log_return, ddof=1)),
        observed_ewma_1bar_log_return_volatility=ewma_1bar_log_return_volatility(
            log_price,
            span=volatility_ewma_span,
        ),
        observed_horizon_volatility=horizon_log_return_volatility(log_price, volatility_horizon),
        last_log_trend=float(log_trend[-1]),
        last_log_residual=float(log_residual[-1]),
    )


def nearest_positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-14)
    return (vectors * values) @ vectors.T


def simulate_paths(
    model: JointTrendOuModel,
    n_paths: int,
    horizon: int,
    seed_policy: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    trend = np.empty((n_paths, horizon + 1), dtype=float)
    residual = np.empty((n_paths, horizon + 1), dtype=float)
    trend[:, 0] = model.last_log_trend
    if seed_policy == "stationary_distribution":
        stationary_std = model.ou_innovation_std / math.sqrt(max(1.0 - model.ou_phi**2, 1e-12))
        residual[:, 0] = model.ou_mean + rng.normal(0.0, stationary_std, size=n_paths)
    elif seed_policy == "last_observed_residual":
        residual[:, 0] = model.last_log_residual
    else:
        raise ValueError(f"Unsupported seed_policy: {seed_policy}")

    innovation_mean = np.array([model.trend_drift, 0.0], dtype=float)
    innovations = rng.multivariate_normal(
        innovation_mean,
        model.innovation_covariance,
        size=(n_paths, horizon),
        check_valid="raise",
    )
    for step in range(1, horizon + 1):
        trend[:, step] = trend[:, step - 1] + innovations[:, step - 1, 0]
        residual[:, step] = (
            model.ou_mean
            + model.ou_phi * (residual[:, step - 1] - model.ou_mean)
            + innovations[:, step - 1, 1]
        )

    log_price = trend + residual
    log_return = log_price - log_price[:, [0]]
    relative_price = np.exp(log_return)
    return {
        "relative_price": relative_price,
        "log_return": log_return,
        "log_trend": trend,
        "log_residual": residual,
    }


def barrier_unit(model: JointTrendOuModel, config: dict[str, Any]) -> float:
    unit = str(config.get("barrier_unit", "horizon_log_return_volatility"))
    if unit == "horizon_log_return_volatility":
        value = model.observed_horizon_volatility
    elif unit == "ewma_1bar_log_return_volatility":
        value = model.observed_ewma_1bar_log_return_volatility
    elif unit == "one_step_log_return_volatility":
        value = model.observed_log_return_volatility
    elif unit == "ou_residual_innovation_std":
        value = model.ou_innovation_std
    else:
        raise ValueError(f"Unsupported barrier_unit: {unit}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid barrier unit value for {unit}: {value}")
    return float(value)


def first_hit_index(mask: np.ndarray, horizon: int) -> np.ndarray:
    has_hit = mask.any(axis=1)
    first = np.full(mask.shape[0], horizon + 1, dtype=np.int32)
    first[has_hit] = np.argmax(mask[has_hit], axis=1).astype(np.int32) + 1
    return first


def optimize_profit_take_stop_loss(
    log_return_paths: np.ndarray,
    model: JointTrendOuModel,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizon = min(int(config.get("vertical_barrier_bars", 100)), log_return_paths.shape[1] - 1)
    returns = log_return_paths[:, 1 : horizon + 1]
    terminal_return = returns[:, -1]
    unit = barrier_unit(model, config)
    pt_grid = grid_values(section(config, "profit_taking_grid"))
    sl_grid = grid_values(section(config, "stop_loss_grid"))
    tie_policy = str(config.get("tie_policy", "stop_loss_first"))
    rows: list[dict[str, Any]] = []

    for pt_multiplier in pt_grid:
        profit_take = float(pt_multiplier * unit)
        up_first = first_hit_index(returns >= profit_take, horizon)
        for sl_multiplier in sl_grid:
            stop_loss = float(sl_multiplier * unit)
            down_first = first_hit_index(returns <= -stop_loss, horizon)
            vertical = (up_first > horizon) & (down_first > horizon)
            if tie_policy == "stop_loss_first":
                up_wins = (up_first < down_first) & ~vertical
                down_wins = (down_first <= up_first) & ~vertical
            elif tie_policy == "profit_taking_first":
                up_wins = (up_first <= down_first) & ~vertical
                down_wins = (down_first < up_first) & ~vertical
            else:
                raise ValueError(f"Unsupported tie_policy: {tie_policy}")

            payoff = terminal_return.copy()
            payoff[up_wins] = profit_take
            payoff[down_wins] = -stop_loss
            payoff_std = float(np.std(payoff, ddof=1))
            rows.append(
                {
                    "pt_multiplier": float(pt_multiplier),
                    "sl_multiplier": float(sl_multiplier),
                    "profit_take_log_return": profit_take,
                    "stop_loss_log_return": stop_loss,
                    "expected_log_return": float(np.mean(payoff)),
                    "std_log_return": payoff_std,
                    "sharpe_like": float(np.mean(payoff) / payoff_std) if payoff_std > 0.0 else None,
                    "profit_take_hit_rate": float(np.mean(up_wins)),
                    "stop_loss_hit_rate": float(np.mean(down_wins)),
                    "vertical_barrier_rate": float(np.mean(vertical)),
                    "mean_exit_step": float(np.mean(np.minimum(np.minimum(up_first, down_first), horizon))),
                },
            )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["expected_log_return", "sharpe_like"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    best = frame.iloc[0].to_dict()
    best["barrier_unit"] = unit
    best["vertical_barrier_bars"] = horizon
    return frame, best


def dtype_from_config(config: dict[str, Any]) -> np.dtype:
    dtype_value = str(config.get("dtype", "float32"))
    if dtype_value == "float32":
        return np.dtype("float32")
    if dtype_value == "float64":
        return np.dtype("float64")
    raise ValueError(f"Unsupported Monte Carlo dtype: {dtype_value}")


def write_paths_npz(path: Path, arrays: dict[str, np.ndarray], dtype: np.dtype) -> None:
    cast_arrays = {name: value.astype(dtype, copy=False) for name, value in arrays.items()}
    cast_arrays["step"] = np.arange(next(iter(arrays.values())).shape[1], dtype=np.int16)
    np.savez_compressed(path, **cast_arrays)


def maybe_write_long_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    n_paths, n_steps = arrays["relative_price"].shape
    path_id = np.repeat(np.arange(n_paths, dtype=np.int32), n_steps)
    step = np.tile(np.arange(n_steps, dtype=np.int16), n_paths)
    frame = pd.DataFrame(
        {
            "path_id": path_id,
            "step": step,
            "relative_price": arrays["relative_price"].reshape(-1),
            "log_return": arrays["log_return"].reshape(-1),
            "log_trend": arrays["log_trend"].reshape(-1),
            "log_residual": arrays["log_residual"].reshape(-1),
        },
    )
    frame.to_csv(path, index=False)


def model_fit_summary(model: JointTrendOuModel) -> dict[str, Any]:
    return {
        "trend_drift": model.trend_drift,
        "trend_std": model.trend_std,
        "ou_intercept": model.ou_intercept,
        "ou_phi": model.ou_phi,
        "ou_mean": model.ou_mean,
        "ou_kappa": model.ou_kappa,
        "ou_half_life_bars": model.ou_half_life_bars,
        "ou_innovation_std": model.ou_innovation_std,
        "ou_sigma": model.ou_sigma,
        "innovation_covariance": model.innovation_covariance,
        "observed_log_return_volatility": model.observed_log_return_volatility,
        "observed_ewma_1bar_log_return_volatility": model.observed_ewma_1bar_log_return_volatility,
        "observed_horizon_volatility": model.observed_horizon_volatility,
    }


def synthetic_path_summary(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "terminal_log_return_mean": float(np.mean(arrays["log_return"][:, -1])),
        "terminal_log_return_std": float(np.std(arrays["log_return"][:, -1], ddof=1)),
        "terminal_relative_price_mean": float(np.mean(arrays["relative_price"][:, -1])),
        "terminal_relative_price_std": float(np.std(arrays["relative_price"][:, -1], ddof=1)),
    }


def generate_model_paths(
    *,
    symbol: str,
    input_csv: Path,
    output_dir: Path,
    output_subdir: Path,
    stem: str,
    model: JointTrendOuModel,
    n_paths: int,
    horizon: int,
    seed: int,
    seed_policy: str,
    monte_carlo_config: dict[str, Any],
    barrier_config: dict[str, Any],
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arrays = simulate_paths(
        model,
        n_paths=n_paths,
        horizon=horizon,
        seed_policy=seed_policy,
        rng=rng,
    )

    symbol_output_dir = output_dir / output_subdir
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = symbol_output_dir / f"{stem}.npz"
    write_paths_npz(npz_path, arrays, dtype=dtype_from_config(monte_carlo_config))

    long_csv_path = None
    if bool(monte_carlo_config.get("write_long_csv", False)):
        long_csv_path = symbol_output_dir / f"{stem}_paths.csv"
        maybe_write_long_csv(long_csv_path, arrays)

    grid_csv_path = None
    best_barrier = None
    if bool(barrier_config.get("enabled", True)):
        grid, best_barrier = optimize_profit_take_stop_loss(
            arrays["log_return"],
            model=model,
            config=barrier_config,
        )
        grid_csv_path = symbol_output_dir / f"{stem}_tp_sl_grid.csv"
        grid.to_csv(grid_csv_path, index=False)

    summary = {
        "symbol": symbol,
        "source_csv": str(input_csv),
        "paths_npz": str(npz_path),
        "long_csv": str(long_csv_path) if long_csv_path is not None else None,
        "tp_sl_grid_csv": str(grid_csv_path) if grid_csv_path is not None else None,
        "n_paths": n_paths,
        "max_horizon_bars": horizon,
        "seed": seed,
        "fit": model_fit_summary(model),
        "synthetic": synthetic_path_summary(arrays),
        "best_barrier": best_barrier,
    }
    if extra_summary:
        summary.update(extra_summary)
    summary_path = symbol_output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def generate_for_symbol(
    symbol: str,
    input_csv: Path,
    output_dir: Path,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    cli_n_paths: int | None,
    cli_horizon: int | None,
    cli_seed: int | None,
) -> dict[str, Any]:
    synthetic_config = section(config, "synthetic_data")
    price_model_config = section(synthetic_config, "price_model")
    fair_value_config = section(synthetic_config, "fair_value")
    trend_config = section(synthetic_config, "trend_process")
    ou_config = section(synthetic_config, "ou_process")
    monte_carlo_config = dict(section(synthetic_config, "monte_carlo"))
    barrier_config = section(strategy_config, "barrier_optimization")

    if cli_n_paths is not None:
        monte_carlo_config["n_paths"] = cli_n_paths
    if cli_horizon is not None:
        monte_carlo_config["max_horizon_bars"] = cli_horizon
    if cli_seed is not None:
        monte_carlo_config["seed"] = cli_seed

    price_column = str(price_model_config.get("price_column", "close"))
    df = read_real_bars(input_csv, price_column=price_column)
    model = fit_joint_trend_ou_model(
        df,
        price_column=price_column,
        fair_value_config=fair_value_config,
        trend_config=trend_config,
        ou_config=ou_config,
        barrier_config=barrier_config,
    )

    n_paths = int(monte_carlo_config.get("n_paths", 25_000))
    horizon = int(monte_carlo_config.get("max_horizon_bars", 100))
    seed = deterministic_symbol_seed(int(monte_carlo_config.get("seed", 42)), symbol)
    rng = np.random.default_rng(seed)
    arrays = simulate_paths(
        model,
        n_paths=n_paths,
        horizon=horizon,
        seed_policy=str(ou_config.get("seed_policy", "stationary_distribution")),
        rng=rng,
    )

    symbol_output_dir = output_dir / "mc_paths" / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    q = float(fair_value_config.get("q_over_r", 0.001))
    stem = f"{symbol}_OU_KALMAN_QR_{q_label(q)}_{n_paths}x{horizon}"
    npz_path = symbol_output_dir / f"{stem}.npz"
    write_paths_npz(npz_path, arrays, dtype=dtype_from_config(monte_carlo_config))

    long_csv_path = None
    if bool(monte_carlo_config.get("write_long_csv", False)):
        long_csv_path = symbol_output_dir / f"{stem}_paths.csv"
        maybe_write_long_csv(long_csv_path, arrays)

    grid_csv_path = None
    best_barrier = None
    if bool(barrier_config.get("enabled", True)):
        grid, best_barrier = optimize_profit_take_stop_loss(
            arrays["log_return"],
            model=model,
            config=barrier_config,
        )
        grid_csv_path = symbol_output_dir / f"{stem}_tp_sl_grid.csv"
        grid.to_csv(grid_csv_path, index=False)

    summary = {
        "symbol": symbol,
        "source_csv": str(input_csv),
        "paths_npz": str(npz_path),
        "long_csv": str(long_csv_path) if long_csv_path is not None else None,
        "tp_sl_grid_csv": str(grid_csv_path) if grid_csv_path is not None else None,
        "n_paths": n_paths,
        "max_horizon_bars": horizon,
        "seed": seed,
        "price_model": price_model_config,
        "fair_value": fair_value_config,
        "trend_process": trend_config,
        "ou_process": ou_config,
        "monte_carlo": monte_carlo_config,
        "barrier_optimization": barrier_config,
        "strategy_config_path": strategy_config.get("_strategy_config_path"),
        "fit": {
            "trend_drift": model.trend_drift,
            "trend_std": model.trend_std,
            "ou_intercept": model.ou_intercept,
            "ou_phi": model.ou_phi,
            "ou_mean": model.ou_mean,
            "ou_kappa": model.ou_kappa,
            "ou_half_life_bars": model.ou_half_life_bars,
            "ou_innovation_std": model.ou_innovation_std,
            "ou_sigma": model.ou_sigma,
            "innovation_covariance": model.innovation_covariance,
            "observed_log_return_volatility": model.observed_log_return_volatility,
            "observed_ewma_1bar_log_return_volatility": model.observed_ewma_1bar_log_return_volatility,
            "observed_horizon_volatility": model.observed_horizon_volatility,
        },
        "synthetic": {
            "terminal_log_return_mean": float(np.mean(arrays["log_return"][:, -1])),
            "terminal_log_return_std": float(np.std(arrays["log_return"][:, -1], ddof=1)),
            "terminal_relative_price_mean": float(np.mean(arrays["relative_price"][:, -1])),
            "terminal_relative_price_std": float(np.std(arrays["relative_price"][:, -1], ddof=1)),
        },
        "best_barrier": best_barrier,
    }
    summary_path = symbol_output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def generate_rolling_for_symbol(
    symbol: str,
    input_csv: Path,
    output_dir: Path,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
    cli_n_paths: int | None,
    cli_horizon: int | None,
    cli_seed: int | None,
    rolling_window_days: float,
    rolling_step_days: float,
    show_progress: bool,
) -> dict[str, Any]:
    synthetic_config = section(config, "synthetic_data")
    price_model_config = section(synthetic_config, "price_model")
    fair_value_config = section(synthetic_config, "fair_value")
    trend_config = section(synthetic_config, "trend_process")
    ou_config = section(synthetic_config, "ou_process")
    monte_carlo_config = dict(section(synthetic_config, "monte_carlo"))
    barrier_config = section(strategy_config, "barrier_optimization")

    if cli_n_paths is not None:
        monte_carlo_config["n_paths"] = cli_n_paths
    if cli_horizon is not None:
        monte_carlo_config["max_horizon_bars"] = cli_horizon
    if cli_seed is not None:
        monte_carlo_config["seed"] = cli_seed

    price_column = str(price_model_config.get("price_column", "close"))
    df = read_real_bars(input_csv, price_column=price_column)
    timestamps = bar_timestamps(df)
    bounds = rolling_window_bounds(
        timestamps,
        window_days=rolling_window_days,
        step_days=rolling_step_days,
    )
    if not bounds:
        raise ValueError(f"No rolling windows generated for {symbol}.")

    n_paths = int(monte_carlo_config.get("n_paths", 25_000))
    horizon = int(monte_carlo_config.get("max_horizon_bars", 100))
    base_seed = int(monte_carlo_config.get("seed", 42))
    q = float(fair_value_config.get("q_over_r", 0.001))
    window_summaries = []
    failures: list[dict[str, Any]] = []
    iterator = tqdm(
        bounds,
        desc=f"{symbol} rolling {int(rolling_window_days)}d",
        disable=not show_progress,
        dynamic_ncols=True,
        unit="window",
    )
    for window_start, window_end in iterator:
        start_label = timestamp_label(window_start)
        end_label = timestamp_label(window_end)
        mask = (timestamps >= window_start) & (timestamps <= window_end)
        window_df = df.loc[mask].reset_index(drop=True)
        try:
            model = fit_joint_trend_ou_model(
                window_df,
                price_column=price_column,
                fair_value_config=fair_value_config,
                trend_config=trend_config,
                ou_config=ou_config,
                barrier_config=barrier_config,
            )
            seed = deterministic_symbol_seed(base_seed, f"{symbol}_{start_label}_{end_label}")
            stem = (
                f"{symbol}_OU_KALMAN_QR_{q_label(q)}_ROLL{int(rolling_window_days)}D_"
                f"{start_label}_{end_label}_{n_paths}x{horizon}"
            )
            summary = generate_model_paths(
                symbol=symbol,
                input_csv=input_csv,
                output_dir=output_dir,
                output_subdir=Path("mc_paths_rolling") / f"{int(rolling_window_days)}d" / symbol,
                stem=stem,
                model=model,
                n_paths=n_paths,
                horizon=horizon,
                seed=seed,
                seed_policy=str(ou_config.get("seed_policy", "stationary_distribution")),
                monte_carlo_config=monte_carlo_config,
                barrier_config=barrier_config,
                extra_summary={
                    "calibration_mode": "rolling",
                    "rolling_window_days": rolling_window_days,
                    "rolling_step_days": rolling_step_days,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "window_n_bars": len(window_df),
                    "price_model": price_model_config,
                    "fair_value": fair_value_config,
                    "trend_process": trend_config,
                    "ou_process": ou_config,
                    "monte_carlo": monte_carlo_config,
                    "barrier_optimization": barrier_config,
                    "strategy_config_path": strategy_config.get("_strategy_config_path"),
                },
            )
            window_summaries.append(summary)
        except ValueError as exc:
            failures.append(
                {
                    "symbol": symbol,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "window_n_bars": len(window_df),
                    "error": str(exc),
                },
            )

    return {
        "symbol": symbol,
        "source_csv": str(input_csv),
        "calibration_mode": "rolling",
        "rolling_window_days": rolling_window_days,
        "rolling_step_days": rolling_step_days,
        "n_windows": len(window_summaries),
        "n_failed_windows": len(failures),
        "n_paths": n_paths,
        "max_horizon_bars": horizon,
        "summaries": window_summaries,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AFML Chapter 13 style Monte Carlo price paths from a calibrated "
            "Kalman-trend plus OU-residual stochastic process."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to AFML data config JSON.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--input-csv", default=None, help="Only valid when generating one symbol.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--max-horizon-bars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy-config", default=None)
    parser.add_argument(
        "--rolling-window-days",
        type=float,
        default=None,
        help="Fit and generate paths on rolling calendar windows of this length.",
    )
    parser.add_argument(
        "--rolling-step-days",
        type=float,
        default=30.0,
        help="Calendar-day step between rolling calibration windows.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_afml_data_config(args.config)
    synthetic_config = section(config, "synthetic_data")
    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else resolve_repo_path(synthetic_config.get("output_dir", "afml_scripts/output/ou_synthetic"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_config = load_strategy_config(config, args.strategy_config)

    symbols = configured_symbols(config, args.symbols)
    summaries = []
    for symbol in symbols:
        input_csv = input_csv_for_symbol(
            symbol,
            config,
            cli_input_csv=args.input_csv,
            total_symbols=len(symbols),
        )
        if args.rolling_window_days is not None:
            summary = generate_rolling_for_symbol(
                symbol,
                input_csv=input_csv,
                output_dir=output_dir,
                config=config,
                strategy_config=strategy_config,
                cli_n_paths=args.n_paths,
                cli_horizon=args.max_horizon_bars,
                cli_seed=args.seed,
                rolling_window_days=args.rolling_window_days,
                rolling_step_days=args.rolling_step_days,
                show_progress=not args.no_progress,
            )
            summaries.append(summary)
            print(
                f"{symbol}: rolling_windows={summary['n_windows']} "
                f"failed={summary['n_failed_windows']} paths/window={summary['n_paths']:,} "
                f"horizon={summary['max_horizon_bars']}",
            )
        else:
            summary = generate_for_symbol(
                symbol,
                input_csv=input_csv,
                output_dir=output_dir,
                config=config,
                strategy_config=strategy_config,
                cli_n_paths=args.n_paths,
                cli_horizon=args.max_horizon_bars,
                cli_seed=args.seed,
            )
            summaries.append(summary)
            best = summary["best_barrier"] or {}
            print(
                f"{symbol}: paths={summary['n_paths']:,} horizon={summary['max_horizon_bars']} "
                f"OU_half_life={summary['fit']['ou_half_life_bars']:.2f} "
                f"best_pt={best.get('pt_multiplier')} best_sl={best.get('sl_multiplier')} "
                f"E={best.get('expected_log_return')}",
            )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "symbols": symbols,
        "calibration_mode": "rolling" if args.rolling_window_days is not None else "full_sample",
        "rolling_window_days": args.rolling_window_days,
        "rolling_step_days": args.rolling_step_days if args.rolling_window_days is not None else None,
        "summaries": summaries,
    }
    manifest_name = (
        "ou_residual_rolling_mc_manifest.json"
        if args.rolling_window_days is not None
        else "ou_residual_mc_manifest.json"
    )
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
