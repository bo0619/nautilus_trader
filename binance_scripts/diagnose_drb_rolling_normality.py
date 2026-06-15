#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Rolling normality diagnostics for generated AFML DRB CSVs.
#
# This script checks how far rolling DRB log returns are from a Gaussian
# distribution using the AFML Chapter 2 Jarque-Bera criterion.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_DATA_ROOT = Path("binance_scripts/data/afml_bars")
DEFAULT_OUTPUT_DIR = Path("binance_scripts/data/afml_bar_diagnostics")


def _latest_drb_csv(data_root: Path, symbol: str) -> Path:
    symbol_dir = data_root / symbol
    candidates = sorted(symbol_dir.glob("*_DRB.csv"))
    if not candidates:
        raise FileNotFoundError(f"No DRB CSV found for {symbol} under {symbol_dir}")
    return candidates[-1]


def _clean_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric):
        return numeric
    return None


def _window_assessment(pvalue: float, distance: float) -> str:
    if pvalue >= 0.05:
        return "not_rejected"
    if distance < 1.0:
        return "rejected_mild"
    if distance < 3.0:
        return "rejected_material"
    return "rejected_severe"


def _normality_metrics(values: np.ndarray) -> dict[str, Any]:
    if values.size < 8:
        raise ValueError("Need at least 8 returns for Jarque-Bera diagnostics.")

    skew = stats.skew(values, bias=False)
    kurtosis = stats.kurtosis(values, fisher=False, bias=False)
    excess_kurtosis = kurtosis - 3.0
    jb = stats.jarque_bera(values)

    # JB = n / 6 * (skew^2 + excess_kurtosis^2 / 4).
    # This distance removes the sample-size multiplier, so windows can be compared
    # without making larger windows look worse only because they have more rows.
    jb_moment_distance = float(np.sqrt(skew * skew + (excess_kurtosis * excess_kurtosis) / 4.0))

    return {
        "return_count": int(values.size),
        "mean_return": _clean_float(np.mean(values)),
        "std_return": _clean_float(np.std(values, ddof=1)),
        "skew": _clean_float(skew),
        "kurtosis": _clean_float(kurtosis),
        "excess_kurtosis": _clean_float(excess_kurtosis),
        "jarque_bera_stat": _clean_float(jb.statistic),
        "jarque_bera_pvalue": _clean_float(jb.pvalue),
        "jb_stat_per_return": _clean_float(jb.statistic / values.size),
        "jb_moment_distance": _clean_float(jb_moment_distance),
        "reject_normal_5pct": bool(jb.pvalue < 0.05),
        "normality_assessment": _window_assessment(float(jb.pvalue), jb_moment_distance),
    }


def _load_drb_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["ts_event", "close"])
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame = frame.sort_values("ts_event").dropna(subset=["close"])
    frame = frame[frame["close"] > 0].reset_index(drop=True)
    frame["log_return"] = np.log(frame["close"]).diff()
    frame = frame.dropna(subset=["log_return"]).reset_index(drop=True)
    frame["day"] = frame["ts_event"].dt.tz_convert(None).dt.to_period("D")
    frame["month"] = frame["ts_event"].dt.tz_convert(None).dt.to_period("M")
    frame["week"] = frame["ts_event"].dt.tz_convert(None).dt.to_period("W-SUN")
    return frame


def diagnose_symbol(
    path: Path,
    *,
    window_count: int,
    period_column: str,
    period_freq: str,
    period_unit: str,
) -> list[dict[str, Any]]:
    frame = _load_drb_returns(path)
    periods = pd.PeriodIndex(sorted(frame[period_column].unique()), freq=period_freq)
    rows: list[dict[str, Any]] = []

    for start_index in range(len(periods) - window_count + 1):
        window_periods = periods[start_index : start_index + window_count]
        start_period = window_periods[0]
        end_period = window_periods[-1]
        window = frame[frame[period_column].isin(window_periods)]
        values = window["log_return"].to_numpy(dtype=float)
        metrics = _normality_metrics(values)
        rows.append(
            {
                "symbol": path.parent.name,
                "source_csv": str(path.resolve()),
                "window_unit": period_unit,
                "window_count": window_count,
                "window_start": str(start_period),
                "window_end": str(end_period),
                "window_start_ts": window["ts_event"].min().isoformat(),
                "window_end_ts": window["ts_event"].max().isoformat(),
                **metrics,
            },
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        worst = group.loc[group["jb_moment_distance"].idxmax()]
        best = group.loc[group["jb_moment_distance"].idxmin()]
        summary_rows.append(
            {
                "symbol": symbol,
                "windows": len(group),
                "rejected_windows": int(group["reject_normal_5pct"].sum()),
                "median_jb_moment_distance": _clean_float(group["jb_moment_distance"].median()),
                "max_jb_moment_distance": _clean_float(group["jb_moment_distance"].max()),
                "median_jarque_bera_stat": _clean_float(group["jarque_bera_stat"].median()),
                "max_jarque_bera_stat": _clean_float(group["jarque_bera_stat"].max()),
                "median_skew": _clean_float(group["skew"].median()),
                "median_kurtosis": _clean_float(group["kurtosis"].median()),
                "max_kurtosis": _clean_float(group["kurtosis"].max()),
                "best_window": f"{best['window_start']}..{best['window_end']}",
                "best_window_distance": _clean_float(best["jb_moment_distance"]),
                "worst_window": f"{worst['window_start']}..{worst['window_end']}",
                "worst_window_distance": _clean_float(worst["jb_moment_distance"]),
                "worst_window_kurtosis": _clean_float(worst["kurtosis"]),
            },
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rolling Jarque-Bera normality diagnostics for DRB returns.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--rolling-months", type=int, default=3)
    parser.add_argument("--rolling-weeks", type=int, default=None)
    parser.add_argument("--rolling-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rolling_days is not None:
        if args.rolling_days < 1:
            raise ValueError("--rolling-days must be positive")
        window_count = args.rolling_days
        period_column = "day"
        period_freq = "D"
        period_unit = "day"
        output_suffix = f"{window_count}d"
    elif args.rolling_weeks is not None:
        if args.rolling_weeks < 1:
            raise ValueError("--rolling-weeks must be positive")
        window_count = args.rolling_weeks
        period_column = "week"
        period_freq = "W-SUN"
        period_unit = "week"
        output_suffix = f"{window_count}w"
    else:
        if args.rolling_months < 1:
            raise ValueError("--rolling-months must be positive")
        window_count = args.rolling_months
        period_column = "month"
        period_freq = "M"
        period_unit = "month"
        output_suffix = f"{window_count}m"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for symbol in args.symbols:
        path = _latest_drb_csv(args.data_root, symbol)
        rows.extend(
            diagnose_symbol(
                path,
                window_count=window_count,
                period_column=period_column,
                period_freq=period_freq,
                period_unit=period_unit,
            ),
        )

    summary_rows = summarize(rows)
    detail_csv = args.output_dir / f"drb_rolling_{output_suffix}_normality.csv"
    summary_csv = args.output_dir / f"drb_rolling_{output_suffix}_normality_summary.csv"
    detail_json = args.output_dir / f"drb_rolling_{output_suffix}_normality.json"

    pd.DataFrame(rows).to_csv(detail_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    detail_json.write_text(
        json.dumps({"windows": rows, "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "symbol",
        "windows",
        "rejected_windows",
        "median_jb_moment_distance",
        "max_jb_moment_distance",
        "median_kurtosis",
        "max_kurtosis",
        "worst_window",
    ]
    print(pd.DataFrame(summary_rows)[display_columns].to_string(index=False))
    print(f"wrote {detail_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {detail_json}")


if __name__ == "__main__":
    main()
