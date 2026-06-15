#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Calendar-period normality diagnostics for generated AFML DRB CSVs.
#
# AFML Chapter 2 compares sampled-bar returns by how close their distribution is
# to Gaussian. This script uses Jarque-Bera as the primary normality test because
# it directly checks skew and excess kurtosis, the two moments emphasized by that
# comparison.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_strategies.config_loader import load_afml_data_config
from afml_strategies.config_loader import section
from afml_strategies.config_loader import string_tuple


DEFAULT_DATA_ROOT = Path("binance_scripts/data/afml_bars")
DEFAULT_OUTPUT_DIR = Path("binance_scripts/data/afml_bar_diagnostics")
DEFAULT_ALPHA = 0.05
MIN_RETURNS = 8

DETAIL_FIELDS = [
    "symbol",
    "bar_type",
    "source_csv",
    "period_level",
    "period",
    "period_start_ts",
    "period_end_ts",
    "alpha",
    "return_count",
    "tested",
    "mean_return",
    "std_return",
    "skew",
    "kurtosis",
    "excess_kurtosis",
    "jarque_bera_stat",
    "jarque_bera_pvalue",
    "jb_stat_per_return",
    "jb_moment_distance",
    "reject_normal",
    "normality_assessment",
]

SUMMARY_FIELDS = [
    "symbol",
    "period_level",
    "periods",
    "tested_periods",
    "rejected_periods",
    "rejected_share",
    "not_rejected_periods",
    "median_return_count",
    "median_jarque_bera_pvalue",
    "median_jb_moment_distance",
    "median_skew",
    "median_kurtosis",
    "max_kurtosis",
    "worst_period",
    "worst_period_jb_moment_distance",
    "worst_period_pvalue",
    "worst_period_kurtosis",
]


def normalize_symbol(value: str) -> str:
    return value.strip().upper().replace("/", "").replace("-", "").replace(".P", "")


def configured_symbols() -> list[str]:
    config = load_afml_data_config()
    real_config = section(config, "real_data")
    return [normalize_symbol(symbol) for symbol in string_tuple(real_config.get("symbols"), default=("BTCUSDT.P",))]


def latest_drb_csv(data_root: Path, symbol: str) -> Path:
    symbol_dir = data_root / normalize_symbol(symbol)
    candidates = sorted(symbol_dir.glob("*_DRB.csv"))
    if not candidates:
        raise FileNotFoundError(f"No DRB CSV found for {symbol} under {symbol_dir}")
    return candidates[-1]


def clean_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def week_label(value: datetime) -> str:
    day = value.date()
    start = day - timedelta(days=day.weekday())
    end = start + timedelta(days=6)
    return f"{start.isoformat()}..{end.isoformat()}"


def period_keys(value: datetime) -> dict[str, str]:
    return {
        "day": value.date().isoformat(),
        "week": week_label(value),
        "month": f"{value.year:04d}-{value.month:02d}",
    }


def load_drb_returns(path: Path) -> list[dict[str, Any]]:
    rows: list[tuple[datetime, float]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            close = float(row["close"])
            if close <= 0.0 or not math.isfinite(close):
                continue
            rows.append((parse_ts(row["ts_event"]), close))

    rows.sort(key=lambda item: item[0])
    returns: list[dict[str, Any]] = []
    previous_log_close: float | None = None
    for timestamp, close in rows:
        log_close = math.log(close)
        if previous_log_close is not None:
            returns.append(
                {
                    "ts_event": timestamp,
                    "log_return": log_close - previous_log_close,
                },
            )
        previous_log_close = log_close
    return returns


def normality_metrics(values: list[float], *, alpha: float) -> dict[str, Any]:
    values = [value for value in values if math.isfinite(value)]
    count = len(values)
    if count == 0:
        mean = None
        std = None
    else:
        mean = sum(values) / count
        if count > 1:
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1))
        else:
            std = None

    if count < MIN_RETURNS:
        return {
            "return_count": count,
            "tested": False,
            "mean_return": clean_float(mean),
            "std_return": clean_float(std),
            "skew": None,
            "kurtosis": None,
            "excess_kurtosis": None,
            "jarque_bera_stat": None,
            "jarque_bera_pvalue": None,
            "jb_stat_per_return": None,
            "jb_moment_distance": None,
            "reject_normal": None,
            "normality_assessment": "too_few_returns",
        }

    centered = [value - mean for value in values]
    second = sum(value**2 for value in centered) / count
    third = sum(value**3 for value in centered) / count
    fourth = sum(value**4 for value in centered) / count
    if second <= 0.0:
        skew = 0.0
        kurtosis = 0.0
    else:
        skew = third / (second**1.5)
        kurtosis = fourth / (second**2)

    excess_kurtosis = kurtosis - 3.0
    jarque_bera_stat = count / 6.0 * (skew * skew + (excess_kurtosis * excess_kurtosis) / 4.0)
    # The Jarque-Bera asymptotic null distribution is chi-square with 2 df.
    # For 2 df, the survival function is exactly exp(-x / 2).
    jarque_bera_pvalue = math.exp(-jarque_bera_stat / 2.0)
    jb_moment_distance = math.sqrt(skew * skew + (excess_kurtosis * excess_kurtosis) / 4.0)
    reject = jarque_bera_pvalue < alpha

    if not reject:
        assessment = "not_rejected"
    elif jb_moment_distance < 1.0:
        assessment = "rejected_mild"
    elif jb_moment_distance < 3.0:
        assessment = "rejected_material"
    else:
        assessment = "rejected_severe"

    return {
        "return_count": count,
        "tested": True,
        "mean_return": clean_float(mean),
        "std_return": clean_float(std),
        "skew": clean_float(skew),
        "kurtosis": clean_float(kurtosis),
        "excess_kurtosis": clean_float(excess_kurtosis),
        "jarque_bera_stat": clean_float(jarque_bera_stat),
        "jarque_bera_pvalue": clean_float(jarque_bera_pvalue),
        "jb_stat_per_return": clean_float(jarque_bera_stat / count),
        "jb_moment_distance": clean_float(jb_moment_distance),
        "reject_normal": reject,
        "normality_assessment": assessment,
    }


def diagnose_symbol(path: Path, *, alpha: float) -> list[dict[str, Any]]:
    symbol = path.parent.name
    returns = load_drb_returns(path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in returns:
        timestamp = row["ts_event"]
        for level, period in period_keys(timestamp).items():
            grouped[(level, period)].append(row)

    rows: list[dict[str, Any]] = []
    for level in ("day", "week", "month"):
        keys = sorted(key for key in grouped if key[0] == level)
        for _, period in keys:
            group = grouped[(level, period)]
            metrics = normality_metrics([row["log_return"] for row in group], alpha=alpha)
            rows.append(
                {
                    "symbol": symbol,
                    "bar_type": "DRB",
                    "source_csv": str(path.resolve()),
                    "period_level": level,
                    "period": period,
                    "period_start_ts": min(row["ts_event"] for row in group).isoformat(),
                    "period_end_ts": max(row["ts_event"] for row in group).isoformat(),
                    "alpha": alpha,
                    **metrics,
                },
            )

    metrics = normality_metrics([row["log_return"] for row in returns], alpha=alpha)
    rows.append(
        {
            "symbol": symbol,
            "bar_type": "DRB",
            "source_csv": str(path.resolve()),
            "period_level": "all",
            "period": "all",
            "period_start_ts": min(row["ts_event"] for row in returns).isoformat(),
            "period_end_ts": max(row["ts_event"] for row in returns).isoformat(),
            "alpha": alpha,
            **metrics,
        },
    )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["symbol"], row["period_level"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for symbol, period_level in sorted(grouped):
        group = grouped[(symbol, period_level)]
        tested = [row for row in group if row["tested"] is True]
        if not tested:
            summary_rows.append(
                {
                    "symbol": symbol,
                    "period_level": period_level,
                    "periods": len(group),
                    "tested_periods": 0,
                    "rejected_periods": 0,
                    "rejected_share": None,
                    "not_rejected_periods": 0,
                    "median_return_count": None,
                    "median_jarque_bera_pvalue": None,
                    "median_jb_moment_distance": None,
                    "median_skew": None,
                    "median_kurtosis": None,
                    "max_kurtosis": None,
                    "worst_period": None,
                    "worst_period_jb_moment_distance": None,
                    "worst_period_pvalue": None,
                    "worst_period_kurtosis": None,
                },
            )
            continue

        rejected = [row for row in tested if row["reject_normal"] is True]
        worst = max(tested, key=lambda row: row["jb_moment_distance"] or float("-inf"))
        summary_rows.append(
            {
                "symbol": symbol,
                "period_level": period_level,
                "periods": len(group),
                "tested_periods": len(tested),
                "rejected_periods": len(rejected),
                "rejected_share": clean_float(len(rejected) / len(tested)),
                "not_rejected_periods": len(tested) - len(rejected),
                "median_return_count": clean_float(median([row["return_count"] for row in tested])),
                "median_jarque_bera_pvalue": clean_float(median([row["jarque_bera_pvalue"] for row in tested])),
                "median_jb_moment_distance": clean_float(median([row["jb_moment_distance"] for row in tested])),
                "median_skew": clean_float(median([row["skew"] for row in tested])),
                "median_kurtosis": clean_float(median([row["kurtosis"] for row in tested])),
                "max_kurtosis": clean_float(max(row["kurtosis"] for row in tested)),
                "worst_period": worst["period"],
                "worst_period_jb_moment_distance": clean_float(worst["jb_moment_distance"]),
                "worst_period_pvalue": clean_float(worst["jarque_bera_pvalue"]),
                "worst_period_kurtosis": clean_float(worst["kurtosis"]),
            },
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    fields = [
        "symbol",
        "period_level",
        "tested_periods",
        "rejected_periods",
        "rejected_share",
        "median_return_count",
        "median_jarque_bera_pvalue",
        "median_jb_moment_distance",
        "median_skew",
        "median_kurtosis",
        "worst_period",
    ]
    widths = {
        field: max(
            len(field),
            *(len("" if row[field] is None else str(row[field])) for row in rows),
        )
        for field in fields
    }
    print(" ".join(field.rjust(widths[field]) for field in fields))
    for row in rows:
        print(" ".join(("" if row[field] is None else str(row[field])).rjust(widths[field]) for field in fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test generated DRB log returns for normality by day, week, month, and full sample.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [normalize_symbol(symbol) for symbol in (args.symbols or configured_symbols())]
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        rows.extend(diagnose_symbol(latest_drb_csv(args.data_root, symbol), alpha=args.alpha))

    summary_rows = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(symbols).lower()
    detail_csv = args.output_dir / f"drb_period_normality_{suffix}.csv"
    summary_csv = args.output_dir / f"drb_period_normality_{suffix}_summary.csv"
    detail_json = args.output_dir / f"drb_period_normality_{suffix}.json"

    write_csv(detail_csv, rows, DETAIL_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    detail_json.write_text(
        json.dumps({"windows": rows, "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )

    print_summary(summary_rows)
    print(f"wrote {detail_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {detail_json}")


if __name__ == "__main__":
    main()
