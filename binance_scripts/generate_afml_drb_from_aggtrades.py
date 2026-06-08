#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Generate AFML dollar runs bars (DRB) directly from Binance aggTrades archives.
#
# This file is intentionally a user-facing entrypoint:
# edit afml_strategies/afml_data_config.json, then run this file from VSCode.
# --------------------------------------------------------------------------

from __future__ import annotations

import calendar
import csv
import json
import shutil
import sys
import time
import zipfile
from bisect import bisect_right
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np


# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_strategies.config_loader import load_afml_data_config
from afml_strategies.config_loader import resolve_repo_path
from afml_strategies.config_loader import section
from afml_strategies.config_loader import string_tuple
from nautilus_trader.data.afml_bars import AfmlBarMeta
from nautilus_trader.data.afml_bars import AfmlDollarRunsBarAggregator
from nautilus_trader.data.afml_bars import estimate_afml_dollar_expectations
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarAggregation
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog


CONFIG = load_afml_data_config()
REAL_DATA_CONFIG = section(CONFIG, "real_data")
INCREMENTAL_CONFIG = section(REAL_DATA_CONFIG, "incremental")


# --------------------------------------------------------------------------
# VSCode Run Button config
# --------------------------------------------------------------------------
SYMBOLS = list(string_tuple(REAL_DATA_CONFIG.get("symbols"), default=("BTCUSDT.P",)))

THRESHOLD_HISTORY_START = str(REAL_DATA_CONFIG.get("threshold_history_start", "2025-01-01"))
GENERATION_START_CONFIG = str(REAL_DATA_CONFIG.get("generation_start", "2025-02-01"))
GENERATION_END_CONFIG = str(REAL_DATA_CONFIG.get("generation_end", "2026-04-30"))
GENERATION_START = GENERATION_START_CONFIG
GENERATION_END = GENERATION_END_CONFIG

PRODUCT = str(REAL_DATA_CONFIG.get("product", "um"))
THRESHOLD_ROLLING_WINDOW_DAYS = int(REAL_DATA_CONFIG.get("threshold_rolling_window_days", 30))
THRESHOLD_DAILY_NOTIONAL_DIVISOR = int(REAL_DATA_CONFIG.get("threshold_daily_notional_divisor", 250))
THRESHOLD_UPDATE_FREQUENCY = str(REAL_DATA_CONFIG.get("threshold_update_frequency", "monthly"))
THRESHOLD_MIN_LOOKBACK_DAYS = int(REAL_DATA_CONFIG.get("threshold_min_lookback_days", 7))
TARGET_BARS_PER_DAY = THRESHOLD_DAILY_NOTIONAL_DIVISOR
EWMA_SPAN = int(REAL_DATA_CONFIG.get("ewma_span", 20))

# Keep E[T] fixed to the target density. The DRB side/notional expectations
# still update online after each closed bar.
UPDATE_EXPECTED_TICKS = bool(REAL_DATA_CONFIG.get("update_expected_ticks", False))
INCREMENTAL_ENABLED = bool(INCREMENTAL_CONFIG.get("enabled", False))
INCREMENTAL_END_POLICY = str(INCREMENTAL_CONFIG.get("end_policy", "configured"))
GENERATION_START_FALLBACK = str(
    INCREMENTAL_CONFIG.get("generation_start_fallback", GENERATION_START_CONFIG),
)

INPUT_DIR = resolve_repo_path(REAL_DATA_CONFIG.get("input_dir", "binance_scripts/data/binance_vision"))
OUTPUT_DIR = resolve_repo_path(REAL_DATA_CONFIG.get("output_dir", "binance_scripts/data/afml_bars"))
CATALOG_DIR = resolve_repo_path(REAL_DATA_CONFIG.get("catalog_dir", "binance_scripts/data/afml_catalog"))
CALIBRATION_OUTPUT_DIR = resolve_repo_path(
    REAL_DATA_CONFIG.get("calibration_output_dir", "binance_scripts/data/afml_bar_calibration"),
)
TICK_CACHE_DIR = resolve_repo_path(REAL_DATA_CONFIG.get("tick_cache_dir", "binance_scripts/data/afml_tick_cache"))

USE_TICK_CACHE = bool(REAL_DATA_CONFIG.get("use_tick_cache", True))
WRITE_CATALOG = bool(REAL_DATA_CONFIG.get("write_catalog", True))
RESUME_CALIBRATION = bool(REAL_DATA_CONFIG.get("resume_calibration", True))
RESUME_COMPLETED_CSV = bool(REAL_DATA_CONFIG.get("resume_completed_csv", True))
BACKUP_EXISTING_CATALOG = bool(REAL_DATA_CONFIG.get("backup_existing_catalog", True))


TICK_CACHE_SCHEMA_VERSION = str(REAL_DATA_CONFIG.get("tick_cache_schema_version", "1"))
TICK_CACHE_WRITE_BATCH_SIZE = int(REAL_DATA_CONFIG.get("tick_cache_write_batch_size", 500_000))
TICK_CACHE_READ_BATCH_SIZE = int(REAL_DATA_CONFIG.get("tick_cache_read_batch_size", 250_000))


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def last_complete_month_end(today: date | None = None) -> date:
    today = today or datetime.now(tz=UTC).date()
    first_day_this_month = date(today.year, today.month, 1)
    return first_day_this_month - timedelta(days=1)


def resolve_generation_end(value: str, *, incremental: bool) -> date:
    if incremental and INCREMENTAL_END_POLICY == "last_complete_month":
        return last_complete_month_end()
    if value in {"auto", "auto_incremental", "last_complete_month"}:
        return last_complete_month_end()
    return parse_date(value)


def set_generation_range(start: date, end: date) -> None:
    global GENERATION_START
    global GENERATION_END

    GENERATION_START = start.isoformat()
    GENERATION_END = end.isoformat()


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def date_to_ms(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)


def next_date_ms(value: date) -> int:
    return date_to_ms(value + timedelta(days=1))


def millis_to_nanos(value: int) -> int:
    return value * 1_000_000


def normalize_symbol(value: str) -> str:
    return value.strip().upper().replace("/", "").replace("-", "").replace(".P", "")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t"}


def decimal_places(value: str) -> int:
    value = value.strip()
    if "." not in value:
        return 0
    return len(value.rstrip("0").split(".", maxsplit=1)[1])


def ns_to_iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).isoformat(timespec="milliseconds")


def day_key(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).date().isoformat()


def month_key(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).strftime("%Y-%m")


def read_zip_rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file in {path}")

        with archive.open(csv_names[0]) as raw:
            text = (line.decode("utf-8").strip() for line in raw)
            reader = csv.reader(text)
            first = next(reader, None)
            if first is None:
                return

            header = [value.strip().lower() for value in first]
            has_header = not first[0].strip().lstrip("-").isdigit()
            if has_header:
                pos = {
                    "agg_trade_id": header.index("agg_trade_id"),
                    "price": header.index("price"),
                    "quantity": header.index("quantity"),
                    "transact_time": header.index("transact_time"),
                    "is_buyer_maker": header.index("is_buyer_maker"),
                }
            else:
                pos = {
                    "agg_trade_id": 0,
                    "price": 1,
                    "quantity": 2,
                    "transact_time": 5,
                    "is_buyer_maker": 6,
                }
                yield first, pos

            for row in reader:
                if row:
                    yield row, pos


def infer_price_size_precision(paths: list[Path]) -> tuple[int, int]:
    price_precision = 0
    size_precision = 0
    for path in paths:
        for row, pos in read_zip_rows(path):
            price_precision = max(price_precision, decimal_places(row[pos["price"]]))
            size_precision = max(size_precision, decimal_places(row[pos["quantity"]]))
            break
    if price_precision == 0 and size_precision == 0:
        raise ValueError("Cannot infer instrument precision from empty archives")
    return price_precision, size_precision


def split_linear_perp_symbol(symbol: str) -> tuple[str, str]:
    raw_symbol = normalize_symbol(symbol).removesuffix("_PERP")
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if raw_symbol.endswith(quote) and len(raw_symbol) > len(quote):
            return raw_symbol[: -len(quote)], quote
    raise ValueError(
        f"Cannot infer base/quote currencies from '{symbol}'. "
        "Expected a linear perpetual symbol ending in USDT, USDC, BUSD, or USD.",
    )


def make_binance_perpetual(symbol: str, paths: list[Path]) -> CryptoPerpetual:
    raw_symbol = normalize_symbol(symbol).removesuffix("_PERP")
    base_code, quote_code = split_linear_perp_symbol(raw_symbol)
    price_precision, size_precision = infer_price_size_precision(paths)
    price_increment = "1" if price_precision == 0 else f"0.{'0' * (price_precision - 1)}1"
    size_increment = "1" if size_precision == 0 else f"0.{'0' * (size_precision - 1)}1"
    max_price = "1000000000" if price_precision == 0 else f"1000000000.{'0' * price_precision}"
    quote_currency = Currency.from_str(quote_code)

    return CryptoPerpetual(
        instrument_id=InstrumentId(
            symbol=Symbol(f"{raw_symbol}-PERP"),
            venue=Venue("BINANCE"),
        ),
        raw_symbol=Symbol(raw_symbol),
        base_currency=Currency.from_str(base_code),
        quote_currency=quote_currency,
        settlement_currency=quote_currency,
        is_inverse=False,
        price_precision=price_precision,
        price_increment=Price.from_str(price_increment),
        size_precision=size_precision,
        size_increment=Quantity.from_str(size_increment),
        max_quantity=None,
        min_quantity=Quantity.from_str(size_increment),
        max_notional=None,
        min_notional=Money(0, quote_currency),
        max_price=Price.from_str(max_price),
        min_price=Price.from_str(price_increment),
        margin_init=Decimal("0.0500"),
        margin_maint=Decimal("0.0250"),
        maker_fee=Decimal("0.000200"),
        taker_fee=Decimal("0.000500"),
        ts_event=0,
        ts_init=0,
    )


def make_bar_type(instrument, target_bars_per_day: int) -> BarType:
    return BarType(
        instrument.id,
        BarSpecification(target_bars_per_day, BarAggregation.VALUE_RUNS, PriceType.LAST),
    )


def daily_archive_path(input_dir: Path, product: str, symbol: str, day: date) -> Path:
    return input_dir / f"futures_{product}" / "aggTrades" / symbol / f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip"


def monthly_archive_path(input_dir: Path, product: str, symbol: str, month_day: date) -> Path:
    return (
        input_dir
        / f"futures_{product}"
        / "monthly"
        / "aggTrades"
        / symbol
        / f"{symbol}-aggTrades-{month_day:%Y-%m}.zip"
    )


def archive_paths_for_range(
    *,
    input_dir: Path,
    product: str,
    symbol: str,
    start: date,
    end: date,
) -> list[Path]:
    if end < start:
        return []

    paths: list[Path] = []
    missing: list[Path] = []
    current = start
    while current <= end:
        monthly_path = monthly_archive_path(input_dir, product, symbol, current)
        if monthly_path.exists():
            if monthly_path not in paths:
                paths.append(monthly_path)
            month_last = calendar.monthrange(current.year, current.month)[1]
            current = date(current.year, current.month, month_last) + timedelta(days=1)
            continue

        daily_path = daily_archive_path(input_dir, product, symbol, current)
        if daily_path.exists():
            paths.append(daily_path)
        else:
            missing.append(daily_path)
        current += timedelta(days=1)

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} input archives, first missing: {missing[0]}")
    return paths


def pyarrow_modules():
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def tick_cache_path_for_archive(path: Path, cache_dir: Path) -> Path:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.startswith("futures_"):
            return cache_dir / Path(*parts[index:]).with_suffix(".parquet")
    return cache_dir / path.with_suffix(".parquet").name


def tick_cache_metadata(path: Path) -> dict[bytes, bytes]:
    stat = path.stat()
    return {
        b"schema_version": TICK_CACHE_SCHEMA_VERSION.encode("ascii"),
        b"source_path": str(path.resolve()).encode("utf-8"),
        b"source_size": str(stat.st_size).encode("ascii"),
        b"source_mtime_ns": str(stat.st_mtime_ns).encode("ascii"),
    }


def tick_cache_is_current(source_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    _, pq = pyarrow_modules()
    expected = tick_cache_metadata(source_path)
    try:
        actual = pq.read_metadata(cache_path).metadata or {}
    except Exception:
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def tick_cache_schema(source_path: Path):
    pa, _ = pyarrow_modules()
    return pa.schema(
        [
            ("trade_id", pa.int64()),
            ("price", pa.float64()),
            ("quantity", pa.float64()),
            ("ts_ms", pa.int64()),
            ("buyer_maker", pa.bool_()),
        ],
        metadata=tick_cache_metadata(source_path),
    )


def write_tick_cache(source_path: Path, cache_path: Path) -> int:
    pa, pq = pyarrow_modules()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    schema = tick_cache_schema(source_path)
    writer = None
    rows_written = 0
    trade_ids: list[int] = []
    prices: list[float] = []
    quantities: list[float] = []
    ts_ms_values: list[int] = []
    buyer_makers: list[bool] = []

    def flush() -> None:
        nonlocal writer
        nonlocal rows_written
        if not trade_ids:
            return
        table = pa.Table.from_pydict(
            {
                "trade_id": trade_ids,
                "price": prices,
                "quantity": quantities,
                "ts_ms": ts_ms_values,
                "buyer_maker": buyer_makers,
            },
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
        writer.write_table(table)
        rows_written += len(trade_ids)
        trade_ids.clear()
        prices.clear()
        quantities.clear()
        ts_ms_values.clear()
        buyer_makers.clear()

    print(f"building tick cache {source_path.name} -> {cache_path}", flush=True)
    try:
        for row, pos in read_zip_rows(source_path):
            trade_ids.append(int(row[pos["agg_trade_id"]]))
            prices.append(float(row[pos["price"]]))
            quantities.append(float(row[pos["quantity"]]))
            ts_ms_values.append(int(row[pos["transact_time"]]))
            buyer_makers.append(parse_bool(row[pos["is_buyer_maker"]]))
            if len(trade_ids) >= TICK_CACHE_WRITE_BATCH_SIZE:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        empty = pa.Table.from_pydict(
            {
                "trade_id": [],
                "price": [],
                "quantity": [],
                "ts_ms": [],
                "buyer_maker": [],
            },
            schema=schema,
        )
        pq.write_table(empty, tmp_path, compression="snappy")

    tmp_path.replace(cache_path)
    print(f"wrote tick cache rows={rows_written:,} -> {cache_path}", flush=True)
    return rows_written


def ensure_tick_cache(source_path: Path, cache_dir: Path) -> Path:
    cache_path = tick_cache_path_for_archive(source_path, cache_dir)
    if tick_cache_is_current(source_path, cache_path):
        return cache_path
    write_tick_cache(source_path, cache_path)
    return cache_path


def iter_cached_trade_ticks(
    paths,
    instrument,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    cache_dir: Path = TICK_CACHE_DIR,
):
    _, pq = pyarrow_modules()
    for path in paths:
        cache_path = ensure_tick_cache(path, cache_dir)
        print(f"streaming cache {cache_path.name}")
        parquet_file = pq.ParquetFile(cache_path)
        for batch in parquet_file.iter_batches(
            batch_size=TICK_CACHE_READ_BATCH_SIZE,
            columns=["trade_id", "price", "quantity", "ts_ms", "buyer_maker"],
        ):
            names = batch.schema.names
            trade_ids = batch.column(names.index("trade_id")).to_numpy(zero_copy_only=False)
            prices = batch.column(names.index("price")).to_numpy(zero_copy_only=False)
            quantities = batch.column(names.index("quantity")).to_numpy(zero_copy_only=False)
            ts_ms_values = batch.column(names.index("ts_ms")).to_numpy(zero_copy_only=False)
            buyer_makers = batch.column(names.index("buyer_maker")).to_numpy(zero_copy_only=False)
            for index in range(batch.num_rows):
                ts_ms = int(ts_ms_values[index])
                if start_ms is not None and ts_ms < start_ms:
                    continue
                if end_ms is not None and ts_ms >= end_ms:
                    return

                buyer_maker = bool(buyer_makers[index])
                ts_ns = millis_to_nanos(ts_ms)
                yield TradeTick(
                    instrument_id=instrument.id,
                    price=instrument.make_price(float(prices[index])),
                    size=instrument.make_qty(float(quantities[index])),
                    aggressor_side=AggressorSide.SELLER if buyer_maker else AggressorSide.BUYER,
                    trade_id=TradeId(str(int(trade_ids[index]))),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )


def iter_trade_ticks(
    paths,
    instrument,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    use_cache: bool = USE_TICK_CACHE,
    cache_dir: Path = TICK_CACHE_DIR,
):
    if use_cache:
        yield from iter_cached_trade_ticks(
            paths,
            instrument,
            start_ms=start_ms,
            end_ms=end_ms,
            cache_dir=cache_dir,
        )
        return

    for path in paths:
        print(f"streaming {path.name}")
        for row, pos in read_zip_rows(path):
            ts_ms = int(row[pos["transact_time"]])
            if start_ms is not None and ts_ms < start_ms:
                continue
            if end_ms is not None and ts_ms >= end_ms:
                return

            buyer_maker = parse_bool(row[pos["is_buyer_maker"]])
            ts_ns = millis_to_nanos(ts_ms)
            yield TradeTick(
                instrument_id=instrument.id,
                price=Price.from_str(row[pos["price"]]),
                size=Quantity.from_str(row[pos["quantity"]]),
                aggressor_side=AggressorSide.SELLER if buyer_maker else AggressorSide.BUYER,
                trade_id=TradeId(row[pos["agg_trade_id"]]),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )


def load_trade_ticks(
    paths,
    instrument,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    use_cache: bool = USE_TICK_CACHE,
    cache_dir: Path = TICK_CACHE_DIR,
) -> list[TradeTick]:
    ticks = list(
        iter_trade_ticks(
            paths,
            instrument,
            start_ms=start_ms,
            end_ms=end_ms,
            use_cache=use_cache,
            cache_dir=cache_dir,
        ),
    )
    ticks.sort(key=lambda tick: (tick.ts_event, tick.trade_id.value))
    return ticks


def next_month_start(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def threshold_update_dates(start: date, end: date, frequency: str) -> list[date]:
    frequency = frequency.strip().lower()
    if frequency not in {"monthly", "weekly"}:
        raise ValueError("THRESHOLD_UPDATE_FREQUENCY must be 'monthly' or 'weekly'")

    updates = [start]
    if frequency == "monthly":
        current = next_month_start(start)
        while current <= end:
            updates.append(current)
            current = next_month_start(current)
    else:
        current = start + timedelta(days=7)
        while current <= end:
            updates.append(current)
            current += timedelta(days=7)
    return updates


def daily_notional_from_ticks(
    paths,
    instrument,
    *,
    start: date,
    end: date,
) -> dict[str, float]:
    totals = {day.isoformat(): 0.0 for day in iter_days(start, end)}
    start_ms = date_to_ms(start)
    end_ms = next_date_ms(end)
    if not USE_TICK_CACHE:
        for tick in iter_trade_ticks(
            paths,
            instrument,
            start_ms=start_ms,
            end_ms=end_ms,
            use_cache=False,
        ):
            totals[day_key(int(tick.ts_event))] += float(tick.price) * float(tick.size)
        return totals

    _, pq = pyarrow_modules()
    day_ms = 86_400_000
    for path in paths:
        cache_path = ensure_tick_cache(path, TICK_CACHE_DIR)
        print(f"summing daily notional {cache_path.name}")
        parquet_file = pq.ParquetFile(cache_path)
        for batch in parquet_file.iter_batches(
            batch_size=TICK_CACHE_READ_BATCH_SIZE,
            columns=["price", "quantity", "ts_ms"],
        ):
            names = batch.schema.names
            prices = batch.column(names.index("price")).to_numpy(zero_copy_only=False)
            quantities = batch.column(names.index("quantity")).to_numpy(zero_copy_only=False)
            ts_ms_values = batch.column(names.index("ts_ms")).to_numpy(zero_copy_only=False)
            mask = (ts_ms_values >= start_ms) & (ts_ms_values < end_ms)
            if not mask.any():
                continue
            day_numbers = ts_ms_values[mask] // day_ms
            notionals = prices[mask] * quantities[mask]
            unique_days, inverse = np.unique(day_numbers, return_inverse=True)
            sums = np.bincount(inverse, weights=notionals)
            for day_number, total in zip(unique_days, sums, strict=True):
                day = datetime.fromtimestamp(int(day_number) * 86_400, tz=UTC).date()
                key = day.isoformat()
                if key in totals:
                    totals[key] += float(total)
    return totals


def build_threshold_schedule(
    daily_notional: dict[str, float],
    *,
    start: date,
    end: date,
) -> list[dict]:
    updates = threshold_update_dates(start, end, THRESHOLD_UPDATE_FREQUENCY)
    rows: list[dict] = []
    for index, effective_start in enumerate(updates):
        effective_end = (
            updates[index + 1] - timedelta(days=1) if index + 1 < len(updates) else end
        )
        lookback_end = effective_start - timedelta(days=1)
        lookback_start = lookback_end - timedelta(days=THRESHOLD_ROLLING_WINDOW_DAYS - 1)
        lookback_days = list(iter_days(lookback_start, lookback_end))
        values = [
            float(daily_notional.get(day.isoformat(), 0.0))
            for day in lookback_days
            if float(daily_notional.get(day.isoformat(), 0.0)) > 0.0
        ]
        if len(values) < THRESHOLD_MIN_LOOKBACK_DAYS:
            raise ValueError(
                f"Not enough positive daily notional values for {effective_start}: "
                f"{len(values)} < THRESHOLD_MIN_LOOKBACK_DAYS={THRESHOLD_MIN_LOOKBACK_DAYS}. "
                "Move GENERATION_START later or download more lookback history.",
            )
        daily_sma = sum(values) / len(values)
        rows.append(
            {
                "effective_start": effective_start.isoformat(),
                "effective_end": effective_end.isoformat(),
                "lookback_start": lookback_start.isoformat(),
                "lookback_end": lookback_end.isoformat(),
                "lookback_days": len(values),
                "daily_notional_sma": daily_sma,
                "threshold": daily_sma / THRESHOLD_DAILY_NOTIONAL_DIVISOR,
            },
        )
    return rows


class NotionalThresholdSchedule:
    def __init__(self, rows: list[dict]) -> None:
        if not rows:
            raise ValueError("threshold schedule cannot be empty")
        self.rows = sorted(rows, key=lambda row: row["effective_start"])
        self.starts_ns = [
            millis_to_nanos(date_to_ms(date.fromisoformat(row["effective_start"])))
            for row in self.rows
        ]
        self.thresholds = [float(row["threshold"]) for row in self.rows]

    def __call__(self, ts_ns: int) -> float | None:
        position = bisect_right(self.starts_ns, int(ts_ns)) - 1
        if position < 0:
            return None
        return self.thresholds[position]


class StreamingDrbCsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.count = 0
        self.daily_counts: dict[str, int] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self.file = self.tmp_path.open("w", newline="", encoding="utf-8")
        fields = [
            "bar_type",
            "ts_event",
            "ts_event_ns",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ticks",
            "buy_ticks",
            "sell_ticks",
            "buy_notional",
            "sell_notional",
            "signed_notional",
            "theta",
            "threshold",
            "threshold_scale",
            "expected_ticks",
            "expected_metric_a",
            "expected_metric_b",
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def handle(self, bar: Bar, meta: AfmlBarMeta) -> None:
        self.writer.writerow(
            {
                "bar_type": "AFML_DRB",
                "ts_event": ns_to_iso(bar.ts_event),
                "ts_event_ns": bar.ts_event,
                "open": str(bar.open),
                "high": str(bar.high),
                "low": str(bar.low),
                "close": str(bar.close),
                "volume": str(bar.volume),
                "ticks": meta.ticks,
                "buy_ticks": meta.buy_ticks,
                "sell_ticks": meta.sell_ticks,
                "buy_notional": f"{meta.buy_notional:.8f}",
                "sell_notional": f"{meta.sell_notional:.8f}",
                "signed_notional": f"{meta.signed_notional:.8f}",
                "theta": f"{meta.theta:.8f}",
                "threshold": f"{meta.threshold:.8f}",
                "threshold_scale": f"{meta.threshold_scale:.8f}",
                "expected_ticks": f"{meta.expected_ticks:.8f}",
                "expected_metric_a": f"{meta.expected_metric_a:.8f}",
                "expected_metric_b": f"{meta.expected_metric_b:.8f}",
            },
        )
        self.count += 1
        key = day_key(int(bar.ts_event))
        self.daily_counts[key] = self.daily_counts.get(key, 0) + 1

    def close(self) -> None:
        self.file.close()

    def commit(self) -> None:
        self.tmp_path.replace(self.path)


class NullCatalogSink:
    catalog_path: Path | None = None
    flushes = 0
    bars_written = 0

    def append(self, bar: Bar) -> None:
        return

    def close(self) -> None:
        return


class MonthlyCatalogSink:
    def __init__(
        self,
        *,
        catalog_path: Path,
        append_existing: bool,
        instrument,
    ) -> None:
        self.catalog_path = catalog_path
        if self.catalog_path.exists() and BACKUP_EXISTING_CATALOG and not append_existing:
            backup_path = backup_existing_path(self.catalog_path)
            print(f"backed up existing catalog -> {backup_path}")
        elif self.catalog_path.exists() and append_existing:
            print(f"appending catalog -> {self.catalog_path}")
        self.catalog = ParquetDataCatalog(self.catalog_path)
        self.catalog.write_data([instrument])
        self.instrument = instrument
        self.current_month: str | None = None
        self.buffer: list[Bar] = []
        self.flushes = 0
        self.bars_written = 0

    def append(self, bar: Bar) -> None:
        key = month_key(int(bar.ts_event))
        if self.current_month is None:
            self.current_month = key
        elif key != self.current_month:
            self.flush()
            self.current_month = key
        self.buffer.append(self._normalize_bar_precision(bar))

    def _normalize_bar_precision(self, bar: Bar) -> Bar:
        return Bar(
            bar_type=bar.bar_type,
            open=self.instrument.make_price(float(bar.open)),
            high=self.instrument.make_price(float(bar.high)),
            low=self.instrument.make_price(float(bar.low)),
            close=self.instrument.make_price(float(bar.close)),
            volume=self.instrument.make_qty(float(bar.volume)),
            ts_event=bar.ts_event,
            ts_init=bar.ts_init,
        )

    def flush(self) -> None:
        if not self.buffer:
            return
        self.catalog.write_data(list(self.buffer))
        self.bars_written += len(self.buffer)
        self.flushes += 1
        print(
            f"flushed catalog month={self.current_month} "
            f"bars={len(self.buffer):,} total={self.bars_written:,} -> {self.catalog_path}",
        )
        self.buffer.clear()

    def close(self) -> None:
        self.flush()


def backup_existing_path(path: Path) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.backup_{stamp}")
    suffix = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.backup_{stamp}_{suffix}")
        suffix += 1
    path.rename(backup_path)
    return backup_path


def threshold_plan_path(symbol: str) -> Path:
    stem = (
        f"{symbol}_{THRESHOLD_HISTORY_START.replace('-', '')}_{GENERATION_START.replace('-', '')}"
        f"_{GENERATION_END.replace('-', '')}_{THRESHOLD_ROLLING_WINDOW_DAYS}d"
        f"_{THRESHOLD_UPDATE_FREQUENCY}_{TARGET_BARS_PER_DAY}tpd_DRB_threshold_plan.json"
    )
    return CALIBRATION_OUTPUT_DIR / symbol / stem


def expected_threshold_plan_config(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "kind": "DRB",
        "threshold_method": "rolling_daily_notional_sma",
        "threshold_history_start": THRESHOLD_HISTORY_START,
        "generation_start": GENERATION_START,
        "generation_end": GENERATION_END,
        "threshold_rolling_window_days": THRESHOLD_ROLLING_WINDOW_DAYS,
        "threshold_daily_notional_divisor": THRESHOLD_DAILY_NOTIONAL_DIVISOR,
        "threshold_update_frequency": THRESHOLD_UPDATE_FREQUENCY,
        "threshold_min_lookback_days": THRESHOLD_MIN_LOOKBACK_DAYS,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "product": PRODUCT,
        "ewma_span": EWMA_SPAN,
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
        "write_catalog": WRITE_CATALOG,
    }


def threshold_plan_mismatch_reason(payload: dict, symbol: str) -> str | None:
    for key, expected_value in expected_threshold_plan_config(symbol).items():
        if payload.get(key) != expected_value:
            return f"{key} cached={payload.get(key)!r} current={expected_value!r}"
    if "threshold_schedule" not in payload:
        return "missing threshold_schedule"
    return None


def reusable_daily_notional_mismatch_reason(payload: dict, symbol: str) -> str | None:
    expected = expected_threshold_plan_config(symbol)
    reusable_keys = {
        "symbol",
        "kind",
        "threshold_method",
        "threshold_history_start",
        "threshold_rolling_window_days",
        "threshold_daily_notional_divisor",
        "threshold_update_frequency",
        "threshold_min_lookback_days",
        "target_bars_per_day",
        "product",
    }
    for key in reusable_keys:
        if payload.get(key) != expected[key]:
            return f"{key} cached={payload.get(key)!r} current={expected[key]!r}"
    if not isinstance(payload.get("daily_notional"), dict):
        return "missing daily_notional"
    return None


def latest_reusable_threshold_plan(symbol: str) -> dict | None:
    plan_dir = CALIBRATION_OUTPUT_DIR / symbol
    if not plan_dir.exists():
        return None

    latest_payload: dict | None = None
    latest_end: date | None = None
    for path in plan_dir.glob("*_DRB_threshold_plan.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if reusable_daily_notional_mismatch_reason(payload, symbol) is not None:
            continue
        end_value = payload.get("generation_end")
        if not isinstance(end_value, str):
            continue
        end_date = parse_date(end_value)
        if latest_end is None or end_date > latest_end:
            latest_payload = payload
            latest_end = end_date
    return latest_payload


def latest_daily_notional_date(daily_notional: dict[str, float]) -> date | None:
    latest: date | None = None
    for key in daily_notional:
        parsed = parse_date(key)
        if latest is None or parsed > latest:
            latest = parsed
    return latest


def load_incremental_daily_notional(
    *,
    symbol: str,
    instrument,
    history_start: date,
    generation_end: date,
) -> dict[str, float]:
    daily_notional: dict[str, float] = {}
    previous_plan = latest_reusable_threshold_plan(symbol)
    if previous_plan is not None:
        daily_notional = {
            str(day): float(value)
            for day, value in previous_plan.get("daily_notional", {}).items()
        }
        latest_day = latest_daily_notional_date(daily_notional)
        latest_text = latest_day.isoformat() if latest_day is not None else "none"
        print(f"{symbol} incremental: reused daily notional through {latest_text}")
    else:
        latest_day = None

    missing_start = history_start if latest_day is None else max(history_start, latest_day + timedelta(days=1))
    if missing_start > generation_end:
        return daily_notional

    paths = archive_paths_for_range(
        input_dir=INPUT_DIR,
        product=PRODUCT,
        symbol=symbol,
        start=missing_start,
        end=generation_end,
    )
    daily_notional.update(
        daily_notional_from_ticks(
            paths,
            instrument,
            start=missing_start,
            end=generation_end,
        ),
    )
    return daily_notional


def load_or_create_threshold_plan(symbol_arg: str) -> dict:
    started = time.perf_counter()
    symbol = normalize_symbol(symbol_arg)
    output_path = threshold_plan_path(symbol)
    if RESUME_CALIBRATION and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        mismatch_reason = threshold_plan_mismatch_reason(payload, symbol)
        if mismatch_reason is None:
            print(f"{symbol} resume: using DRB threshold plan -> {output_path}")
            return payload
        print(f"{symbol} resume: ignoring stale DRB threshold plan ({mismatch_reason}) -> {output_path}")

    history_start = parse_date(THRESHOLD_HISTORY_START)
    generation_start = parse_date(GENERATION_START)
    generation_end = parse_date(GENERATION_END)
    if generation_start <= history_start:
        raise SystemExit("GENERATION_START must be after THRESHOLD_HISTORY_START")
    paths = archive_paths_for_range(
        input_dir=INPUT_DIR,
        product=PRODUCT,
        symbol=symbol,
        start=history_start,
        end=generation_end,
    )
    instrument = make_binance_perpetual(symbol, paths)
    daily_notional = load_incremental_daily_notional(
        symbol=symbol,
        instrument=instrument,
        history_start=history_start,
        generation_end=generation_end,
    )
    threshold_schedule = build_threshold_schedule(
        daily_notional,
        start=generation_start,
        end=generation_end,
    )
    payload = {
        **expected_threshold_plan_config(symbol),
        "use_tick_cache": USE_TICK_CACHE,
        "tick_cache_dir": str(TICK_CACHE_DIR),
        "daily_notional": daily_notional,
        "threshold_schedule": threshold_schedule,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{symbol} threshold plan periods={len(threshold_schedule)} "
        f"first_threshold={threshold_schedule[0]['threshold']:.2f}",
    )
    print(f"{symbol} wrote threshold plan -> {output_path}")
    return payload


def output_label(symbol: str, start: date, end: date) -> str:
    return f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_{TARGET_BARS_PER_DAY}tpd"


def cumulative_catalog_path(symbol: str, end: date) -> Path:
    start = parse_date(GENERATION_START_FALLBACK if INCREMENTAL_ENABLED else GENERATION_START_CONFIG)
    return CATALOG_DIR / symbol / output_label(symbol, start, end)


def summary_matches_generation_config(payload: dict, symbol: str) -> bool:
    expected = {
        "symbol": symbol,
        "kind": "DRB",
        "threshold_method": "rolling_daily_notional_sma",
        "threshold_history_start": THRESHOLD_HISTORY_START,
        "threshold_rolling_window_days": THRESHOLD_ROLLING_WINDOW_DAYS,
        "threshold_daily_notional_divisor": THRESHOLD_DAILY_NOTIONAL_DIVISOR,
        "threshold_update_frequency": THRESHOLD_UPDATE_FREQUENCY,
        "threshold_min_lookback_days": THRESHOLD_MIN_LOOKBACK_DAYS,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "ewma_span": EWMA_SPAN,
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
        "write_catalog": WRITE_CATALOG,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def matching_generated_summaries(symbol_arg: str) -> list[dict]:
    symbol = normalize_symbol(symbol_arg)
    symbol_dir = OUTPUT_DIR / symbol
    if not symbol_dir.exists():
        return []

    summaries: list[dict] = []
    for summary_path in symbol_dir.glob("*_DRB_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not summary_matches_generation_config(payload, symbol):
            continue
        start_value = payload.get("start_date")
        end_date = payload.get("end_date")
        if not isinstance(start_value, str) or not isinstance(end_date, str):
            continue
        try:
            start = parse_date(start_value)
            end = parse_date(end_date)
        except ValueError:
            continue
        catalog_path_value = payload.get("catalog_path")
        summaries.append(
            {
                "summary_path": summary_path,
                "payload": payload,
                "start": start,
                "end": end,
                "catalog_path": Path(catalog_path_value) if isinstance(catalog_path_value, str) else None,
            },
        )
    return sorted(summaries, key=lambda item: (item["start"], item["end"]))


def latest_generated_summary_end(symbol_arg: str) -> date | None:
    latest: date | None = None
    for summary in matching_generated_summaries(symbol_arg):
        end = summary["end"]
        if latest is None or end > latest:
            latest = end
    return latest


def generation_range_for_symbol(symbol_arg: str) -> tuple[date, date]:
    end = resolve_generation_end(GENERATION_END_CONFIG, incremental=INCREMENTAL_ENABLED)
    if INCREMENTAL_ENABLED:
        latest_end = latest_generated_summary_end(symbol_arg)
        start = latest_end + timedelta(days=1) if latest_end is not None else parse_date(GENERATION_START_FALLBACK)
    else:
        start = parse_date(GENERATION_START_CONFIG)
    return start, end


def base_catalog_summary(symbol_arg: str) -> dict | None:
    summaries = [
        summary
        for summary in matching_generated_summaries(symbol_arg)
        if summary["catalog_path"] is not None and summary["catalog_path"].exists()
    ]
    if not summaries:
        return None

    fallback_start = parse_date(GENERATION_START_FALLBACK)
    for summary in summaries:
        if summary["start"] == fallback_start:
            return summary
    return summaries[0]


def catalog_path_for_generation(symbol: str, label: str) -> tuple[Path, bool]:
    default_path = CATALOG_DIR / symbol / label
    if not INCREMENTAL_ENABLED:
        return default_path, False

    summary = base_catalog_summary(symbol)
    if summary is None:
        return default_path, False
    return summary["catalog_path"], True


def update_matching_summary_catalog_paths(symbol_arg: str, catalog_path: Path) -> None:
    catalog_path_text = str(catalog_path)
    for summary in matching_generated_summaries(symbol_arg):
        payload = summary["payload"]
        if payload.get("catalog_path") == catalog_path_text:
            continue
        payload["catalog_path"] = catalog_path_text
        summary["summary_path"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def paths_equal(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def copy_catalog_files(source: Path, destination: Path) -> int:
    copied = 0
    for source_file in source.rglob("*.parquet"):
        relative_path = source_file.relative_to(source)
        destination_file = destination / relative_path
        if destination_file.exists():
            if destination_file.stat().st_size != source_file.stat().st_size:
                raise RuntimeError(
                    f"Conflicting catalog file already exists: {destination_file}",
                )
            continue
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        copied += 1
    return copied


def move_catalog_path(source: Path, destination: Path) -> Path:
    if paths_equal(source, destination):
        return destination
    if destination.exists():
        return destination
    if not source.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    print(f"renamed catalog -> {destination}")
    return destination


def finalize_incremental_catalog_path(symbol: str, catalog_path: Path, end: date) -> Path:
    if not INCREMENTAL_ENABLED or not WRITE_CATALOG:
        return catalog_path

    final_path = move_catalog_path(catalog_path, cumulative_catalog_path(symbol, end))
    update_matching_summary_catalog_paths(symbol, final_path)
    return final_path


def repair_split_incremental_catalogs(symbol_arg: str) -> None:
    if not INCREMENTAL_ENABLED or not WRITE_CATALOG:
        return

    base_summary = base_catalog_summary(symbol_arg)
    if base_summary is None:
        return
    base_path = base_summary["catalog_path"]
    copied = 0
    for summary in matching_generated_summaries(symbol_arg):
        catalog_path = summary["catalog_path"]
        if catalog_path is None or not catalog_path.exists() or paths_equal(catalog_path, base_path):
            continue
        copied += copy_catalog_files(catalog_path, base_path)
    if copied:
        print(f"{normalize_symbol(symbol_arg)} repaired split catalog files={copied} -> {base_path}")
    latest_end = latest_generated_summary_end(symbol_arg)
    if latest_end is not None:
        final_path = finalize_incremental_catalog_path(
            normalize_symbol(symbol_arg),
            base_path,
            latest_end,
        )
        update_matching_summary_catalog_paths(symbol_arg, final_path)


def completed_csv_summary_matches(  # noqa: C901
    *,
    summary_path: Path,
    output_path: Path,
    tmp_path: Path,
    symbol: str,
    start: date,
    end: date,
    threshold_payload: dict,
) -> dict | None:
    if not summary_path.exists() or not output_path.exists() or tmp_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "symbol": symbol,
        "kind": "DRB",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "threshold_method": "rolling_daily_notional_sma",
        "threshold_rolling_window_days": THRESHOLD_ROLLING_WINDOW_DAYS,
        "threshold_daily_notional_divisor": THRESHOLD_DAILY_NOTIONAL_DIVISOR,
        "threshold_update_frequency": THRESHOLD_UPDATE_FREQUENCY,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "ewma_span": int(threshold_payload["ewma_span"]),
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
    }
    for key, expected_value in expected.items():
        actual_value = summary.get(key)
        if isinstance(expected_value, float):
            if actual_value is None or abs(float(actual_value) - expected_value) > 1e-12:
                return None
        elif actual_value != expected_value:
            return None
    if summary.get("threshold_schedule") != threshold_payload.get("threshold_schedule"):
        return None
    if int(summary.get("bars", -1)) <= 0:
        return None
    if WRITE_CATALOG:
        catalog_path = summary.get("catalog_path")
        if catalog_path is None or not Path(catalog_path).exists():
            return None
    return summary


def generate_drb_for_symbol(symbol_arg: str, threshold_payload: dict) -> None:
    started = time.perf_counter()
    symbol = normalize_symbol(symbol_arg)
    start = parse_date(GENERATION_START)
    end = parse_date(GENERATION_END)
    if end < start:
        raise SystemExit("GENERATION_END must be on or after GENERATION_START")

    history_start = parse_date(THRESHOLD_HISTORY_START)
    expectation_start = start - timedelta(days=THRESHOLD_ROLLING_WINDOW_DAYS)
    if expectation_start < history_start:
        expectation_start = history_start
    expectation_end = start - timedelta(days=1)
    ewma_span = int(threshold_payload["ewma_span"])
    threshold_schedule = threshold_payload["threshold_schedule"]
    threshold_provider = NotionalThresholdSchedule(threshold_schedule)
    days = (end - start).days + 1
    target_bars_total = TARGET_BARS_PER_DAY * days

    label = output_label(symbol, start, end)
    catalog_path, append_existing_catalog = catalog_path_for_generation(symbol, label)
    output_dir = OUTPUT_DIR / symbol
    output_path = output_dir / f"{label}_DRB.csv"
    summary_path = output_dir / f"{label}_DRB_summary.json"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if RESUME_COMPLETED_CSV:
        existing_summary = completed_csv_summary_matches(
            summary_path=summary_path,
            output_path=output_path,
            tmp_path=tmp_path,
            symbol=symbol,
            start=start,
            end=end,
            threshold_payload=threshold_payload,
        )
        if existing_summary is not None:
            print(
                f"{symbol} resume: found completed DRB CSV "
                f"bars={int(existing_summary['bars']):,} -> {output_path}",
            )
            return

    generation_paths = archive_paths_for_range(
        input_dir=INPUT_DIR,
        product=PRODUCT,
        symbol=symbol,
        start=start,
        end=end,
    )
    instrument = make_binance_perpetual(symbol, generation_paths)

    expectation_paths = archive_paths_for_range(
        input_dir=INPUT_DIR,
        product=PRODUCT,
        symbol=symbol,
        start=expectation_start,
        end=expectation_end,
    )
    expectation_ticks = load_trade_ticks(
        expectation_paths,
        instrument,
        start_ms=date_to_ms(expectation_start),
        end_ms=next_date_ms(expectation_end),
        use_cache=USE_TICK_CACHE,
        cache_dir=TICK_CACHE_DIR,
    )
    expectation_days = (expectation_end - expectation_start).days + 1
    expectation_target = TARGET_BARS_PER_DAY * expectation_days
    expectations = estimate_afml_dollar_expectations(expectation_ticks, expectation_target)

    print(
        f"{symbol} DRB generation={start:%Y-%m-%d}..{end:%Y-%m-%d} "
        f"threshold=daily_notional_sma({THRESHOLD_ROLLING_WINDOW_DAYS}d)/"
        f"{THRESHOLD_DAILY_NOTIONAL_DIVISOR} "
        f"update={THRESHOLD_UPDATE_FREQUENCY} ewma_span={ewma_span}",
    )

    writer = StreamingDrbCsvWriter(output_path)
    catalog_sink = (
        MonthlyCatalogSink(
            catalog_path=catalog_path,
            append_existing=append_existing_catalog,
            instrument=instrument,
        )
        if WRITE_CATALOG
        else NullCatalogSink()
    )
    bar_type = make_bar_type(instrument, TARGET_BARS_PER_DAY)
    aggregator = AfmlDollarRunsBarAggregator(
        instrument=instrument,
        bar_type=bar_type,
        handler=catalog_sink.append,
        meta_handler=writer.handle,
        expectations=expectations,
        ewma_span=ewma_span,
        threshold_scale=1.0,
        update_expected_ticks=UPDATE_EXPECTED_TICKS,
        threshold_provider=threshold_provider,
    )

    ticks_processed = 0
    try:
        for tick in iter_trade_ticks(
            generation_paths,
            instrument,
            start_ms=date_to_ms(start),
            end_ms=next_date_ms(end),
            use_cache=USE_TICK_CACHE,
            cache_dir=TICK_CACHE_DIR,
        ):
            aggregator.handle_trade_tick(tick)
            ticks_processed += 1
            if ticks_processed % 5_000_000 == 0:
                print(f"processed ticks={ticks_processed:,} DRB={writer.count:,}")
    finally:
        catalog_sink.close()
        writer.close()

    writer.commit()
    final_catalog_path = catalog_sink.catalog_path
    if final_catalog_path is not None:
        final_catalog_path = finalize_incremental_catalog_path(symbol, final_catalog_path, end)
        catalog_sink.catalog_path = final_catalog_path

    summary = {
        "symbol": symbol,
        "instrument_id": str(instrument.id),
        "kind": "DRB",
        "source": "Binance Vision futures aggregate trades",
        "input_archives": [str(path) for path in generation_paths],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": days,
        "ticks": ticks_processed,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "target_bars_total": target_bars_total,
        "bar_type": str(bar_type),
        "threshold_method": "rolling_daily_notional_sma",
        "threshold_history_start": THRESHOLD_HISTORY_START,
        "threshold_rolling_window_days": THRESHOLD_ROLLING_WINDOW_DAYS,
        "threshold_daily_notional_divisor": THRESHOLD_DAILY_NOTIONAL_DIVISOR,
        "threshold_update_frequency": THRESHOLD_UPDATE_FREQUENCY,
        "threshold_min_lookback_days": THRESHOLD_MIN_LOOKBACK_DAYS,
        "threshold_schedule": threshold_schedule,
        "ewma_span": ewma_span,
        "initial_expectation_start_date": expectation_start.isoformat(),
        "initial_expectation_end_date": expectation_end.isoformat(),
        "initial_expectation_days": expectation_days,
        "initial_expectation_target_bars": expectation_target,
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
        "write_catalog": WRITE_CATALOG,
        "use_tick_cache": USE_TICK_CACHE,
        "tick_cache_dir": str(TICK_CACHE_DIR),
        "catalog_path": str(final_catalog_path) if final_catalog_path is not None else None,
        "catalog_append_existing": append_existing_catalog,
        "catalog_flush_granularity": "monthly" if WRITE_CATALOG else None,
        "catalog_flushes": catalog_sink.flushes,
        "catalog_bars_written": catalog_sink.bars_written,
        "bars": writer.count,
        "bars_per_day_avg": writer.count / days,
        "daily_counts": writer.daily_counts,
        "path": str(output_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote DRB bars={writer.count:,} avg/day={writer.count / days:.2f} -> {output_path}")
    if final_catalog_path is not None:
        print(f"wrote Nautilus catalog -> {final_catalog_path}")
    print(f"wrote summary -> {summary_path}")


def main() -> None:
    target_end = resolve_generation_end(GENERATION_END_CONFIG, incremental=INCREMENTAL_ENABLED)
    print(
        "AFML DRB from Binance aggTrades "
        f"threshold_history={THRESHOLD_HISTORY_START}..{target_end:%Y-%m-%d} "
        f"threshold=daily_notional_sma({THRESHOLD_ROLLING_WINDOW_DAYS}d)/"
        f"{THRESHOLD_DAILY_NOTIONAL_DIVISOR} "
        f"update={THRESHOLD_UPDATE_FREQUENCY} "
        f"product={PRODUCT} "
        f"tick_cache={'on' if USE_TICK_CACHE else 'off'} "
        f"catalog={'on' if WRITE_CATALOG else 'off'}",
    )
    for index, symbol in enumerate(SYMBOLS, start=1):
        normalized = normalize_symbol(symbol)
        repair_split_incremental_catalogs(symbol)
        start, end = generation_range_for_symbol(symbol)
        if start > end:
            print(f"\n[{index}/{len(SYMBOLS)}] {normalized} already current through {end:%Y-%m-%d}")
            continue
        set_generation_range(start, end)
        print(f"\n[{index}/{len(SYMBOLS)}] {normalized}")
        print(f"{normalized} generation range {GENERATION_START}..{GENERATION_END}")
        threshold_payload = load_or_create_threshold_plan(symbol)
        generate_drb_for_symbol(symbol, threshold_payload)


if __name__ == "__main__":
    main()
