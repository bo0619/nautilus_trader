#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Generate AFML dollar imbalance bars (DIB) directly from Binance aggTrades archives.
#
# This file is intentionally a user-facing entrypoint:
# edit afml_strategies/afml_data_config.json, then run this file from VSCode.
# --------------------------------------------------------------------------

from __future__ import annotations

import calendar
import csv
import gc
import json
import math
import os
import shutil
import sys
import time
import zipfile
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from pathlib import Path


# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_strategies.config_loader import load_afml_data_config
from afml_strategies.config_loader import resolve_repo_path
from afml_strategies.config_loader import section
from afml_strategies.config_loader import string_tuple
from binance_scripts.afml_generation_utils import RollingDensityConfig
from binance_scripts.afml_generation_utils import RollingDensityController
from binance_scripts.afml_generation_utils import afml_expectations_to_dict
from binance_scripts.afml_generation_utils import assert_no_stale_output_tmp
from nautilus_trader.data.afml_bars import AfmlBarExpectations
from nautilus_trader.data.afml_bars import AfmlBarMeta
from nautilus_trader.data.afml_bars import AfmlDollarImbalanceBarAggregator
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
ROLLING_DENSITY_SECTION = section(REAL_DATA_CONFIG, "rolling_density")


def int_sequence_config(value: object, default: tuple[int, ...]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# VSCode Run Button config
# --------------------------------------------------------------------------
SYMBOLS = list(string_tuple(REAL_DATA_CONFIG.get("symbols"), default=("BTCUSDT.P",)))

GENERATION_START_CONFIG = str(REAL_DATA_CONFIG.get("generation_start", "2025-02-01"))
GENERATION_END_CONFIG = str(REAL_DATA_CONFIG.get("generation_end", "2026-04-30"))
GENERATION_START = GENERATION_START_CONFIG
GENERATION_END = GENERATION_END_CONFIG
DEFAULT_CALIBRATION_END = (date.fromisoformat(GENERATION_START_CONFIG) - timedelta(days=1)).isoformat()
CALIBRATION_START_CONFIG = str(REAL_DATA_CONFIG.get("calibration_start", "2025-01-01"))
CALIBRATION_END_CONFIG = str(REAL_DATA_CONFIG.get("calibration_end", DEFAULT_CALIBRATION_END))

PRODUCT = str(REAL_DATA_CONFIG.get("product", "um"))
TARGET_BARS_PER_DAY = int(REAL_DATA_CONFIG.get("target_bars_per_day", 1440))
EWMA_SPAN_CANDIDATES = int_sequence_config(
    REAL_DATA_CONFIG.get("ewma_span_candidates"),
    (10, 20, 50),
)
SCALE_SEARCH_ITERATIONS = int(REAL_DATA_CONFIG.get("scale_search_iterations", 6))
SCALE_SEARCH_DAMPING = float(REAL_DATA_CONFIG.get("scale_search_damping", 0.80))
IMBALANCE_FLOOR_FRAC = float(REAL_DATA_CONFIG.get("imbalance_floor_frac", 0.01))
ADAPTIVE_DENSITY = bool(REAL_DATA_CONFIG.get("adaptive_density", False))
DENSITY_ADJUSTMENT_STRENGTH = float(REAL_DATA_CONFIG.get("density_adjustment_strength", 0.5))
DENSITY_MIN_SCALE = float(REAL_DATA_CONFIG.get("density_min_scale", 0.25))
DENSITY_MAX_SCALE = float(REAL_DATA_CONFIG.get("density_max_scale", 4.0))
DENSITY_MIN_ELAPSED_FRACTION = float(REAL_DATA_CONFIG.get("density_min_elapsed_fraction", 1.0 / 24.0))
ROLLING_DENSITY_ENABLED = bool(ROLLING_DENSITY_SECTION.get("enabled", False))
ROLLING_DENSITY_CONFIG = RollingDensityConfig(
    target_bars_per_day=TARGET_BARS_PER_DAY,
    activity_half_life_days=float(
        ROLLING_DENSITY_SECTION.get("activity_half_life_days", 7.0),
    ),
    error_half_life_days=float(
        ROLLING_DENSITY_SECTION.get("error_half_life_days", 7.0),
    ),
    feedback_gain=float(ROLLING_DENSITY_SECTION.get("feedback_gain", 0.20)),
    max_daily_scale_change=float(
        ROLLING_DENSITY_SECTION.get("max_daily_scale_change", 0.10),
    ),
    min_scale=float(ROLLING_DENSITY_SECTION.get("min_scale", 0.25)),
    max_scale=float(ROLLING_DENSITY_SECTION.get("max_scale", 4.0)),
    activity_clip_min_ratio=float(
        ROLLING_DENSITY_SECTION.get("activity_clip_min_ratio", 0.25),
    ),
    activity_clip_max_ratio=float(
        ROLLING_DENSITY_SECTION.get("activity_clip_max_ratio", 4.0),
    ),
    density_error_clip_ratio=float(
        ROLLING_DENSITY_SECTION.get("density_error_clip_ratio", 4.0),
    ),
)
if ROLLING_DENSITY_ENABLED and ADAPTIVE_DENSITY:
    raise ValueError("rolling_density and adaptive_density cannot both be enabled")

# Keep E[T] fixed to the target density. The DIB side/notional expectations
# still update online after each closed bar.
UPDATE_EXPECTED_TICKS = bool(REAL_DATA_CONFIG.get("update_expected_ticks", False))
if ROLLING_DENSITY_ENABLED and UPDATE_EXPECTED_TICKS:
    raise ValueError("rolling_density requires update_expected_ticks=false")
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
TICK_CACHE_LOCK_TIMEOUT_SECONDS = int(REAL_DATA_CONFIG.get("tick_cache_lock_timeout_seconds", 3_600))
TICK_CACHE_STALE_LOCK_SECONDS = int(REAL_DATA_CONFIG.get("tick_cache_stale_lock_seconds", 21_600))


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
        BarSpecification(target_bars_per_day, BarAggregation.VALUE_IMBALANCE, PriceType.LAST),
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


def unlink_file_with_retries(path: Path, *, attempts: int = 30, delay_seconds: float = 1.0) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, attempts + 1):
        if not path.exists():
            return
        try:
            path.unlink()
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts:
                break
            gc.collect()
            time.sleep(delay_seconds)
    raise PermissionError(f"Could not remove locked file after {attempts} attempts: {path}") from last_error


def replace_file_with_retries(
    source_path: Path,
    target_path: Path,
    *,
    attempts: int = 30,
    delay_seconds: float = 1.0,
) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, attempts + 1):
        gc.collect()
        try:
            source_path.replace(target_path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts:
                break
            gc.collect()
            time.sleep(delay_seconds)
    raise PermissionError(
        f"Could not replace locked file after {attempts} attempts: {source_path} -> {target_path}",
    ) from last_error


class TickCacheFileLock:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.fd: int | None = None

    def __enter__(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + TICK_CACHE_LOCK_TIMEOUT_SECONDS
        printed_wait = False
        while True:
            try:
                self.fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                payload = f"pid={os.getpid()} created_at={datetime.now(tz=UTC).isoformat()}\n"
                os.write(self.fd, payload.encode("utf-8"))
                return
            except FileExistsError:
                if self._lock_is_stale():
                    print(f"removing stale tick cache lock -> {self.lock_path}", flush=True)
                    unlink_file_with_retries(self.lock_path)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for tick cache lock: {self.lock_path}")
                if not printed_wait:
                    print(f"waiting for tick cache lock -> {self.lock_path}", flush=True)
                    printed_wait = True
                time.sleep(1.0)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        unlink_file_with_retries(self.lock_path)

    def _lock_is_stale(self) -> bool:
        try:
            age_seconds = time.time() - self.lock_path.stat().st_mtime
        except OSError:
            return False
        return age_seconds > TICK_CACHE_STALE_LOCK_SECONDS


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
        print(f"ignoring stale tick cache tmp -> {tmp_path}", flush=True)
    if cache_path.exists():
        unlink_file_with_retries(cache_path)

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
            writer = pq.ParquetWriter(cache_path, schema, compression="snappy")
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
            writer = None
    gc.collect()

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
        pq.write_table(empty, cache_path, compression="snappy")
    print(f"wrote tick cache rows={rows_written:,} -> {cache_path}", flush=True)
    return rows_written


def ensure_tick_cache(source_path: Path, cache_dir: Path) -> Path:
    cache_path = tick_cache_path_for_archive(source_path, cache_dir)
    if tick_cache_is_current(source_path, cache_path):
        return cache_path
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with TickCacheFileLock(lock_path):
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


class StreamingDibCsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.count = 0
        self.daily_counts: dict[str, int] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            raise RuntimeError(f"Refusing to overwrite stale DIB temp output: {self.tmp_path}")
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
                "bar_type": "AFML_DIB",
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
        replace_file_with_retries(self.tmp_path, self.path)


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


def calibration_plan_path(symbol: str) -> Path:
    stem = (
        f"{symbol}_{CALIBRATION_START_CONFIG.replace('-', '')}_"
        f"{CALIBRATION_END_CONFIG.replace('-', '')}_{TARGET_BARS_PER_DAY}tpd_"
        "DIB_calibration_plan.json"
    )
    return CALIBRATION_OUTPUT_DIR / symbol / stem


def expected_calibration_plan_config(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "kind": "DIB",
        "threshold_method": "calibrated_ewma_threshold_scale",
        "calibration_start_date": CALIBRATION_START_CONFIG,
        "calibration_end_date": CALIBRATION_END_CONFIG,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "product": PRODUCT,
        "ewma_span_candidates": EWMA_SPAN_CANDIDATES,
        "scale_search_iterations": SCALE_SEARCH_ITERATIONS,
        "scale_search_damping": SCALE_SEARCH_DAMPING,
        "imbalance_floor_frac": IMBALANCE_FLOOR_FRAC,
        "adaptive_density": ADAPTIVE_DENSITY,
        "density_adjustment_strength": DENSITY_ADJUSTMENT_STRENGTH,
        "density_min_scale": DENSITY_MIN_SCALE,
        "density_max_scale": DENSITY_MAX_SCALE,
        "density_min_elapsed_fraction": DENSITY_MIN_ELAPSED_FRACTION,
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
        "write_catalog": WRITE_CATALOG,
    }


def calibration_plan_mismatch_reason(payload: dict, symbol: str) -> str | None:
    for key, expected_value in expected_calibration_plan_config(symbol).items():
        if payload.get(key) != expected_value:
            return f"{key} cached={payload.get(key)!r} current={expected_value!r}"
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        return "missing selected"
    if "ewma_span" not in selected or "threshold_scale" not in selected:
        return "selected missing ewma_span or threshold_scale"
    return None


def aggregator_density_kwargs() -> dict:
    if not ADAPTIVE_DENSITY:
        return {}
    return {
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "density_adjustment_strength": DENSITY_ADJUSTMENT_STRENGTH,
        "density_min_scale": DENSITY_MIN_SCALE,
        "density_max_scale": DENSITY_MAX_SCALE,
        "density_min_elapsed_fraction": DENSITY_MIN_ELAPSED_FRACTION,
    }


def generation_threshold_method() -> str:
    if ROLLING_DENSITY_ENABLED:
        return "rolling_daily_activity_density_v1"
    return "calibrated_ewma_threshold_scale"


def rolling_density_summary_config() -> dict:
    return {
        "enabled": ROLLING_DENSITY_ENABLED,
        **ROLLING_DENSITY_CONFIG.to_dict(),
        "monthly_state_reset": True,
    }


def has_valid_rolling_density_checkpoint(payload: dict) -> bool:
    if not ROLLING_DENSITY_ENABLED:
        return True
    state = payload.get("density_controller_state")
    expectations = payload.get("final_expectations")
    return (
        isinstance(state, dict)
        and state.get("version") == RollingDensityController.STATE_VERSION
        and {"tick_forecast", "threshold_scale", "log_density_error"} <= state.keys()
        and isinstance(expectations, dict)
        and {
            "expected_ticks",
            "p_buy",
            "buy_mean_notional",
            "sell_mean_notional",
            "mean_notional",
        }
        <= expectations.keys()
    )


def count_daily_dib_bars(
    *,
    ticks: list[TradeTick],
    instrument,
    target_bars: int,
    ewma_span: int,
    threshold_scale: float,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    expectations = estimate_afml_dollar_expectations(ticks, target_bars)
    bar_type = make_bar_type(instrument, TARGET_BARS_PER_DAY)

    def record(bar: Bar, _meta: AfmlBarMeta) -> None:
        key = day_key(int(bar.ts_event))
        counts[key] = counts.get(key, 0) + 1

    aggregator = AfmlDollarImbalanceBarAggregator(
        instrument=instrument,
        bar_type=bar_type,
        handler=None,
        meta_handler=record,
        expectations=expectations,
        ewma_span=ewma_span,
        threshold_scale=threshold_scale,
        imbalance_floor_frac=IMBALANCE_FLOOR_FRAC,
        update_expected_ticks=UPDATE_EXPECTED_TICKS,
        **aggregator_density_kwargs(),
    )
    for tick in ticks:
        aggregator.handle_trade_tick(tick)
    return counts


def daily_bar_loss(
    counts: dict[str, int],
    *,
    days: list[date],
) -> tuple[float, int, float, int, int]:
    values = [int(counts.get(day.isoformat(), 0)) for day in days]
    total = sum(values)
    mean_abs_error = sum(abs(value - TARGET_BARS_PER_DAY) for value in values) / len(values)
    avg_per_day = total / len(values)
    return mean_abs_error, total, avg_per_day, min(values), max(values)


def next_threshold_scale(scale: float, total_bars: int, target_bars: int) -> float:
    if total_bars <= 0 or target_bars <= 0:
        return scale * 0.5
    ratio = total_bars / target_bars
    if not math.isfinite(ratio) or ratio <= 0.0:
        return scale
    return max(1e-9, scale * (ratio**SCALE_SEARCH_DAMPING))


def calibrate_dib_parameters(
    *,
    ticks: list[TradeTick],
    instrument,
    days: list[date],
    target_bars: int,
) -> tuple[dict, list[dict]]:
    best: dict | None = None
    rows: list[dict] = []
    for ewma_span in EWMA_SPAN_CANDIDATES:
        scale = 1.0
        for iteration in range(1, SCALE_SEARCH_ITERATIONS + 1):
            counts = count_daily_dib_bars(
                ticks=ticks,
                instrument=instrument,
                target_bars=target_bars,
                ewma_span=ewma_span,
                threshold_scale=scale,
            )
            mean_abs_error, total, avg_per_day, min_daily, max_daily = daily_bar_loss(
                counts,
                days=days,
            )
            row = {
                "ewma_span": ewma_span,
                "threshold_scale": scale,
                "iteration": iteration,
                "bars": total,
                "bars_per_day_avg": avg_per_day,
                "min_daily_bars": min_daily,
                "max_daily_bars": max_daily,
                "mean_abs_daily_error": mean_abs_error,
                "daily_counts": counts,
            }
            rows.append(row)
            if best is None or (mean_abs_error, abs(total - target_bars)) < (
                best["mean_abs_daily_error"],
                abs(best["bars"] - target_bars),
            ):
                best = row
            scale = next_threshold_scale(scale, total, target_bars)
            gc.collect()
    if best is None:
        raise RuntimeError("DIB calibration produced no candidates")
    selected = dict(best)
    selected["threshold_scale"] = float(selected["threshold_scale"])
    return selected, rows


def load_or_create_calibration_plan(symbol_arg: str) -> dict:
    started = time.perf_counter()
    symbol = normalize_symbol(symbol_arg)
    output_path = calibration_plan_path(symbol)
    if RESUME_CALIBRATION and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        mismatch_reason = calibration_plan_mismatch_reason(payload, symbol)
        if mismatch_reason is None:
            print(f"{symbol} resume: using DIB calibration plan -> {output_path}")
            return payload
        print(f"{symbol} resume: ignoring stale DIB calibration plan ({mismatch_reason}) -> {output_path}")

    calibration_start = parse_date(CALIBRATION_START_CONFIG)
    calibration_end = parse_date(CALIBRATION_END_CONFIG)
    generation_start = parse_date(GENERATION_START_CONFIG)
    if calibration_end < calibration_start:
        raise SystemExit("CALIBRATION_END must be on or after CALIBRATION_START")
    if calibration_end >= generation_start:
        raise SystemExit("CALIBRATION_END must be before GENERATION_START to avoid overlap")

    calibration_paths = archive_paths_for_range(
        input_dir=INPUT_DIR,
        product=PRODUCT,
        symbol=symbol,
        start=calibration_start,
        end=calibration_end,
    )
    instrument = make_binance_perpetual(symbol, calibration_paths)
    calibration_ticks = load_trade_ticks(
        calibration_paths,
        instrument,
        start_ms=date_to_ms(calibration_start),
        end_ms=next_date_ms(calibration_end),
        use_cache=USE_TICK_CACHE,
        cache_dir=TICK_CACHE_DIR,
    )
    calibration_days = list(iter_days(calibration_start, calibration_end))
    calibration_target_bars = TARGET_BARS_PER_DAY * len(calibration_days)
    selected, candidates = calibrate_dib_parameters(
        ticks=calibration_ticks,
        instrument=instrument,
        days=calibration_days,
        target_bars=calibration_target_bars,
    )
    payload = {
        **expected_calibration_plan_config(symbol),
        "calibration_days": len(calibration_days),
        "calibration_target_bars": calibration_target_bars,
        "calibration_ticks": len(calibration_ticks),
        "selected": selected,
        "candidates": candidates,
        "use_tick_cache": USE_TICK_CACHE,
        "tick_cache_dir": str(TICK_CACHE_DIR),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{symbol} DIB calibration selected ewma_span={selected['ewma_span']} "
        f"scale={selected['threshold_scale']:.8f} bars={selected['bars']:,} "
        f"avg/day={selected['bars_per_day_avg']:.2f}",
    )
    print(f"{symbol} wrote DIB calibration plan -> {output_path}")
    return payload


def output_label(symbol: str, start: date, end: date) -> str:
    return f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_{TARGET_BARS_PER_DAY}tpd"


def output_label_with_kind(symbol: str, start: date, end: date) -> str:
    return f"{output_label(symbol, start, end)}_DIB"


def cumulative_catalog_path(symbol: str, end: date) -> Path:
    start = parse_date(GENERATION_START_FALLBACK if INCREMENTAL_ENABLED else GENERATION_START_CONFIG)
    return CATALOG_DIR / symbol / output_label_with_kind(symbol, start, end)


def summary_matches_generation_config(
    payload: dict,
    symbol: str,
    calibration_payload: dict | None = None,
) -> bool:
    expected = {
        "symbol": symbol,
        "kind": "DIB",
        "threshold_method": generation_threshold_method(),
        "calibration_start_date": CALIBRATION_START_CONFIG,
        "calibration_end_date": CALIBRATION_END_CONFIG,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "ewma_span_candidates": EWMA_SPAN_CANDIDATES,
        "scale_search_iterations": SCALE_SEARCH_ITERATIONS,
        "scale_search_damping": SCALE_SEARCH_DAMPING,
        "imbalance_floor_frac": IMBALANCE_FLOOR_FRAC,
        "adaptive_density": ADAPTIVE_DENSITY,
        "rolling_density": rolling_density_summary_config(),
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
        "write_catalog": WRITE_CATALOG,
    }
    if not all(payload.get(key) == value for key, value in expected.items()):
        return False
    if not has_valid_rolling_density_checkpoint(payload):
        return False
    if calibration_payload is None:
        return True
    selected = calibration_payload["selected"]
    if payload.get("ewma_span") != int(selected["ewma_span"]):
        return False
    return abs(
        float(payload.get("threshold_scale", 0.0)) - float(selected["threshold_scale"]),
    ) <= 1e-12


def matching_generated_summaries(
    symbol_arg: str,
    calibration_payload: dict | None = None,
) -> list[dict]:
    symbol = normalize_symbol(symbol_arg)
    symbol_dir = OUTPUT_DIR / symbol
    if not symbol_dir.exists():
        return []

    summaries: list[dict] = []
    for summary_path in symbol_dir.glob("*_DIB_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not summary_matches_generation_config(payload, symbol, calibration_payload):
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


def latest_generated_summary_end(
    symbol_arg: str,
    calibration_payload: dict | None = None,
) -> date | None:
    latest: date | None = None
    for summary in matching_generated_summaries(symbol_arg, calibration_payload):
        end = summary["end"]
        if latest is None or end > latest:
            latest = end
    return latest


def generation_range_for_symbol(
    symbol_arg: str,
    calibration_payload: dict | None = None,
) -> tuple[date, date]:
    end = resolve_generation_end(GENERATION_END_CONFIG, incremental=INCREMENTAL_ENABLED)
    if INCREMENTAL_ENABLED:
        latest_end = latest_generated_summary_end(symbol_arg, calibration_payload)
        start = latest_end + timedelta(days=1) if latest_end is not None else parse_date(GENERATION_START_FALLBACK)
    else:
        start = parse_date(GENERATION_START_CONFIG)
    return start, end


def previous_generation_summary(
    symbol_arg: str,
    *,
    start: date,
    calibration_payload: dict,
) -> dict | None:
    previous_end = start - timedelta(days=1)
    matches = [
        summary
        for summary in matching_generated_summaries(symbol_arg, calibration_payload)
        if summary["end"] == previous_end
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item["start"])


def base_catalog_summary(
    symbol_arg: str,
    calibration_payload: dict | None = None,
) -> dict | None:
    summaries = [
        summary
        for summary in matching_generated_summaries(symbol_arg, calibration_payload)
        if summary["catalog_path"] is not None and summary["catalog_path"].exists()
    ]
    if not summaries:
        return None

    fallback_start = parse_date(GENERATION_START_FALLBACK)
    for summary in summaries:
        if summary["start"] == fallback_start:
            return summary
    return summaries[0]


def catalog_path_for_generation(
    symbol: str,
    label: str,
    calibration_payload: dict | None = None,
) -> tuple[Path, bool]:
    default_path = CATALOG_DIR / symbol / label
    if not INCREMENTAL_ENABLED:
        return default_path, False

    summary = base_catalog_summary(symbol, calibration_payload)
    if summary is None:
        return default_path, False
    return summary["catalog_path"], True


def update_matching_summary_catalog_paths(
    symbol_arg: str,
    catalog_path: Path,
    calibration_payload: dict | None = None,
) -> None:
    catalog_path_text = str(catalog_path)
    for summary in matching_generated_summaries(symbol_arg, calibration_payload):
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


def finalize_incremental_catalog_path(
    symbol: str,
    catalog_path: Path,
    end: date,
    calibration_payload: dict | None = None,
) -> Path:
    if not INCREMENTAL_ENABLED or not WRITE_CATALOG:
        return catalog_path

    final_path = move_catalog_path(catalog_path, cumulative_catalog_path(symbol, end))
    update_matching_summary_catalog_paths(symbol, final_path, calibration_payload)
    return final_path


def repair_split_incremental_catalogs(
    symbol_arg: str,
    calibration_payload: dict | None = None,
) -> None:
    if not INCREMENTAL_ENABLED or not WRITE_CATALOG:
        return

    base_summary = base_catalog_summary(symbol_arg, calibration_payload)
    if base_summary is None:
        return
    base_path = base_summary["catalog_path"]
    copied = 0
    for summary in matching_generated_summaries(symbol_arg, calibration_payload):
        catalog_path = summary["catalog_path"]
        if catalog_path is None or not catalog_path.exists() or paths_equal(catalog_path, base_path):
            continue
        copied += copy_catalog_files(catalog_path, base_path)
    if copied:
        print(f"{normalize_symbol(symbol_arg)} repaired split catalog files={copied} -> {base_path}")
    latest_end = latest_generated_summary_end(symbol_arg, calibration_payload)
    if latest_end is not None:
        final_path = finalize_incremental_catalog_path(
            normalize_symbol(symbol_arg),
            base_path,
            latest_end,
            calibration_payload,
        )
        update_matching_summary_catalog_paths(symbol_arg, final_path, calibration_payload)


def completed_csv_summary_matches(  # noqa: C901
    *,
    summary_path: Path,
    output_path: Path,
    tmp_path: Path,
    symbol: str,
    start: date,
    end: date,
    calibration_payload: dict,
) -> dict | None:
    if not summary_path.exists() or not output_path.exists() or tmp_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "symbol": symbol,
        "kind": "DIB",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "threshold_method": generation_threshold_method(),
        "calibration_start_date": CALIBRATION_START_CONFIG,
        "calibration_end_date": CALIBRATION_END_CONFIG,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "ewma_span_candidates": EWMA_SPAN_CANDIDATES,
        "scale_search_iterations": SCALE_SEARCH_ITERATIONS,
        "scale_search_damping": SCALE_SEARCH_DAMPING,
        "imbalance_floor_frac": IMBALANCE_FLOOR_FRAC,
        "adaptive_density": ADAPTIVE_DENSITY,
        "rolling_density": rolling_density_summary_config(),
        "ewma_span": int(calibration_payload["selected"]["ewma_span"]),
        "update_expected_ticks": UPDATE_EXPECTED_TICKS,
    }
    for key, expected_value in expected.items():
        actual_value = summary.get(key)
        if isinstance(expected_value, float):
            if actual_value is None or abs(float(actual_value) - expected_value) > 1e-12:
                return None
        elif actual_value != expected_value:
            return None
    if not has_valid_rolling_density_checkpoint(summary):
        return None
    if abs(
        float(summary.get("threshold_scale", 0.0))
        - float(calibration_payload["selected"]["threshold_scale"])
    ) > 1e-12:
        return None
    if int(summary.get("bars", -1)) <= 0:
        return None
    if WRITE_CATALOG:
        catalog_path = summary.get("catalog_path")
        if catalog_path is None or not Path(catalog_path).exists():
            return None
    return summary


def generate_dib_for_symbol(  # noqa: C901
    symbol_arg: str,
    calibration_payload: dict,
) -> None:
    started = time.perf_counter()
    symbol = normalize_symbol(symbol_arg)
    start = parse_date(GENERATION_START)
    end = parse_date(GENERATION_END)
    if end < start:
        raise SystemExit("GENERATION_END must be on or after GENERATION_START")

    calibration_start = parse_date(CALIBRATION_START_CONFIG)
    calibration_end = parse_date(CALIBRATION_END_CONFIG)
    if calibration_end >= start:
        raise SystemExit("CALIBRATION_END must be before generation start to avoid overlap")
    selected = calibration_payload["selected"]
    ewma_span = int(selected["ewma_span"])
    threshold_scale = float(selected["threshold_scale"])
    days = (end - start).days + 1
    target_bars_total = TARGET_BARS_PER_DAY * days

    label = output_label(symbol, start, end)
    catalog_label = output_label_with_kind(symbol, start, end)
    catalog_path, append_existing_catalog = catalog_path_for_generation(
        symbol,
        catalog_label,
        calibration_payload,
    )
    output_dir = OUTPUT_DIR / symbol
    output_path = output_dir / f"{label}_DIB.csv"
    summary_path = output_dir / f"{label}_DIB_summary.json"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    assert_no_stale_output_tmp(
        tmp_path=tmp_path,
        output_path=output_path,
        summary_path=summary_path,
        kind="DIB",
    )

    if RESUME_COMPLETED_CSV:
        existing_summary = completed_csv_summary_matches(
            summary_path=summary_path,
            output_path=output_path,
            tmp_path=tmp_path,
            symbol=symbol,
            start=start,
            end=end,
            calibration_payload=calibration_payload,
        )
        if existing_summary is not None:
            print(
                f"{symbol} resume: found completed DIB CSV "
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

    calibration_paths = archive_paths_for_range(
        input_dir=INPUT_DIR,
        product=PRODUCT,
        symbol=symbol,
        start=calibration_start,
        end=calibration_end,
    )
    calibration_ticks = load_trade_ticks(
        calibration_paths,
        instrument,
        start_ms=date_to_ms(calibration_start),
        end_ms=next_date_ms(calibration_end),
        use_cache=USE_TICK_CACHE,
        cache_dir=TICK_CACHE_DIR,
    )
    calibration_days = (calibration_end - calibration_start).days + 1
    calibration_target = TARGET_BARS_PER_DAY * calibration_days
    expectations = estimate_afml_dollar_expectations(calibration_ticks, calibration_target)
    previous_summary = previous_generation_summary(
        symbol,
        start=start,
        calibration_payload=calibration_payload,
    )
    density_controller: RollingDensityController | None = None
    density_state_source: str | None = None
    if ROLLING_DENSITY_ENABLED:
        if previous_summary is None:
            density_controller = RollingDensityController(
                config=ROLLING_DENSITY_CONFIG,
                initial_tick_forecast=expectations.expected_ticks * TARGET_BARS_PER_DAY,
                initial_threshold_scale=threshold_scale,
            )
        else:
            previous_payload = previous_summary["payload"]
            expectations = AfmlBarExpectations(**previous_payload["final_expectations"])
            density_controller = RollingDensityController.from_state(
                config=ROLLING_DENSITY_CONFIG,
                state=previous_payload["density_controller_state"],
            )
            density_state_source = str(previous_summary["summary_path"])
        expectations.expected_ticks = density_controller.expected_ticks

    print(
        f"{symbol} DIB generation={start:%Y-%m-%d}..{end:%Y-%m-%d} "
        f"threshold={generation_threshold_method()} "
        f"calibration={calibration_start:%Y-%m-%d}..{calibration_end:%Y-%m-%d} "
        f"ewma_span={ewma_span} scale={threshold_scale:.8f}",
    )

    writer = StreamingDibCsvWriter(output_path)
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

    def create_aggregator(
        expectation_state: AfmlBarExpectations,
    ) -> AfmlDollarImbalanceBarAggregator:
        effective_scale = (
            density_controller.threshold_scale
            if density_controller is not None
            else threshold_scale
        )
        return AfmlDollarImbalanceBarAggregator(
            instrument=instrument,
            bar_type=bar_type,
            handler=catalog_sink.append,
            meta_handler=writer.handle,
            expectations=expectation_state,
            ewma_span=ewma_span,
            threshold_scale=effective_scale,
            imbalance_floor_frac=IMBALANCE_FLOOR_FRAC,
            update_expected_ticks=UPDATE_EXPECTED_TICKS,
            **aggregator_density_kwargs(),
        )

    aggregator = create_aggregator(expectations)
    active_month: str | None = None
    monthly_state_resets: list[dict] = []

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
            if density_controller is not None:
                settings_changed = density_controller.on_tick(
                    int(tick.ts_event),
                    total_bars=writer.count,
                )
                if settings_changed:
                    current_day = density_controller.current_day_iso
                    assert current_day is not None
                    current_month = current_day[:7]
                    if active_month is None:
                        active_month = current_month
                    elif current_month != active_month:
                        previous_month = active_month
                        dropped_partial_ticks = aggregator.ticks
                        expectations = AfmlBarExpectations(
                            **afml_expectations_to_dict(aggregator.expectations),
                        )
                        aggregator = create_aggregator(expectations)
                        monthly_state_resets.append(
                            {
                                "effective_month": current_month,
                                "previous_month": previous_month,
                                "dropped_partial_ticks": dropped_partial_ticks,
                            },
                        )
                        active_month = current_month
                    aggregator.expectations.expected_ticks = density_controller.expected_ticks
                    aggregator.threshold_scale = density_controller.threshold_scale
            aggregator.handle_trade_tick(tick)
            ticks_processed += 1
            if ticks_processed % 5_000_000 == 0:
                print(f"processed ticks={ticks_processed:,} DIB={writer.count:,}")
        if density_controller is not None:
            density_controller.finish(total_bars=writer.count)
    finally:
        catalog_sink.close()
        writer.close()

    writer.commit()
    final_catalog_path = catalog_sink.catalog_path
    if final_catalog_path is not None:
        final_catalog_path = finalize_incremental_catalog_path(
            symbol,
            final_catalog_path,
            end,
            calibration_payload,
        )
        catalog_sink.catalog_path = final_catalog_path

    summary = {
        "symbol": symbol,
        "instrument_id": str(instrument.id),
        "kind": "DIB",
        "source": "Binance Vision futures aggregate trades",
        "input_archives": [str(path) for path in generation_paths],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": days,
        "ticks": ticks_processed,
        "target_bars_per_day": TARGET_BARS_PER_DAY,
        "target_bars_total": target_bars_total,
        "bar_type": str(bar_type),
        "threshold_method": generation_threshold_method(),
        "calibration_start_date": calibration_start.isoformat(),
        "calibration_end_date": calibration_end.isoformat(),
        "calibration_days": calibration_days,
        "calibration_target_bars": calibration_target,
        "calibration_ticks": len(calibration_ticks),
        "ewma_span_candidates": EWMA_SPAN_CANDIDATES,
        "scale_search_iterations": SCALE_SEARCH_ITERATIONS,
        "scale_search_damping": SCALE_SEARCH_DAMPING,
        "imbalance_floor_frac": IMBALANCE_FLOOR_FRAC,
        "ewma_span": ewma_span,
        "threshold_scale": threshold_scale,
        "calibration_selected": selected,
        "adaptive_density": ADAPTIVE_DENSITY,
        "rolling_density": rolling_density_summary_config(),
        "density_adjustment_strength": DENSITY_ADJUSTMENT_STRENGTH,
        "density_min_scale": DENSITY_MIN_SCALE,
        "density_max_scale": DENSITY_MAX_SCALE,
        "density_min_elapsed_fraction": DENSITY_MIN_ELAPSED_FRACTION,
        "initial_expectation_start_date": calibration_start.isoformat(),
        "initial_expectation_end_date": calibration_end.isoformat(),
        "initial_expectation_days": calibration_days,
        "initial_expectation_target_bars": calibration_target,
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
        "density_state_source": density_state_source,
        "density_daily_schedule": (
            density_controller.daily_records if density_controller is not None else []
        ),
        "density_controller_state": (
            density_controller.state_dict() if density_controller is not None else None
        ),
        "final_expectations": {
            **afml_expectations_to_dict(aggregator.expectations),
            "expected_ticks": (
                density_controller.expected_ticks
                if density_controller is not None
                else aggregator.expectations.expected_ticks
            ),
        },
        "monthly_state_resets": monthly_state_resets,
        "end_partial_ticks_dropped": aggregator.ticks,
        "path": str(output_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote DIB bars={writer.count:,} avg/day={writer.count / days:.2f} -> {output_path}")
    if final_catalog_path is not None:
        print(f"wrote Nautilus catalog -> {final_catalog_path}")
    print(f"wrote summary -> {summary_path}")


def main() -> None:
    target_end = resolve_generation_end(GENERATION_END_CONFIG, incremental=INCREMENTAL_ENABLED)
    print(
        "AFML DIB from Binance aggTrades "
        f"calibration={CALIBRATION_START_CONFIG}..{CALIBRATION_END_CONFIG} "
        f"generation_end={target_end:%Y-%m-%d} "
        f"threshold={generation_threshold_method()} target={TARGET_BARS_PER_DAY}/day "
        f"ewma_candidates={EWMA_SPAN_CANDIDATES} "
        f"product={PRODUCT} "
        f"tick_cache={'on' if USE_TICK_CACHE else 'off'} "
        f"catalog={'on' if WRITE_CATALOG else 'off'}",
    )
    for index, symbol in enumerate(SYMBOLS, start=1):
        normalized = normalize_symbol(symbol)
        calibration_payload = load_or_create_calibration_plan(symbol)
        repair_split_incremental_catalogs(symbol, calibration_payload)
        start, end = generation_range_for_symbol(symbol, calibration_payload)
        if start > end:
            print(f"\n[{index}/{len(SYMBOLS)}] {normalized} already current through {end:%Y-%m-%d}")
            continue
        set_generation_range(start, end)
        print(f"\n[{index}/{len(SYMBOLS)}] {normalized}")
        print(f"{normalized} generation range {GENERATION_START}..{GENERATION_END}")
        generate_dib_for_symbol(symbol, calibration_payload)


if __name__ == "__main__":
    main()
