#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Parallel downloader for Binance Vision futures aggTrades archives.
# --------------------------------------------------------------------------

from __future__ import annotations

# ruff: noqa: E402
import argparse
import calendar
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afml_strategies.config_loader import load_afml_data_config
from afml_strategies.config_loader import resolve_repo_path
from afml_strategies.config_loader import section
from afml_strategies.config_loader import string_tuple


BASE_URL = "https://data.binance.vision/data/futures"
READ_SIZE = 1024 * 1024
CONFIG = load_afml_data_config()
REAL_DATA_CONFIG = section(CONFIG, "real_data")
DOWNLOAD_CONFIG = section(REAL_DATA_CONFIG, "download")
INCREMENTAL_CONFIG = section(REAL_DATA_CONFIG, "incremental")


# --------------------------------------------------------------------------
# VSCode Run Button config
# --------------------------------------------------------------------------
# When this file is run without command-line arguments, values are read from
# afml_strategies/afml_data_config.json.
SYMBOLS = list(string_tuple(REAL_DATA_CONFIG.get("symbols"), default=("SOLUSDT.P",)))
START_DATE = str(
    REAL_DATA_CONFIG.get(
        "download_start",
        INCREMENTAL_CONFIG.get("download_start_fallback", REAL_DATA_CONFIG.get("threshold_history_start", "2025-01-01")),
    ),
)
END_DATE = str(
    REAL_DATA_CONFIG.get(
        "download_end",
        REAL_DATA_CONFIG.get("generation_end", "2026-04-30"),
    ),
)

# Perpetual futures:
# - "um" = USD-M perpetuals such as BTCUSDT, ETHUSDT, SOLUSDT
# - "cm" = COIN-M perpetuals
PRODUCT = str(REAL_DATA_CONFIG.get("product", "um"))

# "monthly" downloads one full monthly archive for every month touched by the
# date range. This is usually fastest and works with downstream date filtering.
# "daily" downloads one archive per day.
PERIOD_MODE = str(DOWNLOAD_CONFIG.get("period_mode", "monthly"))

OUTPUT_DIR = resolve_repo_path(REAL_DATA_CONFIG.get("input_dir", "binance_scripts/data/binance_vision"))
# Number of range requests used inside one archive download.
WORKERS = int(DOWNLOAD_CONFIG.get("workers", 32))
# Number of symbol/month archive downloads allowed to run at the same time.
JOB_WORKERS = int(DOWNLOAD_CONFIG.get("job_workers", 4))
CHUNK_MB = int(DOWNLOAD_CONFIG.get("chunk_mb", 8))
RETRIES = int(DOWNLOAD_CONFIG.get("retries", 8))
TIMEOUT = float(DOWNLOAD_CONFIG.get("timeout", 120.0))
INCREMENTAL_ENABLED = bool(INCREMENTAL_CONFIG.get("enabled", False))
INCREMENTAL_END_POLICY = str(INCREMENTAL_CONFIG.get("end_policy", "configured"))


@dataclass(frozen=True)
class DownloadJob:
    symbol: str
    product: str
    period: str
    is_monthly: bool
    output_dir: Path
    workers: int
    chunk_mb: int
    retries: int
    timeout: float
    expected_size: int | None = None


def normalize_symbol(value: str) -> str:
    return value.strip().upper().replace("/", "").replace("-", "").replace(".P", "")


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def parse_month(value: str) -> date:
    return date.fromisoformat(f"{value}-01")


def month_periods(start: date, end: date) -> list[str]:
    periods: list[str] = []
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        periods.append(f"{current:%Y-%m}")
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return periods


def day_periods(start: date, end: date) -> list[str]:
    periods: list[str] = []
    current = start
    while current <= end:
        periods.append(f"{current:%Y-%m-%d}")
        current = date.fromordinal(current.toordinal() + 1)
    return periods


def month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def last_complete_month_end(today: date | None = None) -> date:
    today = today or datetime.now(tz=UTC).date()
    first_day_this_month = date(today.year, today.month, 1)
    return first_day_this_month - timedelta(days=1)


def resolve_end_date(value: str, *, incremental: bool) -> date:
    if incremental and INCREMENTAL_END_POLICY == "last_complete_month":
        return last_complete_month_end()
    if value in {"auto", "auto_incremental", "last_complete_month"}:
        return last_complete_month_end()
    return parse_day(value)


def monthly_archive_url(product: str, symbol: str, month: str) -> str:
    return f"{BASE_URL}/{product}/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{month}.zip"


def daily_archive_url(product: str, symbol: str, day: str) -> str:
    return f"{BASE_URL}/{product}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"


def archive_name(symbol: str, period: str) -> str:
    return f"{symbol}-aggTrades-{period}.zip"


def archive_output_dir(base_dir: Path, product: str, symbol: str, is_monthly: bool) -> Path:
    if is_monthly:
        return base_dir / f"futures_{product}" / "monthly" / "aggTrades" / symbol
    return base_dir / f"futures_{product}" / "aggTrades" / symbol


def archive_period_end(period: str, *, is_monthly: bool) -> date:
    if is_monthly:
        return month_end(parse_month(period))
    return parse_day(period)


def archive_period_from_path(path: Path, symbol: str, *, is_monthly: bool) -> str | None:
    prefix = f"{symbol}-aggTrades-"
    suffix = ".zip"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    period = name[len(prefix) : -len(suffix)]
    try:
        if is_monthly:
            parse_month(period)
        else:
            parse_day(period)
    except ValueError:
        return None
    return period


def latest_local_archive_end(base_dir: Path, product: str, symbol_arg: str, *, is_monthly: bool) -> date | None:
    symbol = normalize_symbol(symbol_arg)
    output_dir = archive_output_dir(base_dir, product, symbol, is_monthly)
    if not output_dir.exists():
        return None

    latest: date | None = None
    for path in output_dir.glob(f"{symbol}-aggTrades-*.zip"):
        period = archive_period_from_path(path, symbol, is_monthly=is_monthly)
        if period is None:
            continue
        period_end = archive_period_end(period, is_monthly=is_monthly)
        if latest is None or period_end > latest:
            latest = period_end
    return latest


def content_length(url: str, timeout: float, retries: int) -> int:
    request = Request(  # noqa: S310
        url,
        method="HEAD",
        headers={"User-Agent": "nautilus-trader-binance-vision-range-downloader/1.0"},
    )
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                length = response.headers.get("Content-Length")
                if length is None:
                    raise RuntimeError("Missing Content-Length header")
                return int(length)
        except (HTTPError, TimeoutError, URLError, RuntimeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed HEAD request for {url}: {exc}") from exc
            time.sleep(min(2.0 * 2**attempt, 30.0))
    raise RuntimeError("unreachable")


def download_range(  # noqa: C901
    *,
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: float,
) -> int:
    expected_size = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return expected_size

    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "nautilus-trader-binance-vision-range-downloader/1.0",
    }
    request = Request(url, headers=headers)  # noqa: S310
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                status = response.getcode()
                if status != 206:
                    raise RuntimeError(f"Expected HTTP 206 for range request, got {status}")
                with tmp_path.open("wb") as file:
                    while True:
                        chunk = response.read(READ_SIZE)
                        if not chunk:
                            break
                        file.write(chunk)
            actual_size = tmp_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Range {start}-{end} wrote {actual_size} bytes, expected {expected_size}",
                )
            tmp_path.replace(part_path)
            return expected_size
        except (HTTPError, TimeoutError, URLError, RuntimeError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                raise RuntimeError(f"Failed range {start}-{end}: {exc}") from exc
            time.sleep(min(2.0 * 2**attempt, 30.0))

    raise RuntimeError("unreachable")


def combine_parts(parts: list[Path], output_path: Path) -> None:
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    with tmp_output.open("wb") as out:
        for part in parts:
            with part.open("rb") as file:
                while True:
                    chunk = file.read(READ_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
    tmp_output.replace(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel job and range downloader for Binance Vision futures aggTrades archives. "
            "Run without args to use the VSCode config at the top of this file."
        ),
    )
    parser.add_argument("--symbol", help="Single symbol such as BTCUSDT or BTCUSDT.P.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Batch symbols such as BTCUSDT.P ETHUSDT.P SOLUSDT.P.",
    )
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--month", help="Monthly archive as YYYY-MM.")
    period.add_argument("--date", help="Daily archive as YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Batch first UTC date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Batch last UTC date, inclusive, YYYY-MM-DD.")
    parser.add_argument("--period-mode", choices=["monthly", "daily"], default=PERIOD_MODE)
    parser.add_argument("--product", choices=["um", "cm"], default="um")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--job-workers", type=int, default=JOB_WORKERS)
    parser.add_argument("--chunk-mb", type=int, default=CHUNK_MB)
    parser.add_argument("--retries", type=int, default=RETRIES)
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--expected-size", type=int, default=None)
    return parser.parse_args(argv)


def download_archive(job: DownloadJob) -> Path:
    symbol = normalize_symbol(job.symbol)
    if job.is_monthly:
        parse_month(job.period)
        url = monthly_archive_url(job.product, symbol, job.period)
    else:
        parse_day(job.period)
        url = daily_archive_url(job.product, symbol, job.period)

    output_dir = archive_output_dir(job.output_dir, job.product, symbol, job.is_monthly)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / archive_name(symbol, job.period)

    total_size = job.expected_size or content_length(url, job.timeout, job.retries)
    if output_path.exists() and output_path.stat().st_size == total_size:
        print(f"already complete {output_path} ({total_size:,} bytes)")
        return output_path

    chunk_size = job.chunk_mb * 1024 * 1024
    part_count = math.ceil(total_size / chunk_size)
    parts_dir = output_dir / f".{output_path.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    ranges: list[tuple[int, int, int, Path]] = []
    parts: list[Path] = []
    for index in range(part_count):
        start = index * chunk_size
        end = min(start + chunk_size - 1, total_size - 1)
        part_path = parts_dir / f"part-{index:05d}"
        ranges.append((index, start, end, part_path))
        parts.append(part_path)

    print(
        f"downloading {url} size={total_size:,} parts={part_count} "
        f"workers={job.workers} chunk_mb={job.chunk_mb}",
        flush=True,
    )
    completed = 0
    bytes_done = 0

    with ThreadPoolExecutor(max_workers=job.workers) as executor:
        futures = [
            executor.submit(
                download_range,
                url=url,
                part_path=part,
                start=start,
                end=end,
                retries=job.retries,
                timeout=job.timeout,
            )
            for _, start, end, part in ranges
        ]
        for future in as_completed(futures):
            size = future.result()
            completed += 1
            bytes_done += size
            if completed == 1 or completed % 5 == 0 or completed == len(futures):
                print(
                    f"completed_parts={completed}/{len(futures)} "
                    f"bytes={min(bytes_done, total_size):,}/{total_size:,}",
                    flush=True,
                )

    combine_parts(parts, output_path)
    final_size = output_path.stat().st_size
    if final_size != total_size:
        raise RuntimeError(f"Combined file has {final_size} bytes, expected {total_size}")
    print(f"wrote {output_path} ({final_size:,} bytes)", flush=True)
    return output_path


def jobs_from_config() -> list[DownloadJob]:
    if PERIOD_MODE == "monthly":
        is_monthly = True
    elif PERIOD_MODE == "daily":
        is_monthly = False
    else:
        raise ValueError("PERIOD_MODE must be 'monthly' or 'daily'")

    end = resolve_end_date(END_DATE, incremental=INCREMENTAL_ENABLED)
    fallback_start = parse_day(START_DATE)
    jobs: list[DownloadJob] = []
    for symbol in SYMBOLS:
        latest_end = (
            latest_local_archive_end(OUTPUT_DIR, PRODUCT, symbol, is_monthly=is_monthly)
            if INCREMENTAL_ENABLED
            else None
        )
        start = latest_end + timedelta(days=1) if latest_end is not None else fallback_start
        if start > end:
            print(f"{normalize_symbol(symbol)} download incremental: already current through {end}")
            continue
        periods = month_periods(start, end) if is_monthly else day_periods(start, end)
        print(
            f"{normalize_symbol(symbol)} download range {start:%Y-%m-%d}..{end:%Y-%m-%d} "
            f"periods={len(periods)} mode={PERIOD_MODE}",
        )
        jobs.extend(
            DownloadJob(
                symbol=symbol,
                product=PRODUCT,
                period=period,
                is_monthly=is_monthly,
                output_dir=OUTPUT_DIR,
                workers=WORKERS,
                chunk_mb=CHUNK_MB,
                retries=RETRIES,
                timeout=TIMEOUT,
            )
            for period in periods
        )
    return jobs


def jobs_from_args(args: argparse.Namespace) -> list[DownloadJob]:
    if args.month is not None or args.date is not None:
        if args.symbol is None:
            raise ValueError("--symbol is required with --month or --date")
        return [
            DownloadJob(
                symbol=args.symbol,
                product=args.product,
                period=args.month or args.date,
                is_monthly=args.month is not None,
                output_dir=args.output_dir,
                workers=args.workers,
                chunk_mb=args.chunk_mb,
                retries=args.retries,
                timeout=args.timeout,
                expected_size=args.expected_size,
            ),
        ]

    symbols = args.symbols or ([args.symbol] if args.symbol else None)
    if not symbols:
        raise ValueError("Provide --symbol/--symbols, or run without args to use the top config")
    if args.start_date is None or args.end_date is None:
        raise ValueError("--start-date and --end-date are required for batch downloads")
    if args.expected_size is not None and len(symbols) > 1:
        raise ValueError("--expected-size is only supported for a single --symbol with --month/--date")

    start = parse_day(args.start_date)
    end = parse_day(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    if args.period_mode == "monthly":
        periods = month_periods(start, end)
        is_monthly = True
    else:
        periods = day_periods(start, end)
        is_monthly = False

    return [
        DownloadJob(
            symbol=symbol,
            product=args.product,
            period=period,
            is_monthly=is_monthly,
            output_dir=args.output_dir,
            workers=args.workers,
            chunk_mb=args.chunk_mb,
            retries=args.retries,
            timeout=args.timeout,
        )
        for symbol in symbols
        for period in periods
    ]


def describe_job(index: int, total: int, job: DownloadJob) -> str:
    kind = "monthly" if job.is_monthly else "daily"
    return f"[{index}/{total}] {normalize_symbol(job.symbol)} {job.product} {kind} {job.period}"


def run_jobs(jobs: list[DownloadJob], *, job_workers: int = JOB_WORKERS) -> None:
    if not jobs:
        print("No download jobs configured; local archives are already up to date.")
        return

    if job_workers < 1:
        raise ValueError("--job-workers must be positive")

    active_job_workers = min(job_workers, len(jobs))
    print(f"jobs={len(jobs)} job_workers={active_job_workers}")
    if active_job_workers == 1:
        for index, job in enumerate(jobs, start=1):
            print(describe_job(index, len(jobs), job), flush=True)
            download_archive(job)
        return

    with ThreadPoolExecutor(max_workers=active_job_workers) as executor:
        futures = {
            executor.submit(download_archive, job): (index, job)
            for index, job in enumerate(jobs, start=1)
        }
        for future in as_completed(futures):
            index, job = futures[future]
            try:
                output_path = future.result()
            except Exception as exc:
                raise RuntimeError(f"download job failed {describe_job(index, len(jobs), job)}") from exc
            print(f"finished {describe_job(index, len(jobs), job)} -> {output_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        jobs = jobs_from_config()
        run_jobs(jobs, job_workers=JOB_WORKERS)
    else:
        args = parse_args(argv)
        jobs = jobs_from_args(args)
        run_jobs(jobs, job_workers=args.job_workers)


if __name__ == "__main__":
    main()
