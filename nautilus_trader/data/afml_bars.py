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
AFML-style information-driven dollar bar aggregators.

This module implements the dynamic-threshold dollar imbalance and dollar runs
bars described in chapter 2 of *Advances in Financial Machine Learning*. It
uses NautilusTrader primitives (`TradeTick`, `BarBuilder`, `BarType`) while
keeping the threshold recursion in Python so it can be iterated on for research.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from nautilus_trader.data.aggregation import BarBuilder
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide


DAY_NANOS = 86_400 * 1_000_000_000


@dataclass(slots=True)
class AfmlBarExpectations:
    """
    Dynamic expectation state used by AFML dollar bars.

    Parameters
    ----------
    expected_ticks : float
        Expected number of ticks in a bar, E[T].
    p_buy : float
        Expected probability that a tick is buyer-initiated.
    buy_mean_notional : float
        Expected notional value conditional on buyer-initiated ticks.
    sell_mean_notional : float
        Expected notional value conditional on seller-initiated ticks.
    mean_notional : float
        Unconditional expected tick notional value.
    """

    expected_ticks: float
    p_buy: float
    buy_mean_notional: float
    sell_mean_notional: float
    mean_notional: float


@dataclass(slots=True)
class AfmlBarMeta:
    """
    Diagnostic metadata captured when an AFML bar closes.
    """

    ticks: int
    buy_ticks: int
    sell_ticks: int
    buy_notional: float
    sell_notional: float
    signed_notional: float
    theta: float
    threshold: float
    expected_ticks: float
    expected_metric_a: float
    expected_metric_b: float
    threshold_scale: float = 1.0


def _ewma(previous: float, value: float, alpha: float) -> float:
    if not math.isfinite(value):
        return previous
    return alpha * value + (1.0 - alpha) * previous


def estimate_afml_dollar_expectations(
    ticks: list[TradeTick],
    target_bars: int,
    min_sample_size: int = 1_000,
    sample_multiplier: float = 50.0,
) -> AfmlBarExpectations:
    """
    Estimate initial AFML dollar bar expectations from a tick sample.

    The expected tick count is initialized from the desired sample frequency.
    The side probabilities and notional expectations are initialized from the
    first `max(min_sample_size, expected_ticks * sample_multiplier)` ticks.
    """
    if not ticks:
        raise ValueError("Cannot estimate AFML expectations from an empty tick list")

    expected_ticks = max(1.0, len(ticks) / max(target_bars, 1))
    sample_size = min(len(ticks), max(min_sample_size, int(expected_ticks * sample_multiplier)))
    sample = ticks[:sample_size]

    notionals: list[float] = []
    buy_notionals: list[float] = []
    sell_notionals: list[float] = []
    for tick in sample:
        notional = float(tick.price) * float(tick.size)
        notionals.append(notional)
        if tick.aggressor_side == AggressorSide.BUYER:
            buy_notionals.append(notional)
        elif tick.aggressor_side == AggressorSide.SELLER:
            sell_notionals.append(notional)

    mean_notional = sum(notionals) / len(notionals)
    p_buy = len(buy_notionals) / len(sample)
    p_buy = min(max(p_buy, 1e-6), 1.0 - 1e-6)

    return AfmlBarExpectations(
        expected_ticks=expected_ticks,
        p_buy=p_buy,
        buy_mean_notional=(
            sum(buy_notionals) / len(buy_notionals) if buy_notionals else mean_notional
        ),
        sell_mean_notional=(
            sum(sell_notionals) / len(sell_notionals) if sell_notionals else mean_notional
        ),
        mean_notional=mean_notional,
    )


class _AfmlDollarBarAggregator:
    def __init__(
        self,
        *,
        instrument,
        bar_type: BarType,
        handler: Callable[[Bar], None] | None,
        meta_handler: Callable[[Bar, AfmlBarMeta], None] | None,
        expectations: AfmlBarExpectations,
        ewma_span: int,
        threshold_scale: float,
        update_expected_ticks: bool,
        threshold_provider: Callable[[int], float | None] | None = None,
        target_bars_per_day: int | None = None,
        density_adjustment_strength: float = 0.5,
        density_min_scale: float = 0.25,
        density_max_scale: float = 4.0,
        density_min_elapsed_fraction: float = 1.0 / 24.0,
    ) -> None:
        if instrument.id != bar_type.instrument_id:
            raise ValueError(
                f"instrument.id ({instrument.id}) != bar_type.instrument_id ({bar_type.instrument_id})",
            )
        if ewma_span < 1:
            raise ValueError("ewma_span must be greater than zero")
        if threshold_scale <= 0.0:
            raise ValueError("threshold_scale must be positive")
        if target_bars_per_day is not None and target_bars_per_day < 1:
            raise ValueError("target_bars_per_day must be positive")
        if density_adjustment_strength < 0.0:
            raise ValueError("density_adjustment_strength cannot be negative")
        if density_min_scale <= 0.0:
            raise ValueError("density_min_scale must be positive")
        if density_max_scale < density_min_scale:
            raise ValueError("density_max_scale must be greater than or equal to density_min_scale")
        if not 0.0 < density_min_elapsed_fraction <= 1.0:
            raise ValueError("density_min_elapsed_fraction must be in the interval (0, 1]")

        self.builder = BarBuilder(instrument, bar_type)
        self.bar_type = bar_type
        self.handler = handler
        self.meta_handler = meta_handler
        self.expectations = AfmlBarExpectations(
            expected_ticks=expectations.expected_ticks,
            p_buy=expectations.p_buy,
            buy_mean_notional=expectations.buy_mean_notional,
            sell_mean_notional=expectations.sell_mean_notional,
            mean_notional=expectations.mean_notional,
        )
        self.alpha = 2.0 / (ewma_span + 1.0)
        self.threshold_scale = threshold_scale
        self.update_expected_ticks = update_expected_ticks
        self.threshold_provider = threshold_provider
        self.target_bars_per_day = target_bars_per_day
        self.density_adjustment_strength = density_adjustment_strength
        self.density_min_scale = density_min_scale
        self.density_max_scale = density_max_scale
        self.density_min_elapsed_fraction = density_min_elapsed_fraction
        self.bar_count = 0
        self.last_meta: AfmlBarMeta | None = None
        self._density_day_start_ns: int | None = None
        self._density_day_bar_count = 0
        self._reset_bar_state()

    def _reset_bar_state(self) -> None:
        self.ticks = 0
        self.buy_ticks = 0
        self.sell_ticks = 0
        self.notional = 0.0
        self.buy_notional = 0.0
        self.sell_notional = 0.0
        self.signed_notional = 0.0

    def _handle_valid_trade_tick(self, tick: TradeTick) -> None:
        side_sign = 1.0 if tick.aggressor_side == AggressorSide.BUYER else -1.0
        notional = float(tick.price) * float(tick.size)

        self.builder.update(tick.price, tick.size, tick.ts_init)
        self.ticks += 1
        self.notional += notional
        self.signed_notional += side_sign * notional
        if side_sign > 0.0:
            self.buy_ticks += 1
            self.buy_notional += notional
        else:
            self.sell_ticks += 1
            self.sell_notional += notional

    def _build_now_and_send(self, meta: AfmlBarMeta) -> None:
        bar = self.builder.build_now()
        self.bar_count += 1
        if self.target_bars_per_day is not None:
            self._density_day_bar_count += 1
        self.last_meta = meta
        if self.meta_handler is not None:
            self.meta_handler(bar, meta)
        if self.handler is not None:
            self.handler(bar)
        self._update_expectations()
        self._reset_bar_state()

    def _update_expectations(self) -> None:
        if self.ticks == 0:
            return

        if self.update_expected_ticks:
            self.expectations.expected_ticks = _ewma(
                self.expectations.expected_ticks,
                float(self.ticks),
                self.alpha,
            )

        self.expectations.p_buy = _ewma(
            self.expectations.p_buy,
            self.buy_ticks / self.ticks,
            self.alpha,
        )
        if self.buy_ticks:
            self.expectations.buy_mean_notional = _ewma(
                self.expectations.buy_mean_notional,
                self.buy_notional / self.buy_ticks,
                self.alpha,
            )
        if self.sell_ticks:
            self.expectations.sell_mean_notional = _ewma(
                self.expectations.sell_mean_notional,
                self.sell_notional / self.sell_ticks,
                self.alpha,
            )
        self.expectations.mean_notional = _ewma(
            self.expectations.mean_notional,
            self.notional / self.ticks,
            self.alpha,
        )

    def _update_density_day(self, ts_ns: int) -> None:
        day_start_ns = ts_ns - (ts_ns % DAY_NANOS)
        if self._density_day_start_ns != day_start_ns:
            self._density_day_start_ns = day_start_ns
            self._density_day_bar_count = 0

    def _effective_threshold_scale(self, ts_ns: int) -> float:
        if self.target_bars_per_day is None:
            return self.threshold_scale

        self._update_density_day(ts_ns)
        assert self._density_day_start_ns is not None

        elapsed_ns = max(0, ts_ns - self._density_day_start_ns)
        elapsed_fraction = min(
            1.0,
            max(elapsed_ns / DAY_NANOS, self.density_min_elapsed_fraction),
        )
        target_so_far = max(1.0, self.target_bars_per_day * elapsed_fraction)
        observed_so_far = self._density_day_bar_count + 1.0
        density_ratio = max(observed_so_far / target_so_far, 1e-12)
        density_multiplier = density_ratio**self.density_adjustment_strength
        density_multiplier = min(
            self.density_max_scale,
            max(self.density_min_scale, density_multiplier),
        )
        return self.threshold_scale * density_multiplier

    def _resolve_threshold(self, ts_ns: int, base_threshold: float) -> tuple[float, float]:
        threshold_scale = self._effective_threshold_scale(ts_ns)
        threshold = threshold_scale * base_threshold
        if self.threshold_provider is None:
            return threshold, threshold_scale

        scheduled_threshold = self.threshold_provider(ts_ns)
        if scheduled_threshold is None:
            return threshold, threshold_scale
        scheduled_threshold = float(scheduled_threshold)
        if not math.isfinite(scheduled_threshold) or scheduled_threshold <= 0.0:
            raise ValueError("threshold_provider must return a positive finite threshold")
        equivalent_scale = scheduled_threshold / max(base_threshold, 1e-12)
        return scheduled_threshold, equivalent_scale


class AfmlDollarImbalanceBarAggregator(_AfmlDollarBarAggregator):
    """
    Builds AFML dynamic-threshold dollar imbalance bars from `TradeTick` data.

    The bar closes when:

    `abs(theta_T) >= E[T] * abs(P_buy * E[v|buy] - P_sell * E[v|sell])`

    where `theta_T` is the cumulative signed notional value in the current bar.
    """

    def __init__(
        self,
        *,
        instrument,
        bar_type: BarType,
        handler: Callable[[Bar], None] | None,
        expectations: AfmlBarExpectations,
        ewma_span: int = 20,
        threshold_scale: float = 1.0,
        imbalance_floor_frac: float = 0.0,
        update_expected_ticks: bool = True,
        meta_handler: Callable[[Bar, AfmlBarMeta], None] | None = None,
        threshold_provider: Callable[[int], float | None] | None = None,
        target_bars_per_day: int | None = None,
        density_adjustment_strength: float = 0.5,
        density_min_scale: float = 0.25,
        density_max_scale: float = 4.0,
        density_min_elapsed_fraction: float = 1.0 / 24.0,
    ) -> None:
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            handler=handler,
            meta_handler=meta_handler,
            expectations=expectations,
            ewma_span=ewma_span,
            threshold_scale=threshold_scale,
            update_expected_ticks=update_expected_ticks,
            threshold_provider=threshold_provider,
            target_bars_per_day=target_bars_per_day,
            density_adjustment_strength=density_adjustment_strength,
            density_min_scale=density_min_scale,
            density_max_scale=density_max_scale,
            density_min_elapsed_fraction=density_min_elapsed_fraction,
        )
        if imbalance_floor_frac < 0.0:
            raise ValueError("imbalance_floor_frac cannot be negative")
        self.imbalance_floor_frac = imbalance_floor_frac

    def handle_trade_tick(self, tick: TradeTick) -> None:
        if tick.aggressor_side == AggressorSide.NO_AGGRESSOR or float(tick.price) == 0.0:
            self.builder.update(tick.price, tick.size, tick.ts_init)
            return

        self._handle_valid_trade_tick(tick)

        expected_signed_notional = (
            self.expectations.p_buy * self.expectations.buy_mean_notional
            - (1.0 - self.expectations.p_buy) * self.expectations.sell_mean_notional
        )
        expected_abs = max(
            abs(expected_signed_notional),
            self.expectations.mean_notional * self.imbalance_floor_frac,
            1e-12,
        )
        threshold, threshold_scale = self._resolve_threshold(
            int(tick.ts_event),
            self.expectations.expected_ticks * expected_abs,
        )
        theta = abs(self.signed_notional)
        if theta < threshold:
            return

        self._build_now_and_send(
            AfmlBarMeta(
                ticks=self.ticks,
                buy_ticks=self.buy_ticks,
                sell_ticks=self.sell_ticks,
                buy_notional=self.buy_notional,
                sell_notional=self.sell_notional,
                signed_notional=self.signed_notional,
                theta=theta,
                threshold=threshold,
                expected_ticks=self.expectations.expected_ticks,
                expected_metric_a=expected_signed_notional,
                expected_metric_b=self.expectations.mean_notional,
                threshold_scale=threshold_scale,
            ),
        )


class AfmlDollarRunsBarAggregator(_AfmlDollarBarAggregator):
    """
    Builds AFML dynamic-threshold dollar runs bars from `TradeTick` data.

    The bar closes when:

    `max(theta_buy, theta_sell) >= E[T] * max(P_buy * E[v|buy], P_sell * E[v|sell])`

    Unlike NautilusTrader's fixed-threshold `ValueRunsBarAggregator`, AFML runs
    bars accumulate both sides within the bar and do not reset the bar when the
    aggressor side changes.
    """

    def __init__(
        self,
        *,
        instrument,
        bar_type: BarType,
        handler: Callable[[Bar], None] | None,
        expectations: AfmlBarExpectations,
        ewma_span: int = 20,
        threshold_scale: float = 1.0,
        update_expected_ticks: bool = True,
        meta_handler: Callable[[Bar, AfmlBarMeta], None] | None = None,
        threshold_provider: Callable[[int], float | None] | None = None,
        target_bars_per_day: int | None = None,
        density_adjustment_strength: float = 0.5,
        density_min_scale: float = 0.25,
        density_max_scale: float = 4.0,
        density_min_elapsed_fraction: float = 1.0 / 24.0,
    ) -> None:
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            handler=handler,
            meta_handler=meta_handler,
            expectations=expectations,
            ewma_span=ewma_span,
            threshold_scale=threshold_scale,
            update_expected_ticks=update_expected_ticks,
            threshold_provider=threshold_provider,
            target_bars_per_day=target_bars_per_day,
            density_adjustment_strength=density_adjustment_strength,
            density_min_scale=density_min_scale,
            density_max_scale=density_max_scale,
            density_min_elapsed_fraction=density_min_elapsed_fraction,
        )

    def handle_trade_tick(self, tick: TradeTick) -> None:
        if tick.aggressor_side == AggressorSide.NO_AGGRESSOR or float(tick.price) == 0.0:
            self.builder.update(tick.price, tick.size, tick.ts_init)
            return

        self._handle_valid_trade_tick(tick)

        expected_run_notional = max(
            self.expectations.p_buy * self.expectations.buy_mean_notional,
            (1.0 - self.expectations.p_buy) * self.expectations.sell_mean_notional,
            1e-12,
        )
        threshold, threshold_scale = self._resolve_threshold(
            int(tick.ts_event),
            self.expectations.expected_ticks * expected_run_notional,
        )
        theta = max(self.buy_notional, self.sell_notional)
        if theta < threshold:
            return

        self._build_now_and_send(
            AfmlBarMeta(
                ticks=self.ticks,
                buy_ticks=self.buy_ticks,
                sell_ticks=self.sell_ticks,
                buy_notional=self.buy_notional,
                sell_notional=self.sell_notional,
                signed_notional=self.signed_notional,
                theta=theta,
                threshold=threshold,
                expected_ticks=self.expectations.expected_ticks,
                expected_metric_a=self.expectations.p_buy,
                expected_metric_b=max(
                    self.expectations.buy_mean_notional,
                    self.expectations.sell_mean_notional,
                ),
                threshold_scale=threshold_scale,
            ),
        )


def count_afml_bars(
    aggregator_factory: Callable[[Callable[[Bar], None] | None], _AfmlDollarBarAggregator],
    ticks: list[TradeTick],
) -> int:
    aggregator = aggregator_factory(None)
    for tick in ticks:
        aggregator.handle_trade_tick(tick)
    return aggregator.bar_count


def calibrate_threshold_scale(
    count_fn: Callable[[float], int],
    target_bars: int,
) -> tuple[float, int]:
    """
    Calibrate a threshold scale to approximate a target bar count.

    The AFML recursion is not guaranteed to be monotonic in the scale parameter,
    so this uses a coarse-to-fine grid search rather than binary search.
    """
    if target_bars < 1:
        raise ValueError("target_bars must be positive")

    scales = [10 ** (-4.0 + i * (6.0 / 60.0)) for i in range(61)]
    best_scale = scales[0]
    best_count = count_fn(best_scale)
    best_error = abs(best_count - target_bars)

    for scale in scales[1:]:
        count = count_fn(scale)
        error = abs(count - target_bars)
        if error < best_error:
            best_scale = scale
            best_count = count
            best_error = error

    for _ in range(2):
        low = best_scale / 1.6
        high = best_scale * 1.6
        for i in range(1, 25):
            scale = low + (high - low) * i / 25.0
            count = count_fn(scale)
            error = abs(count - target_bars)
            if error < best_error:
                best_scale = scale
                best_count = count
                best_error = error

    return best_scale, best_count
