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

import pytest

from nautilus_trader.data.afml_bars import DAY_NANOS
from nautilus_trader.data.afml_bars import AfmlBarExpectations
from nautilus_trader.data.afml_bars import AfmlDollarImbalanceBarAggregator
from nautilus_trader.data.afml_bars import AfmlDollarRunsBarAggregator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider


BTCUSDT_BINANCE = TestInstrumentProvider.btcusdt_perp_binance()


def _bar_type(aggregation: BarAggregation) -> BarType:
    return BarType(
        BTCUSDT_BINANCE.id,
        BarSpecification(1440, aggregation, PriceType.LAST),
    )


def _expectations() -> AfmlBarExpectations:
    return AfmlBarExpectations(
        expected_ticks=2.0,
        p_buy=0.5,
        buy_mean_notional=100.0,
        sell_mean_notional=100.0,
        mean_notional=100.0,
    )


def _trade(index: int, side: AggressorSide) -> TradeTick:
    return TradeTick(
        instrument_id=BTCUSDT_BINANCE.id,
        price=Price.from_str("100.0"),
        size=Quantity.from_str("1"),
        aggressor_side=side,
        trade_id=TradeId(str(index)),
        ts_event=index,
        ts_init=index,
    )


def test_afml_dollar_imbalance_bar_closes_on_dynamic_threshold():
    bars: list[Bar] = []
    metas = []
    aggregator = AfmlDollarImbalanceBarAggregator(
        instrument=BTCUSDT_BINANCE,
        bar_type=_bar_type(BarAggregation.VALUE_IMBALANCE),
        handler=bars.append,
        meta_handler=lambda _bar, meta: metas.append(meta),
        expectations=_expectations(),
        threshold_scale=1.0,
        imbalance_floor_frac=1.0,
        update_expected_ticks=False,
    )

    aggregator.handle_trade_tick(_trade(1, AggressorSide.BUYER))
    aggregator.handle_trade_tick(_trade(2, AggressorSide.BUYER))

    assert aggregator.bar_count == 1
    assert len(bars) == 1
    assert metas[0].ticks == 2
    assert metas[0].theta == 200.0
    assert metas[0].threshold == 200.0


def test_afml_dollar_runs_bar_closes_on_dynamic_threshold():
    bars: list[Bar] = []
    metas = []
    aggregator = AfmlDollarRunsBarAggregator(
        instrument=BTCUSDT_BINANCE,
        bar_type=_bar_type(BarAggregation.VALUE_RUNS),
        handler=bars.append,
        meta_handler=lambda _bar, meta: metas.append(meta),
        expectations=_expectations(),
        threshold_scale=2.0,
        update_expected_ticks=False,
    )

    aggregator.handle_trade_tick(_trade(1, AggressorSide.BUYER))
    aggregator.handle_trade_tick(_trade(2, AggressorSide.SELLER))
    aggregator.handle_trade_tick(_trade(3, AggressorSide.BUYER))

    assert aggregator.bar_count == 1
    assert len(bars) == 1
    assert metas[0].ticks == 3
    assert metas[0].theta == 200.0
    assert metas[0].threshold == 200.0


def test_afml_dollar_runs_bar_can_use_scheduled_absolute_threshold():
    bars: list[Bar] = []
    metas = []
    aggregator = AfmlDollarRunsBarAggregator(
        instrument=BTCUSDT_BINANCE,
        bar_type=_bar_type(BarAggregation.VALUE_RUNS),
        handler=bars.append,
        meta_handler=lambda _bar, meta: metas.append(meta),
        expectations=_expectations(),
        threshold_scale=10.0,
        threshold_provider=lambda _ts_ns: 150.0,
        update_expected_ticks=False,
    )

    aggregator.handle_trade_tick(_trade(1, AggressorSide.BUYER))
    aggregator.handle_trade_tick(_trade(2, AggressorSide.BUYER))

    assert aggregator.bar_count == 1
    assert len(bars) == 1
    assert metas[0].theta == 200.0
    assert metas[0].threshold == 150.0
    assert metas[0].threshold_scale == pytest.approx(1.5)


def test_adaptive_daily_density_raises_scale_when_bar_count_is_ahead_of_schedule():
    aggregator = AfmlDollarRunsBarAggregator(
        instrument=BTCUSDT_BINANCE,
        bar_type=_bar_type(BarAggregation.VALUE_RUNS),
        handler=None,
        expectations=_expectations(),
        threshold_scale=2.0,
        update_expected_ticks=False,
        target_bars_per_day=24,
        density_adjustment_strength=1.0,
        density_min_scale=0.25,
        density_max_scale=8.0,
        density_min_elapsed_fraction=1.0 / 24.0,
    )

    noon = DAY_NANOS // 2
    assert aggregator._effective_threshold_scale(noon) == 0.5

    aggregator._density_day_bar_count = 48

    assert aggregator._effective_threshold_scale(noon) == pytest.approx(2.0 * 49.0 / 12.0)
