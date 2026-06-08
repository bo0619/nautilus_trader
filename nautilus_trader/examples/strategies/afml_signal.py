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
Nautilus strategy that executes precomputed AFML model signals.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import PositionClosed
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


class AfmlSignalStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``AfmlSignalStrategy``.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    signal_path: str
    trade_size: Decimal
    strategy_config_path: str | None = "afml_strategies/cvdslope.json"
    confidence_threshold: float = 0.0
    flatten_on_zero: bool = True
    profit_taking_mult: float | None = None
    stop_loss_mult: float | None = None
    vertical_barrier_bars: int | None = None
    vertical_barrier_days: float | None = None
    request_bars: bool = False
    unsubscribe_data_on_stop: bool = True
    order_quantity_precision: int | None = None
    order_time_in_force: TimeInForce | None = None
    allow_position_resizing: bool = False
    min_position_change: Decimal = Decimal(0)
    reset_barrier_on_resize: bool = False
    close_positions_on_stop: bool = True
    reduce_only_on_stop: bool = True


class AfmlSignalStrategy(Strategy):
    """
    Executes AFML signals as Chapter-10 target position sizes.
    """

    def __init__(self, config: AfmlSignalStrategyConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self._strategy_config = self._load_strategy_config(config.strategy_config_path)
        self._profit_taking_mult = self._configured_float(
            config.profit_taking_mult,
            "profit_taking_mult",
        )
        self._stop_loss_mult = self._configured_float(config.stop_loss_mult, "stop_loss_mult")
        self._vertical_barrier_bars = self._configured_int(
            config.vertical_barrier_bars,
            "vertical_barrier_bars",
        )
        self._vertical_barrier_days = self._configured_float(
            config.vertical_barrier_days,
            "vertical_barrier_days",
        )
        self._signals = self._load_signals(config.signal_path)
        self._pending_target_side: OrderSide | None = None
        self._pending_target_size_multiplier = Decimal(1)
        self._pending_entry_price: float | None = None
        self._pending_entry_target: float | None = None
        self._pending_entry_profit_taking_mult: float | None = None
        self._pending_entry_stop_loss_mult: float | None = None
        self._pending_entry_ts_event: int | None = None
        self._close_pending = False
        self._entry_side: OrderSide | None = None
        self._entry_price: float | None = None
        self._entry_target: float | None = None
        self._entry_profit_taking_mult: float | None = None
        self._entry_stop_loss_mult: float | None = None
        self._entry_ts_event: int | None = None
        self._entry_bar_count = 0

    @staticmethod
    def _load_strategy_config(path: str | None) -> dict[str, Any]:
        if path is None:
            return {}
        strategy_path = Path(path)
        if not strategy_path.exists():
            if path == "afml_strategies/cvdslope.json":
                return {}
            raise FileNotFoundError(f"Strategy config does not exist: {strategy_path}")
        payload = json.loads(strategy_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Strategy config root must be a JSON object: {strategy_path}")
        return payload

    def _runtime_strategy_config(self) -> dict[str, Any]:
        value = self._strategy_config.get("runtime_strategy", {})
        if not isinstance(value, dict):
            raise TypeError("strategy_config.runtime_strategy must be a JSON object")
        return value

    def _barrier_config(self) -> dict[str, Any]:
        value = self._strategy_config.get("barrier_optimization", {})
        if not isinstance(value, dict):
            raise TypeError("strategy_config.barrier_optimization must be a JSON object")
        return value

    def _configured_float(self, explicit: float | None, key: str) -> float | None:
        if explicit is not None:
            return float(explicit)
        runtime = self._runtime_strategy_config()
        value = runtime.get(key)
        if value is None:
            return None
        return float(value)

    def _configured_int(self, explicit: int | None, key: str) -> int | None:
        if explicit is not None:
            return int(explicit)
        runtime = self._runtime_strategy_config()
        value = runtime.get(key)
        if value is None and key == "vertical_barrier_bars":
            value = self._barrier_config().get("vertical_barrier_bars")
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _load_signals(path: str) -> dict[int, tuple[int, float, float, float | None, float | None, float | None]]:
        signal_path = Path(path)
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal file does not exist: {signal_path}")

        frame = pd.read_csv(signal_path, low_memory=False)
        if "signal" not in frame.columns:
            raise ValueError("Signal CSV must contain a 'signal' column")
        if "timestamp_ns" in frame.columns:
            timestamps = frame["timestamp_ns"].astype("int64")
        elif "timestamp" in frame.columns:
            timestamps = pd.to_datetime(frame["timestamp"], utc=True).astype("int64")
        else:
            raise ValueError("Signal CSV must contain 'timestamp_ns' or 'timestamp'")

        confidence = (
            frame["confidence"].astype(float)
            if "confidence" in frame.columns
            else pd.Series(1.0, index=frame.index)
        )
        bet_size = (
            frame["bet_size"].astype(float)
            if "bet_size" in frame.columns
            else frame["signal"].astype(float)
        )
        target = (
            frame["trgt"].astype(float)
            if "trgt" in frame.columns
            else frame["target"].astype(float)
            if "target" in frame.columns
            else pd.Series(float("nan"), index=frame.index)
        )
        profit_taking_mult = (
            frame["profit_taking_mult"].astype(float)
            if "profit_taking_mult" in frame.columns
            else pd.Series(float("nan"), index=frame.index)
        )
        stop_loss_mult = (
            frame["stop_loss_mult"].astype(float)
            if "stop_loss_mult" in frame.columns
            else pd.Series(float("nan"), index=frame.index)
        )
        return {
            int(timestamp): (
                int(signal),
                float(conf),
                float(size),
                float(tgt) if pd.notna(tgt) else None,
                float(pt) if pd.notna(pt) else None,
                float(sl) if pd.notna(sl) else None,
            )
            for timestamp, signal, conf, size, tgt, pt, sl in zip(
                timestamps,
                frame["signal"],
                confidence,
                bet_size,
                target,
                profit_taking_mult,
                stop_loss_mult,
                strict=True,
            )
        }

    def on_start(self) -> None:
        """
        Subscribe to bars and optionally request recent history.
        """
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        if self.config.request_bars:
            self.request_bars(self.config.bar_type)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        """
        Execute a signal exactly timestamped to the completed bar.
        """
        if self._check_triple_barrier_exit(bar):
            return

        item = self._signals.get(int(bar.ts_event))
        if item is None:
            return

        signal, confidence, bet_size, target, profit_taking_mult, stop_loss_mult = item
        if confidence < self.config.confidence_threshold:
            signal = 0

        size_multiplier = Decimal(str(max(0.0, min(1.0, abs(bet_size)))))
        if size_multiplier == 0:
            signal = 0

        if signal > 0:
            self._target_long(
                size_multiplier,
                bar=bar,
                target=target,
                profit_taking_mult=profit_taking_mult,
                stop_loss_mult=stop_loss_mult,
            )
        elif signal < 0:
            self._target_short(
                size_multiplier,
                bar=bar,
                target=target,
                profit_taking_mult=profit_taking_mult,
                stop_loss_mult=stop_loss_mult,
            )
        elif self.config.flatten_on_zero:
            self._flatten()

    def _target_long(
        self,
        size_multiplier: Decimal,
        *,
        bar: Bar,
        target: float | None,
        profit_taking_mult: float | None,
        stop_loss_mult: float | None,
    ) -> None:
        self._target_side(
            OrderSide.BUY,
            size_multiplier,
            bar=bar,
            target=target,
            profit_taking_mult=profit_taking_mult,
            stop_loss_mult=stop_loss_mult,
        )

    def _target_short(
        self,
        size_multiplier: Decimal,
        *,
        bar: Bar,
        target: float | None,
        profit_taking_mult: float | None,
        stop_loss_mult: float | None,
    ) -> None:
        self._target_side(
            OrderSide.SELL,
            size_multiplier,
            bar=bar,
            target=target,
            profit_taking_mult=profit_taking_mult,
            stop_loss_mult=stop_loss_mult,
        )

    def _target_side(
        self,
        side: OrderSide,
        size_multiplier: Decimal,
        *,
        bar: Bar,
        target: float | None,
        profit_taking_mult: float | None,
        stop_loss_mult: float | None,
    ) -> None:
        if self._close_pending:
            self._set_pending_target(
                side,
                size_multiplier,
                bar=bar,
                target=target,
                profit_taking_mult=profit_taking_mult,
                stop_loss_mult=stop_loss_mult,
            )
            return

        target_signed_qty = self._target_signed_qty(side, size_multiplier)
        current_signed_qty = self._current_signed_qty()

        if self.portfolio.is_flat(self.config.instrument_id):
            self._pending_target_side = None
            self._pending_target_size_multiplier = Decimal(1)
            self._close_pending = False
            self._submit_market(side, abs(target_signed_qty))
            self._set_barrier_entry(
                side,
                bar=bar,
                target=target,
                profit_taking_mult=profit_taking_mult,
                stop_loss_mult=stop_loss_mult,
            )
            return

        if self._same_position_side(current_signed_qty, target_signed_qty):
            if self.config.allow_position_resizing:
                resized = self._resize_position_to_target(current_signed_qty, target_signed_qty)
                if resized and self.config.reset_barrier_on_resize:
                    self._set_barrier_entry(
                        side,
                        bar=bar,
                        target=target,
                        profit_taking_mult=profit_taking_mult,
                        stop_loss_mult=stop_loss_mult,
                    )
            return

        self._set_pending_target(
            side,
            size_multiplier,
            bar=bar,
            target=target,
            profit_taking_mult=profit_taking_mult,
            stop_loss_mult=stop_loss_mult,
        )
        if not self._close_pending:
            self._close_pending = True
            self._close_positions_reduce_only()

    def _set_pending_target(
        self,
        side: OrderSide,
        size_multiplier: Decimal,
        *,
        bar: Bar,
        target: float | None,
        profit_taking_mult: float | None,
        stop_loss_mult: float | None,
    ) -> None:
        self._pending_target_side = side
        self._pending_target_size_multiplier = size_multiplier
        self._pending_entry_price = float(bar.close)
        self._pending_entry_target = target
        self._pending_entry_profit_taking_mult = profit_taking_mult
        self._pending_entry_stop_loss_mult = stop_loss_mult
        self._pending_entry_ts_event = int(bar.ts_event)

    def _target_signed_qty(self, side: OrderSide, size_multiplier: Decimal) -> Decimal:
        target_qty = self.config.trade_size * size_multiplier
        return target_qty if side == OrderSide.BUY else -target_qty

    def _current_signed_qty(self) -> Decimal:
        return Decimal(str(self.portfolio.net_position(self.config.instrument_id)))

    @staticmethod
    def _same_position_side(current_signed_qty: Decimal, target_signed_qty: Decimal) -> bool:
        return (
            (current_signed_qty > 0 and target_signed_qty > 0)
            or (current_signed_qty < 0 and target_signed_qty < 0)
        )

    @staticmethod
    def _position_adjustment(
        current_signed_qty: Decimal,
        target_signed_qty: Decimal,
        min_position_change: Decimal = Decimal(0),
    ) -> tuple[OrderSide, Decimal, bool] | None:
        delta = target_signed_qty - current_signed_qty
        if abs(delta) <= min_position_change:
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        reduce_only = (
            (current_signed_qty > 0 and delta < 0)
            or (current_signed_qty < 0 and delta > 0)
        )
        return side, abs(delta), reduce_only

    def _resize_position_to_target(
        self,
        current_signed_qty: Decimal,
        target_signed_qty: Decimal,
    ) -> bool:
        adjustment = self._position_adjustment(
            current_signed_qty,
            target_signed_qty,
            self.config.min_position_change,
        )
        if adjustment is None:
            return False
        side, quantity, reduce_only = adjustment
        return self._submit_market(side, quantity, reduce_only=reduce_only)

    def _flatten(self) -> None:
        self._pending_target_side = None
        self._pending_target_size_multiplier = Decimal(1)
        self._pending_entry_price = None
        self._pending_entry_target = None
        self._pending_entry_profit_taking_mult = None
        self._pending_entry_stop_loss_mult = None
        self._pending_entry_ts_event = None
        if self.portfolio.is_flat(self.config.instrument_id):
            self._close_pending = False
            self._clear_barrier_entry()
            return
        if not self._close_pending:
            self._close_pending = True
            self._close_positions_reduce_only()

    def _check_triple_barrier_exit(self, bar: Bar) -> bool:
        if self._entry_side is None or self._entry_price is None:
            return False
        if self._close_pending or self.portfolio.is_flat(self.config.instrument_id):
            return False

        self._entry_bar_count += 1
        side = 1.0 if self._entry_side == OrderSide.BUY else -1.0
        path_return = (float(bar.close) / self._entry_price - 1.0) * side
        stop_loss_mult = self._entry_stop_loss_multiplier()
        profit_taking_mult = self._entry_profit_taking_multiplier()

        if self._entry_target is not None and self._entry_target > 0.0:
            if (
                stop_loss_mult is not None
                and stop_loss_mult > 0.0
                and path_return < -stop_loss_mult * self._entry_target
            ):
                self._flatten()
                return True
            if (
                profit_taking_mult is not None
                and profit_taking_mult > 0.0
                and path_return > profit_taking_mult * self._entry_target
            ):
                self._flatten()
                return True

        if (
            self._vertical_barrier_bars is not None
            and self._entry_bar_count >= self._vertical_barrier_bars
        ):
            self._flatten()
            return True

        if (
            self._vertical_barrier_days is not None
            and self._entry_ts_event is not None
            and int(bar.ts_event) - self._entry_ts_event
            >= int(self._vertical_barrier_days * 86_400_000_000_000)
        ):
            self._flatten()
            return True

        return False

    def _entry_profit_taking_multiplier(self) -> float | None:
        return (
            self._entry_profit_taking_mult
            if self._entry_profit_taking_mult is not None
            else self._profit_taking_mult
        )

    def _entry_stop_loss_multiplier(self) -> float | None:
        return self._entry_stop_loss_mult if self._entry_stop_loss_mult is not None else self._stop_loss_mult

    def _set_barrier_entry(
        self,
        side: OrderSide,
        *,
        bar: Bar,
        target: float | None,
        profit_taking_mult: float | None,
        stop_loss_mult: float | None,
    ) -> None:
        self._entry_side = side
        self._entry_price = float(bar.close)
        self._entry_target = target
        self._entry_profit_taking_mult = profit_taking_mult
        self._entry_stop_loss_mult = stop_loss_mult
        self._entry_ts_event = int(bar.ts_event)
        self._entry_bar_count = 0

    def _clear_barrier_entry(self) -> None:
        self._entry_side = None
        self._entry_price = None
        self._entry_target = None
        self._entry_profit_taking_mult = None
        self._entry_stop_loss_mult = None
        self._entry_ts_event = None
        self._entry_bar_count = 0

    def _close_positions_reduce_only(self) -> None:
        self.close_all_positions(
            self.config.instrument_id,
            time_in_force=self.config.order_time_in_force or TimeInForce.GTC,
            reduce_only=True,
        )

    def on_position_closed(self, event: PositionClosed) -> None:
        if event.instrument_id != self.config.instrument_id:
            return

        pending_side = self._pending_target_side
        if not self.portfolio.is_flat(self.config.instrument_id):
            self._close_pending = True
            return

        self._close_pending = False
        self._pending_target_side = None
        if pending_side is not None:
            pending_size_multiplier = self._pending_target_size_multiplier
            pending_price = self._pending_entry_price
            pending_target = self._pending_entry_target
            pending_profit_taking_mult = self._pending_entry_profit_taking_mult
            pending_stop_loss_mult = self._pending_entry_stop_loss_mult
            pending_ts_event = self._pending_entry_ts_event
            self._pending_target_size_multiplier = Decimal(1)
            self._pending_entry_price = None
            self._pending_entry_target = None
            self._pending_entry_profit_taking_mult = None
            self._pending_entry_stop_loss_mult = None
            self._pending_entry_ts_event = None
            self._submit_market(pending_side, self.config.trade_size * pending_size_multiplier)
            if pending_price is not None and pending_ts_event is not None:
                self._entry_side = pending_side
                self._entry_price = pending_price
                self._entry_target = pending_target
                self._entry_profit_taking_mult = pending_profit_taking_mult
                self._entry_stop_loss_mult = pending_stop_loss_mult
                self._entry_ts_event = pending_ts_event
                self._entry_bar_count = 0
        else:
            self._clear_barrier_entry()

    def _submit_market(
        self,
        side: OrderSide,
        quantity: Decimal,
        *,
        reduce_only: bool = False,
    ) -> bool:
        if quantity <= 0:
            return False
        if self.config.min_position_change > 0 and quantity < self.config.min_position_change:
            return False
        order_qty = self._order_qty(quantity)
        if self._quantity_is_zero(order_qty):
            return False

        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=order_qty,
            time_in_force=self.config.order_time_in_force or TimeInForce.GTC,
            reduce_only=reduce_only,
        )
        self.submit_order(order)
        return True

    @staticmethod
    def _quantity_is_zero(quantity: Quantity) -> bool:
        return quantity.raw == 0

    def _order_qty(self, quantity: Decimal) -> Quantity:
        assert self.instrument is not None
        if self.config.order_quantity_precision is not None:
            return Quantity(quantity, self.config.order_quantity_precision)
        return self.instrument.make_qty(quantity)

    def on_stop(self) -> None:
        """
        Clean up orders, positions, and subscriptions.
        """
        self._pending_target_side = None
        self._pending_entry_price = None
        self._pending_entry_target = None
        self._pending_entry_profit_taking_mult = None
        self._pending_entry_stop_loss_mult = None
        self._pending_entry_ts_event = None
        self._close_pending = False
        self._clear_barrier_entry()
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(
                instrument_id=self.config.instrument_id,
                reduce_only=self.config.reduce_only_on_stop,
            )
        if self.config.unsubscribe_data_on_stop:
            self.unsubscribe_bars(self.config.bar_type)
