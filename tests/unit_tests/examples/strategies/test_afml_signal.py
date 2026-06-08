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

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nautilus_trader.examples.strategies.afml_signal import AfmlSignalStrategy
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Quantity


def test_afml_signal_strategy_loads_timestamp_ns_csv():
    output_dir = Path("build") / "afml_test_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"strategy_signals_{uuid4().hex}.csv"
    frame = pd.DataFrame(
        {
            "timestamp_ns": [1_700_000_000_000_000_000, 1_700_000_060_000_000_000],
            "signal": [1, -1],
            "confidence": [0.7, 0.8],
        },
    )
    frame.to_csv(path, index=False)

    signals = AfmlSignalStrategy._load_signals(str(path))

    assert signals[1_700_000_000_000_000_000] == (1, 0.7, 1.0, None, None, None)
    assert signals[1_700_000_060_000_000_000] == (-1, 0.8, -1.0, None, None, None)


def test_afml_signal_strategy_loads_bet_size_csv():
    output_dir = Path("build") / "afml_test_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"strategy_sized_signals_{uuid4().hex}.csv"
    frame = pd.DataFrame(
        {
            "timestamp_ns": [1_700_000_000_000_000_000],
            "signal": [1],
            "confidence": [0.9],
            "bet_size": [0.35],
            "trgt": [0.012],
        },
    )
    frame.to_csv(path, index=False)

    signals = AfmlSignalStrategy._load_signals(str(path))

    assert signals[1_700_000_000_000_000_000] == (1, 0.9, 0.35, 0.012, None, None)


def test_afml_signal_strategy_loads_signal_level_barrier_multipliers():
    output_dir = Path("build") / "afml_test_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"strategy_sized_barrier_signals_{uuid4().hex}.csv"
    frame = pd.DataFrame(
        {
            "timestamp_ns": [1_700_000_000_000_000_000],
            "signal": [-1],
            "confidence": [0.9],
            "bet_size": [-0.5],
            "trgt": [0.012],
            "profit_taking_mult": [0.5],
            "stop_loss_mult": [2.0],
        },
    )
    frame.to_csv(path, index=False)

    signals = AfmlSignalStrategy._load_signals(str(path))

    assert signals[1_700_000_000_000_000_000] == (-1, 0.9, -0.5, 0.012, 0.5, 2.0)


def test_afml_signal_strategy_calculates_same_side_position_adjustments():
    increase = AfmlSignalStrategy._position_adjustment(Decimal(3), Decimal(6))
    reduce = AfmlSignalStrategy._position_adjustment(Decimal(-6), Decimal(-3))
    unchanged = AfmlSignalStrategy._position_adjustment(
        Decimal(3),
        Decimal("3.01"),
        Decimal("0.05"),
    )

    assert increase == (OrderSide.BUY, Decimal(3), False)
    assert reduce == (OrderSide.BUY, Decimal(3), True)
    assert unchanged is None


def test_afml_signal_strategy_quantity_zero_check_uses_runtime_raw_value():
    assert AfmlSignalStrategy._quantity_is_zero(Quantity(0, 0))
    assert not hasattr(Quantity(0, 0), "is_zero")
