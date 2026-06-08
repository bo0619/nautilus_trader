import pandas as pd

from afml_strategies.cvdslope_primary_only import make_primary_only_signal_frame


def test_primary_only_signals_use_full_size_for_active_primary_sides():
    index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    primary = pd.DataFrame(
        {
            "primary_side": [1, -1, 0, 1],
            "primary_prediction": [1, -1, 0, 1],
            "primary_confidence": [1.0, 1.0, 1.0, 1.0],
            "primary_prob_-1": [0.0, 1.0, 0.0, 0.0],
            "primary_prob_0": [0.0, 0.0, 1.0, 0.0],
            "primary_prob_1": [1.0, 0.0, 0.0, 1.0],
            "primary_prob_margin": [1.0, -1.0, 0.0, 1.0],
        },
        index=index,
    )

    signals = make_primary_only_signal_frame(primary, active_index=pd.DatetimeIndex(index[:3]))

    assert signals["signal"].to_list() == [1, -1, 0, 0]
    assert signals["bet_size"].to_list() == [1.0, -1.0, 0.0, 0.0]
    assert signals["bet_size_abs"].to_list() == [1.0, 1.0, 0.0, 0.0]
    assert signals["confidence"].to_list() == [1.0, 1.0, 1.0, 1.0]
