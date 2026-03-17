import math
import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from RSI.momentum_confirmation_rsi import (  # noqa: E402
    RSIEngine,
    RSIMomentumFilter,
    RSIMomentumMode,
    compute_rsi,
    get_rsi_at_bar,
)
from entry_exit_rules.entry_exit import PositionDirection  # noqa: E402


class RSITestCase(unittest.TestCase):
    def test_incremental_engine_matches_batch_series(self):
        closes = [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
            46.21,
        ]

        batch = compute_rsi(pd.Series(closes), period=14)
        engine = RSIEngine(length=14)
        incremental = [engine.update(close) for close in closes]

        self.assertEqual(len(batch), len(incremental))
        for batch_value, incremental_value in zip(batch.tolist(), incremental):
            if pd.isna(batch_value):
                self.assertIsNone(incremental_value)
            else:
                self.assertIsNotNone(incremental_value)
                self.assertAlmostEqual(float(batch_value), float(incremental_value), places=10)

    def test_get_rsi_at_bar_matches_batch_value(self):
        closes = [100, 101, 102, 101, 103, 104, 103, 105, 107, 106, 108, 109, 108, 110, 111, 110, 112]
        batch = compute_rsi(closes, period=14)
        last_index = len(closes) - 1

        self.assertTrue(math.isnan(get_rsi_at_bar(closes, 10, rsi_period=14)))
        self.assertAlmostEqual(
            get_rsi_at_bar(closes, last_index, rsi_period=14),
            float(batch.iloc[last_index]),
            places=10,
        )

    def test_threshold_filter_blocks_neutral_longs(self):
        filt = RSIMomentumFilter(mode=RSIMomentumMode.THRESHOLD, long_threshold=55.0, short_threshold=45.0)
        decision = filt.allow_entry(direction=PositionDirection.LONG, rsi_now=52.0)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "RSI_LONG_THRESHOLD_BLOCK")
        self.assertEqual(decision.threshold, 55.0)

    def test_cross_filter_requires_transition(self):
        filt = RSIMomentumFilter(mode=RSIMomentumMode.CROSS, cross_level_long=50.0, cross_level_short=50.0)
        blocked = filt.allow_entry(direction=PositionDirection.LONG, rsi_now=56.0, rsi_prev=54.0)
        allowed = filt.allow_entry(direction=PositionDirection.LONG, rsi_now=50.0, rsi_prev=49.0)

        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, "RSI_LONG_CROSS_BLOCK")
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.reason, "RSI_LONG_CROSS_PASS")


if __name__ == "__main__":
    unittest.main()
