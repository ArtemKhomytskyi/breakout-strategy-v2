from __future__ import annotations
from typing import Iterable, Protocol, runtime_checkable
import pandas as pd
from atr_module.atr_module import DEFAULT_ATR_PERIOD, get_atr_from_bars


def compute_compression_series(
    df: pd.DataFrame,
    range_lookback: int = 10,
    atr_multiplier: float = 1.5,
) -> pd.Series:
    """
    Range Compression Filter (V2).

    A market is considered "compressed" if the recent price box is narrow relative
    to the baseline volatility (ATR):

        Range_Width[t] = max(High[t-k+1:t]) - min(Low[t-k+1:t])
        Compressed[t]  = Range_Width[t] <= ATR[t] * atr_multiplier

    Requirements:
      - df must contain columns: 'High', 'Low', 'ATR'

    Returns:
      - Boolean Series aligned to df.index.
      - Bars without sufficient history or ATR are treated as False (conservative).
    """
    if range_lookback <= 0:
        raise ValueError(f"range_lookback must be > 0, got {range_lookback}")
    if atr_multiplier <= 0:
        raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")

    required = {"High", "Low", "ATR"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"compute_compression_series requires columns {sorted(required)}, missing {sorted(missing)}")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    atr = df["ATR"].astype(float)

    rolling_max = high.rolling(window=range_lookback, min_periods=range_lookback).max()
    rolling_min = low.rolling(window=range_lookback, min_periods=range_lookback).min()
    range_width = (rolling_max - rolling_min).astype(float)

    compressed = range_width <= (atr * float(atr_multiplier))
    compressed = compressed.fillna(False)
    return compressed.astype(bool)


@runtime_checkable
class _BarLike(Protocol):
    high: float
    low: float
    close: float


def get_compression_state_at_bar(
    bars: Iterable[_BarLike],
    current_index: int,
    range_lookback: int = 10,
    atr_multiplier: float = 1.5,
) -> bool:
    """
    Online / live-loop compression check at a single bar index.

    Uses the same definition as compute_compression_series, but operates on a list
    of Bar-like objects (with .high/.low/.close) and computes ATR internally using
    the existing ATR module.

    Returns:
      - True  if compressed at current_index
      - False otherwise (including insufficient history / NaN ATR)
    """
    if range_lookback <= 0:
        raise ValueError(f"range_lookback must be > 0, got {range_lookback}")
    if atr_multiplier <= 0:
        raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")

    bars_list = list(bars)
    n = len(bars_list)
    if n == 0:
        raise IndexError("bars is empty")
    if current_index < 0 or current_index >= n:
        raise IndexError(f"current_index {current_index} out of range for bars length {n}")

    start = current_index - range_lookback + 1
    if start < 0:
        return False  # not enough history to define the box

    window = bars_list[start : current_index + 1]
    hi = max(float(b.high) for b in window)
    lo = min(float(b.low) for b in window)
    range_width = float(hi - lo)

    atr = float(get_atr_from_bars(bars_list, current_index, DEFAULT_ATR_PERIOD))
    if not pd.notna(atr) or atr <= 0:
        return False

    return range_width <= (atr * float(atr_multiplier))
