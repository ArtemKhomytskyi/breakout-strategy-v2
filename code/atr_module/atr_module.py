"""
ATR Volatility Module (V2).

This module measures market VOLATILITY, not direction.
It does NOT generate trades or signals. It does NOT allow or block entries.
It only provides ATR (Average True Range) as infrastructure for downstream use.

ATR meaning:
  - higher ATR = wider / more volatile market
  - lower ATR = calmer / tighter market

This module is infrastructure. It will later be used by:
  - ATR-based Stop Loss
  - Breakout Buffer
  - Trailing Stop
  - other volatility-aware logic

Default parameter:
  - atr_period = 14
"""

from typing import List, Union

import pandas as pd

DEFAULT_ATR_PERIOD = 14


def wilder_smooth(series: Union[pd.Series, List[float]], period: int) -> pd.Series:
    """
    Wilder's smoothing (RMA) seeded with a simple average.

    The first non-NaN value is emitted at index ``period - 1`` and is the simple
    average of the first ``period`` observations. Later values use the standard
    Wilder recurrence:

        rma[t] = (rma[t-1] * (period - 1) + value[t]) / period
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    if isinstance(series, list):
        series = pd.Series(series, dtype=float)
    else:
        series = series.astype(float).copy()

    smoothed = pd.Series(float("nan"), index=series.index, dtype=float)
    if len(series) < period:
        return smoothed

    seed = float(series.iloc[:period].mean())
    smoothed.iloc[period - 1] = seed

    prev = seed
    for i in range(period, len(series)):
        prev = wilder_smooth_step(prev, float(series.iloc[i]), period)
        smoothed.iloc[i] = prev

    return smoothed


def wilder_smooth_step(previous_value: float, new_value: float, period: int) -> float:
    """Single-step Wilder smoothing update."""
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    return ((previous_value * (period - 1)) + new_value) / period


def compute_true_range(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
) -> pd.Series:
    """
    Compute True Range from OHLC data.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    First bar: TR = high - low (no prev_close).

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        True Range series (same length as input).
    """
    if isinstance(high, list):
        high = pd.Series(high)
    if isinstance(low, list):
        low = pd.Series(low)
    if isinstance(close, list):
        close = pd.Series(close)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    tr = tr.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
    return tr


def compute_atr(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.Series:
    """
    Compute ATR (Average True Range) from OHLC data.

    ATR = Wilder's smoothing of True Range.
    Measures volatility only; does not indicate direction.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ATR period (default 14).

    Returns:
        ATR series (same length as input). Early bars may have partial warmup.
    """
    if isinstance(high, list):
        high = pd.Series(high)
    if isinstance(low, list):
        low = pd.Series(low)
    if isinstance(close, list):
        close = pd.Series(close)

    tr = compute_true_range(high, low, close)
    atr = wilder_smooth(tr, period)
    atr = atr.replace([float("inf"), float("-inf")], float("nan"))
    return atr


def compute_atr_series(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """
    Compute TR and ATR for the full OHLC series at once.

    Returns a DataFrame with columns: close, tr, atr.
    Bars without enough history have NaN in atr (conservative).

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        atr_period: ATR period.

    Returns:
        DataFrame with columns: close, tr, atr.
    """
    if isinstance(high, list):
        high = pd.Series(high)
    if isinstance(low, list):
        low = pd.Series(low)
    if isinstance(close, list):
        close = pd.Series(close)

    tr = compute_true_range(high, low, close)
    atr = compute_atr(high, low, close, atr_period)

    min_history = atr_period
    atr_safe = atr.copy()
    atr_safe.iloc[: min_history - 1] = float("nan")

    return pd.DataFrame(
        {"close": close, "tr": tr, "atr": atr_safe},
        index=close.index,
    )


def get_atr_at_bar(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
    bar_index: int,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> float:
    """
    Get ATR value at a specific bar index.

    Convenience function: uses compute_atr_series on data up to bar_index,
    returns ATR at bar_index. Returns NaN if insufficient history.

    Args:
        high: High prices (full history up to and including current bar).
        low: Low prices.
        close: Close prices.
        bar_index: Index of bar to query (0-based).
        atr_period: ATR period.

    Returns:
        ATR value at bar_index, or NaN if insufficient history.

    Raises:
        IndexError: If bar_index is negative or out of range.
    """
    if isinstance(high, list):
        high = pd.Series(high)
    if isinstance(low, list):
        low = pd.Series(low)
    if isinstance(close, list):
        close = pd.Series(close)

    n = len(close)
    if bar_index < 0:
        raise IndexError(f"bar_index must be >= 0, got {bar_index}")
    if bar_index >= n:
        raise IndexError(f"bar_index {bar_index} out of range for series of length {n}")

    min_history = atr_period
    if (bar_index + 1) < min_history:
        return float("nan")

    slice_high = high.iloc[: bar_index + 1]
    slice_low = low.iloc[: bar_index + 1]
    slice_close = close.iloc[: bar_index + 1]
    df = compute_atr_series(slice_high, slice_low, slice_close, atr_period)
    return float(df.iloc[-1]["atr"])


def get_atr_from_bars(
    bars: List,
    bar_index: int,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> float:
    """
    Get ATR value at bar_index from a list of Bar-like objects.

    Bar-like objects must have .high, .low, .close attributes.
    For use with bars_15m in main.py / QuantConnect.

    Args:
        bars: List of Bar objects (or any with .high, .low, .close).
        bar_index: Index of bar to query.
        atr_period: ATR period.

    Returns:
        ATR value at bar_index, or NaN if insufficient history.

    Raises:
        IndexError: If bar_index is negative or out of range.
    """
    n = len(bars)
    if n == 0:
        raise IndexError("bars list is empty")
    if bar_index < 0:
        raise IndexError(f"bar_index must be >= 0, got {bar_index}")
    if bar_index >= n:
        raise IndexError(f"bar_index {bar_index} out of range for bars list of length {n}")

    high = [float(b.high) for b in bars]
    low = [float(b.low) for b in bars]
    close = [float(b.close) for b in bars]
    return get_atr_at_bar(high, low, close, bar_index, atr_period)
