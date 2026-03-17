"""
ADX Trend-Strength Filter (V2).

This module measures trend STRENGTH, not trend direction.
It does NOT generate trades or signals. It only answers:
  - Is trend strong enough for breakout trading? (ADX > threshold)
  - Is trend weak / range-like? (ADX <= threshold)

ADX rules:
  - trend strong: ADX > adx_threshold -> breakout trading allowed
  - trend weak:   ADX <= adx_threshold -> no breakout trading

This filter is meant to work TOGETHER with the EMA regime filter:
  - EMA filter checks direction (trend_up / trend_down / range)
  - ADX filter checks strength (strong enough for breakouts?)
  - Both must pass before BOS entry is allowed

Default parameters:
  - adx_period = 14
  - adx_threshold = 25
"""

from typing import List, Union

import pandas as pd

from atr_module.atr_module import wilder_smooth

DEFAULT_ADX_PERIOD = 14
DEFAULT_ADX_THRESHOLD = 25
def compute_adx(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
    period: int = DEFAULT_ADX_PERIOD,
) -> pd.Series:
    """
    Compute ADX (Average Directional Index) from OHLC data.

    ADX measures trend strength (0-100). Higher values indicate stronger trends.
    Does NOT indicate direction.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ADX period (default 14).

    Returns:
        ADX series (same length as input). Early bars may be NaN.
    """
    if isinstance(high, list):
        high = pd.Series(high)
    if isinstance(low, list):
        low = pd.Series(low)
    if isinstance(close, list):
        close = pd.Series(close)

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = wilder_smooth(tr, period)
    atr_ok = atr > 1e-10
    plus_di = (
        (100 * wilder_smooth(plus_dm, period) / atr)
        .where(atr_ok, 0.0)
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0)
    )
    minus_di = (
        (100 * wilder_smooth(minus_dm, period) / atr)
        .where(atr_ok, 0.0)
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0)
    )

    di_sum = plus_di + minus_di
    di_sum_ok = di_sum > 1e-10
    dx = (
        (100 * (plus_di - minus_di).abs() / di_sum)
        .where(di_sum_ok, 0.0)
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0)
    )

    adx = wilder_smooth(dx, period)
    adx = adx.replace([float("inf"), float("-inf")], 0.0).fillna(0)
    return adx


def is_trend_strong(adx_value: float, threshold: float = DEFAULT_ADX_THRESHOLD) -> bool:
    """
    True if trend strength is sufficient for breakout trading.

    Args:
        adx_value: ADX value at the bar.
        threshold: ADX threshold (default 25).

    Returns:
        True if ADX > threshold, False otherwise.
        Returns False if adx_value is NaN or inf (conservative).
    """
    if pd.isna(adx_value) or adx_value == float("inf") or adx_value == float("-inf"):
        return False
    return adx_value > threshold


def compute_adx_series(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
    adx_period: int = DEFAULT_ADX_PERIOD,
    adx_threshold: float = DEFAULT_ADX_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute ADX and trend strength for the full OHLC series at once.

    Returns a DataFrame with columns: close, adx, trend_strong.
    Bars without enough history are treated as weak trend (conservative).

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        adx_period: ADX period.
        adx_threshold: ADX threshold for trend strength.

    Returns:
        DataFrame with columns: close, adx, trend_strong.
    """
    if isinstance(high, list):
        high = pd.Series(high)
    if isinstance(low, list):
        low = pd.Series(low)
    if isinstance(close, list):
        close = pd.Series(close)

    adx = compute_adx(high, low, close, adx_period)
    min_history = adx_period * 2
    trend_strong = []
    for i in range(len(close)):
        if (i + 1) < min_history:
            trend_strong.append(False)
        else:
            trend_strong.append(is_trend_strong(float(adx.iloc[i]), adx_threshold))

    return pd.DataFrame(
        {"close": close, "adx": adx, "trend_strong": trend_strong},
        index=close.index,
    )


def get_adx_strength_at_bar(
    high: Union[pd.Series, List[float]],
    low: Union[pd.Series, List[float]],
    close: Union[pd.Series, List[float]],
    bar_index: int,
    adx_period: int = DEFAULT_ADX_PERIOD,
    adx_threshold: float = DEFAULT_ADX_THRESHOLD,
) -> bool:
    """
    Check if trend is strong enough at a specific bar index.

    Convenience function: uses compute_adx_series on data up to bar_index,
    returns trend_strong at bar_index. Returns False if insufficient history.

    Args:
        high: High prices (full history up to and including current bar).
        low: Low prices.
        close: Close prices.
        bar_index: Index of bar to check (0-based).
        adx_period: ADX period.
        adx_threshold: ADX threshold.

    Returns:
        True if ADX > threshold and enough history, False otherwise.

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

    slice_high = high.iloc[: bar_index + 1]
    slice_low = low.iloc[: bar_index + 1]
    slice_close = close.iloc[: bar_index + 1]
    df = compute_adx_series(slice_high, slice_low, slice_close, adx_period, adx_threshold)
    return bool(df.iloc[-1]["trend_strong"])


def get_adx_strength_from_bars(
    bars: List,
    bar_index: int,
    adx_period: int = DEFAULT_ADX_PERIOD,
    adx_threshold: float = DEFAULT_ADX_THRESHOLD,
) -> bool:
    """
    Check trend strength at bar_index from a list of Bar-like objects.

    Bar-like objects must have .high, .low, .close attributes.
    For use with bars_15m in main.py / QuantConnect.

    Args:
        bars: List of Bar objects (or any with .high, .low, .close).
        bar_index: Index of bar to check.
        adx_period: ADX period.
        adx_threshold: ADX threshold.

    Returns:
        True if trend is strong enough, False otherwise.

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
    return get_adx_strength_at_bar(high, low, close, bar_index, adx_period, adx_threshold)
