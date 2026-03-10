"""
EMA Regime Filter (V2).

This module classifies market regime and gates BOS-based entries.
It does NOT generate trades or signals. It only answers:
  - Is a LONG entry allowed? (only in trend_up)
  - Is a SHORT entry allowed? (only in trend_down)
  - range -> no entry allowed

Regime rules:
  - trend_up:   close > EMA AND EMA_slope > 0
  - trend_down: close < EMA AND EMA_slope < 0
  - range:      all other cases

EMA slope: EMA[t] - EMA[t-1]

Default ema_period = 200 (center of stable parameter region from research).
"""

from typing import List, Union

import pandas as pd

DEFAULT_EMA_PERIOD = 200


def compute_ema(close_series: Union[pd.Series, List[float]], period: int = DEFAULT_EMA_PERIOD) -> pd.Series:
    """
    Compute EMA of close prices.

    Args:
        close_series: Close prices (pandas Series or list).
        period: EMA period (default 200).

    Returns:
        EMA series (same length as input).
    """
    if isinstance(close_series, list):
        close_series = pd.Series(close_series)
    return close_series.ewm(span=period, adjust=False).mean()


def compute_ema_slope(ema_series: pd.Series) -> pd.Series:
    """
    Compute EMA slope: EMA[t] - EMA[t-1].

    First value is NaN (no previous bar).
    """
    return ema_series.diff()


def classify_regime(close: float, ema: float, ema_slope: float) -> str:
    """
    Classify regime for a single bar.

    Args:
        close: Bar close price.
        ema: EMA value at this bar.
        ema_slope: EMA[t] - EMA[t-1]. Use pd.NA/np.nan for first bar.

    Returns:
        "trend_up", "trend_down", or "range".
    """
    if pd.isna(ema_slope) or pd.isna(ema):
        return "range"
    if close > ema and ema_slope > 0:
        return "trend_up"
    if close < ema and ema_slope < 0:
        return "trend_down"
    return "range"


def is_long_allowed(regime: str) -> bool:
    """True if LONG entries are allowed (trend_up only)."""
    return regime == "trend_up"


def is_short_allowed(regime: str) -> bool:
    """True if SHORT entries are allowed (trend_down only)."""
    return regime == "trend_down"


def compute_regime_series(
    closes: Union[pd.Series, List[float]],
    ema_period: int = DEFAULT_EMA_PERIOD,
) -> pd.DataFrame:
    """
    Compute regime for the full close series at once.

    Returns a DataFrame with columns: close, ema, ema_slope, regime.
    Bars without enough history for a meaningful EMA are classified as "range".

    Args:
        closes: Close prices (pandas Series or list).
        ema_period: EMA period.

    Returns:
        DataFrame with index aligned to input, columns: close, ema, ema_slope, regime.
    """
    if isinstance(closes, list):
        closes = pd.Series(closes)
    ema = compute_ema(closes, ema_period)
    ema_slope = compute_ema_slope(ema)
    min_history = ema_period
    regimes = []
    for i in range(len(closes)):
        if i < 1 or (i + 1) < min_history:
            regimes.append("range")
        else:
            c = float(closes.iloc[i])
            e = float(ema.iloc[i])
            s = float(ema_slope.iloc[i])
            regimes.append(classify_regime(c, e, s))
    return pd.DataFrame(
        {"close": closes, "ema": ema, "ema_slope": ema_slope, "regime": regimes},
        index=closes.index,
    )


def get_regime_at_bar(
    closes: Union[pd.Series, List[float]],
    bar_index: int,
    ema_period: int = DEFAULT_EMA_PERIOD,
) -> str:
    """
    Get regime at a specific bar index.

    Convenience function: uses compute_regime_series on closes[0:bar_index+1],
    returns regime at bar_index. Returns "range" if insufficient history.

    Args:
        closes: Close prices (full history up to and including current bar).
        bar_index: Index of bar to classify (0-based).
        ema_period: EMA period.

    Returns:
        "trend_up", "trend_down", or "range".
        Returns "range" if bar_index < 1 (no slope) or insufficient data for EMA.
    """
    if isinstance(closes, list):
        closes = pd.Series(closes)
    slice_closes = closes.iloc[: bar_index + 1]
    df = compute_regime_series(slice_closes, ema_period)
    return str(df.iloc[-1]["regime"])


def get_regime_from_bars(bars: List, bar_index: int, ema_period: int = DEFAULT_EMA_PERIOD) -> str:
    """
    Get regime at bar_index from a list of Bar-like objects (must have .close).

    For use with bars_15m in main.py / QuantConnect.

    Args:
        bars: List of Bar objects (or any with .close attribute).
        bar_index: Index of bar to classify.
        ema_period: EMA period.

    Returns:
        "trend_up", "trend_down", or "range".
    """
    closes = [float(b.close) for b in bars]
    return get_regime_at_bar(closes, bar_index, ema_period)
