"""
RSI Momentum Confirmation Filter (V2).

This module provides two layers:
  - RSI computation using Wilder's smoothing (RMA)
  - RSI-based momentum gating for entry confirmation

It is designed to fit the rest of the project:
  - compute_* functions return indicator series
  - get_* helpers query a specific bar
  - RSIMomentumFilter returns a pass/fail decision with diagnostics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import pandas as pd

from atr_module.atr_module import wilder_smooth, wilder_smooth_step
from entry_exit_rules.entry_exit import PositionDirection

DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_LONG_THRESHOLD = 55.0
DEFAULT_RSI_SHORT_THRESHOLD = 45.0
DEFAULT_RSI_CROSS_LEVEL_LONG = 50.0
DEFAULT_RSI_CROSS_LEVEL_SHORT = 50.0
DEFAULT_RSI_BULL_SUPPORT = 45.0
DEFAULT_RSI_BEAR_RESISTANCE = 55.0


class RSIMomentumMode(str, Enum):
    THRESHOLD = "THRESHOLD"
    CROSS = "CROSS"
    TREND_RANGE = "TREND_RANGE"


@dataclass(frozen=True)
class RSIFilterDecision:
    allowed: bool
    reason: str
    mode: RSIMomentumMode
    direction: PositionDirection
    rsi_value: Optional[float]
    previous_rsi_value: Optional[float]
    threshold: Optional[float] = None
    regime: Optional[str] = None


@dataclass
class RSIEngine:
    """
    Incremental RSI engine.

    The engine emits None until it has seen ``length + 1`` closes, then updates
    in O(1) per new close using Wilder's recurrence.
    """

    length: int = DEFAULT_RSI_PERIOD
    prev_close: Optional[float] = None
    avg_gain: Optional[float] = None
    avg_loss: Optional[float] = None
    seed_gains: List[float] = field(default_factory=list)
    seed_losses: List[float] = field(default_factory=list)
    current_rsi: Optional[float] = None
    previous_rsi: Optional[float] = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")

    def reset(self) -> None:
        self.prev_close = None
        self.avg_gain = None
        self.avg_loss = None
        self.seed_gains.clear()
        self.seed_losses.clear()
        self.current_rsi = None
        self.previous_rsi = None

    def update(self, new_close: float) -> Optional[float]:
        if not math.isfinite(new_close):
            raise ValueError(f"new_close must be finite, got {new_close}")

        if self.prev_close is None:
            self.prev_close = float(new_close)
            self.previous_rsi = self.current_rsi
            self.current_rsi = None
            return None

        delta = float(new_close) - self.prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self.prev_close = float(new_close)

        self.previous_rsi = self.current_rsi

        if self.avg_gain is None or self.avg_loss is None:
            self.seed_gains.append(gain)
            self.seed_losses.append(loss)

            if len(self.seed_gains) < self.length:
                self.current_rsi = None
                return None

            self.avg_gain = sum(self.seed_gains) / self.length
            self.avg_loss = sum(self.seed_losses) / self.length
        else:
            self.avg_gain = wilder_smooth_step(self.avg_gain, gain, self.length)
            self.avg_loss = wilder_smooth_step(self.avg_loss, loss, self.length)

        self.current_rsi = _compute_rsi_value(self.avg_gain, self.avg_loss)
        return self.current_rsi


def _coerce_series(values: Union[pd.Series, List[float]]) -> pd.Series:
    if isinstance(values, list):
        return pd.Series(values, dtype=float)
    return values.astype(float).copy()


def _compute_rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_rsi_from_averages(avg_gain: pd.Series, avg_loss: pd.Series) -> pd.Series:
    rsi = pd.Series(float("nan"), index=avg_gain.index, dtype=float)
    ready_mask = avg_gain.notna() & avg_loss.notna()

    zero_loss_mask = ready_mask & (avg_loss == 0)
    zero_gain_mask = ready_mask & (avg_gain == 0) & ~zero_loss_mask
    regular_mask = ready_mask & ~zero_loss_mask & ~zero_gain_mask

    rsi.loc[zero_loss_mask] = 100.0
    rsi.loc[zero_gain_mask] = 0.0

    rs = avg_gain.loc[regular_mask] / avg_loss.loc[regular_mask]
    rsi.loc[regular_mask] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_rsi(
    closes: Union[pd.Series, List[float]],
    period: int = DEFAULT_RSI_PERIOD,
) -> pd.Series:
    """
    Compute RSI from close prices using Wilder's smoothing.

    The first meaningful RSI value is emitted only after ``period + 1`` closes.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _coerce_series(closes)
    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = pd.Series(float("nan"), index=closes.index, dtype=float)
    avg_loss = pd.Series(float("nan"), index=closes.index, dtype=float)

    if len(closes) > 1:
        gain_rma = wilder_smooth(gains.iloc[1:].tolist(), period)
        loss_rma = wilder_smooth(losses.iloc[1:].tolist(), period)
        avg_gain.iloc[1:] = gain_rma.to_numpy()
        avg_loss.iloc[1:] = loss_rma.to_numpy()

    rsi = _compute_rsi_from_averages(avg_gain, avg_loss)
    return rsi.replace([float("inf"), float("-inf")], float("nan"))


def compute_rsi_series(
    closes: Union[pd.Series, List[float]],
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> pd.DataFrame:
    """
    Compute RSI for the full close series at once.

    Returns a DataFrame with columns: close, rsi.
    Bars without enough history have NaN in rsi.
    """
    closes = _coerce_series(closes)
    rsi = compute_rsi(closes, rsi_period)

    min_history = rsi_period + 1
    rsi_safe = rsi.copy()
    rsi_safe.iloc[: min_history - 1] = float("nan")

    return pd.DataFrame({"close": closes, "rsi": rsi_safe}, index=closes.index)


def get_rsi_at_bar(
    closes: Union[pd.Series, List[float]],
    bar_index: int,
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> float:
    """
    Get RSI value at a specific bar index.

    Returns NaN if there is not enough history for a seeded RSI.
    """
    closes = _coerce_series(closes)

    n = len(closes)
    if bar_index < 0:
        raise IndexError(f"bar_index must be >= 0, got {bar_index}")
    if bar_index >= n:
        raise IndexError(f"bar_index {bar_index} out of range for series of length {n}")

    min_history = rsi_period + 1
    if (bar_index + 1) < min_history:
        return float("nan")

    slice_closes = closes.iloc[: bar_index + 1]
    df = compute_rsi_series(slice_closes, rsi_period)
    return float(df.iloc[-1]["rsi"])


def get_rsi_from_bars(
    bars: List,
    bar_index: int,
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> float:
    """
    Get RSI value at bar_index from a list of Bar-like objects.

    Bar-like objects must have a .close attribute.
    """
    n = len(bars)
    if n == 0:
        raise IndexError("bars list is empty")
    if bar_index < 0:
        raise IndexError(f"bar_index must be >= 0, got {bar_index}")
    if bar_index >= n:
        raise IndexError(f"bar_index {bar_index} out of range for bars list of length {n}")

    closes = [float(b.close) for b in bars]
    return get_rsi_at_bar(closes, bar_index, rsi_period)


def crossed_above(previous_value: Optional[float], current_value: Optional[float], level: float) -> bool:
    if previous_value is None or current_value is None:
        return False
    return previous_value < level <= current_value


def crossed_below(previous_value: Optional[float], current_value: Optional[float], level: float) -> bool:
    if previous_value is None or current_value is None:
        return False
    return previous_value > level >= current_value


class RSIMomentumFilter:
    """
    Direction-aware RSI gate for pre-entry momentum confirmation.
    """

    def __init__(
        self,
        *,
        mode: Union[RSIMomentumMode, str] = RSIMomentumMode.THRESHOLD,
        long_threshold: float = DEFAULT_RSI_LONG_THRESHOLD,
        short_threshold: float = DEFAULT_RSI_SHORT_THRESHOLD,
        cross_level_long: float = DEFAULT_RSI_CROSS_LEVEL_LONG,
        cross_level_short: float = DEFAULT_RSI_CROSS_LEVEL_SHORT,
        bull_support: float = DEFAULT_RSI_BULL_SUPPORT,
        bear_resistance: float = DEFAULT_RSI_BEAR_RESISTANCE,
    ):
        self.mode = RSIMomentumMode(mode)
        self.long_threshold = self._validate_level(long_threshold, "long_threshold")
        self.short_threshold = self._validate_level(short_threshold, "short_threshold")
        self.cross_level_long = self._validate_level(cross_level_long, "cross_level_long")
        self.cross_level_short = self._validate_level(cross_level_short, "cross_level_short")
        self.bull_support = self._validate_level(bull_support, "bull_support")
        self.bear_resistance = self._validate_level(bear_resistance, "bear_resistance")

    def allow_entry(
        self,
        *,
        direction: PositionDirection,
        rsi_now: Optional[float],
        rsi_prev: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> RSIFilterDecision:
        rsi_value = self._normalize_value(rsi_now)
        previous_rsi_value = self._normalize_value(rsi_prev)

        if rsi_value is None:
            return RSIFilterDecision(
                allowed=False,
                reason="RSI_NOT_READY",
                mode=self.mode,
                direction=direction,
                rsi_value=None,
                previous_rsi_value=previous_rsi_value,
            )

        if self.mode == RSIMomentumMode.THRESHOLD:
            return self._threshold_decision(direction, rsi_value, previous_rsi_value)

        if self.mode == RSIMomentumMode.CROSS:
            return self._cross_decision(direction, rsi_value, previous_rsi_value)

        return self._trend_range_decision(direction, rsi_value, previous_rsi_value, regime)

    @staticmethod
    def _validate_level(value: float, name: str) -> float:
        if value < 0 or value > 100:
            raise ValueError(f"{name} must be within [0, 100], got {value}")
        return float(value)

    @staticmethod
    def _normalize_value(value: Optional[float]) -> Optional[float]:
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    def _threshold_decision(
        self,
        direction: PositionDirection,
        rsi_value: float,
        previous_rsi_value: Optional[float],
    ) -> RSIFilterDecision:
        if direction == PositionDirection.LONG:
            threshold = self.long_threshold
            allowed = rsi_value >= threshold
            reason = "RSI_LONG_THRESHOLD_PASS" if allowed else "RSI_LONG_THRESHOLD_BLOCK"
        else:
            threshold = self.short_threshold
            allowed = rsi_value <= threshold
            reason = "RSI_SHORT_THRESHOLD_PASS" if allowed else "RSI_SHORT_THRESHOLD_BLOCK"

        return RSIFilterDecision(
            allowed=allowed,
            reason=reason,
            mode=self.mode,
            direction=direction,
            rsi_value=rsi_value,
            previous_rsi_value=previous_rsi_value,
            threshold=threshold,
        )

    def _cross_decision(
        self,
        direction: PositionDirection,
        rsi_value: float,
        previous_rsi_value: Optional[float],
    ) -> RSIFilterDecision:
        if previous_rsi_value is None:
            threshold = self.cross_level_long if direction == PositionDirection.LONG else self.cross_level_short
            return RSIFilterDecision(
                allowed=False,
                reason="RSI_PREV_NOT_READY",
                mode=self.mode,
                direction=direction,
                rsi_value=rsi_value,
                previous_rsi_value=None,
                threshold=threshold,
            )

        if direction == PositionDirection.LONG:
            threshold = self.cross_level_long
            allowed = crossed_above(previous_rsi_value, rsi_value, threshold)
            reason = "RSI_LONG_CROSS_PASS" if allowed else "RSI_LONG_CROSS_BLOCK"
        else:
            threshold = self.cross_level_short
            allowed = crossed_below(previous_rsi_value, rsi_value, threshold)
            reason = "RSI_SHORT_CROSS_PASS" if allowed else "RSI_SHORT_CROSS_BLOCK"

        return RSIFilterDecision(
            allowed=allowed,
            reason=reason,
            mode=self.mode,
            direction=direction,
            rsi_value=rsi_value,
            previous_rsi_value=previous_rsi_value,
            threshold=threshold,
        )

    def _trend_range_decision(
        self,
        direction: PositionDirection,
        rsi_value: float,
        previous_rsi_value: Optional[float],
        regime: Optional[str],
    ) -> RSIFilterDecision:
        if regime is None:
            return RSIFilterDecision(
                allowed=False,
                reason="RSI_REGIME_NOT_READY",
                mode=self.mode,
                direction=direction,
                rsi_value=rsi_value,
                previous_rsi_value=previous_rsi_value,
            )

        if direction == PositionDirection.LONG:
            threshold = self.bull_support
            if regime != "trend_up":
                return RSIFilterDecision(
                    allowed=False,
                    reason="RSI_REGIME_NOT_BULL",
                    mode=self.mode,
                    direction=direction,
                    rsi_value=rsi_value,
                    previous_rsi_value=previous_rsi_value,
                    threshold=threshold,
                    regime=regime,
                )
            allowed = rsi_value >= threshold
            reason = "RSI_LONG_TREND_RANGE_PASS" if allowed else "RSI_LONG_TREND_RANGE_BLOCK"
        else:
            threshold = self.bear_resistance
            if regime != "trend_down":
                return RSIFilterDecision(
                    allowed=False,
                    reason="RSI_REGIME_NOT_BEAR",
                    mode=self.mode,
                    direction=direction,
                    rsi_value=rsi_value,
                    previous_rsi_value=previous_rsi_value,
                    threshold=threshold,
                    regime=regime,
                )
            allowed = rsi_value <= threshold
            reason = "RSI_SHORT_TREND_RANGE_PASS" if allowed else "RSI_SHORT_TREND_RANGE_BLOCK"

        return RSIFilterDecision(
            allowed=allowed,
            reason=reason,
            mode=self.mode,
            direction=direction,
            rsi_value=rsi_value,
            previous_rsi_value=previous_rsi_value,
            threshold=threshold,
            regime=regime,
        )
