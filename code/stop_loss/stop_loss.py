import math
from entry_exit_rules.entry_exit import Bar, PositionDirection, has_level
from atr_module.atr_module import get_atr_from_bars
from typing import List, Optional


class StopLossManager:
    """
    Stop loss manager.

    Computes SL on entry and checks its trigger on every bar.
    Supports three modes:
    - fixed: fixed percent from entry
    - structural: behind last swing level + buffer
    - bos: beyond the signal candle extreme (Break of Structure) + buffer

    Stop loss is fixed once at entry and then stays unchanged.
    """

    VALID_MODES = {"fixed", "structural", "bos", "atr"}

    def __init__(
        self,
        mode: str = "fixed",
        fixed_pct: float = 0.01,
        buffer_pct: float = 0.001,
        k_sl: float = 2.0,
        atr_period: int = 14,
    ):
        """
        Initialize stop loss manager.

        Args:
            mode: SL mode ("fixed" | "structural" | "bos" | "atr")
            fixed_pct: Percent of entry for fixed mode (e.g., 0.01 = 1%)
            buffer_pct: Buffer percent of entry for structural and bos
            k_sl: ATR multiplier for atr mode (LONG: SL = entry - k_sl*ATR, SHORT: SL = entry + k_sl*ATR)
            atr_period: ATR period for atr mode

        Raises:
            ValueError: If mode is invalid or parameters are non-positive
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: '{mode}'. Allowed: {', '.join(self.VALID_MODES)}"
            )
        if fixed_pct <= 0:
            raise ValueError(f"fixed_pct must be > 0, got {fixed_pct}")
        if buffer_pct < 0:
            raise ValueError(f"buffer_pct cannot be negative, got {buffer_pct}")
        if mode == "atr" and (k_sl <= 0 or atr_period <= 0):
            raise ValueError(f"atr mode: k_sl and atr_period must be > 0, got k_sl={k_sl}, atr_period={atr_period}")

        self.mode = mode
        self.fixed_pct = fixed_pct
        self.buffer_pct = buffer_pct
        self.k_sl = k_sl
        self.atr_period = atr_period
        self.reset()

    def reset(self) -> None:
        """
        Reset manager state.

        Call after a position is closed to prepare for the next entry.
        """
        self.direction: Optional[PositionDirection] = None
        self.entry_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.active = False

    def on_entry(
        self,
        direction: PositionDirection,
        entry_price: float,
        *,
        last_swing_high: Optional[float] = None,
        last_swing_low: Optional[float] = None,
        signal_bar: Optional[Bar] = None,
        bars: Optional[List[Bar]] = None,
        entry_candle_index: Optional[int] = None,
    ) -> float:
        """
        Compute and fix stop loss on position entry.

        Args:
            direction: Position direction (PositionDirection.LONG or .SHORT)
            entry_price: Position entry price
            last_swing_high: Last swing high price (used in structural for SHORT)
            last_swing_low: Last swing low price (used in structural for LONG)
            signal_bar: BOS candle t (closed at t, entry at open t+1).
                StopLossManager does not compute BOS — caller responsibility.
            bars: Full bar history for ATR mode (data up to and including entry bar only).
            entry_candle_index: Bar index of entry candle for ATR mode.

        Returns:
            float: Calculated stop loss price

        Raises:
            RuntimeError: If manager is already active (reset() not called)
            ValueError: If required parameters for the chosen mode are missing
            ValueError: If entry_price <= 0
            ValueError: If ATR is invalid in atr mode (refuse trade)
        """
        if self.active:
            raise RuntimeError(
                "StopLossManager already active! Call reset() before a new entry."
            )

        if entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {entry_price}")

        self.direction = direction
        self.entry_price = entry_price

        if self.mode == "fixed":
            self.stop_price = self._fixed_sl()

        elif self.mode == "atr":
            if bars is None or entry_candle_index is None:
                raise ValueError("For 'atr' mode you must provide bars and entry_candle_index.")
            atr = get_atr_from_bars(bars, entry_candle_index, self.atr_period)
            if not math.isfinite(atr) or atr <= 0:
                raise ValueError(
                    f"ATR invalid at entry bar (atr={atr}); cannot set stop. Refuse trade."
                )
            self.stop_price = self._atr_sl(atr)

        elif self.mode == "structural":
            if direction == PositionDirection.LONG:
                if not has_level(last_swing_low):
                    raise ValueError("For 'structural' LONG you must provide last_swing_low")
                assert last_swing_low is not None
                self.stop_price = self._structural_sl_long(last_swing_low)
            elif direction == PositionDirection.SHORT:
                if not has_level(last_swing_high):
                    raise ValueError("For 'structural' SHORT you must provide last_swing_high")
                assert last_swing_high is not None
                self.stop_price = self._structural_sl_short(last_swing_high)

        elif self.mode == "bos":
            if signal_bar is None:
                raise ValueError("For 'bos' you must provide signal_bar!")
            # bos: stop is placed beyond the signal candle extreme (t) with no lookahead for future swings.
            self.stop_price = self._bos_sl(signal_bar)

        if self.stop_price is None:
            raise ValueError(f"stop_price not computed (mode={self.mode}, direction={direction})")
        self._validate_stop_price(
            self.stop_price,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            signal_bar=signal_bar,
        )
        self.active = True
        return self.stop_price

    def should_exit(self, bar: Bar) -> bool:
        """
        Check if stop loss is hit on the current bar.

        Args:
            bar: Current candle (Bar)

        Returns:
            bool: True if price crosses stop loss, else False
        """
        if not self.active:
            return False

        if self.direction == PositionDirection.LONG:
            return bar.low <= self.stop_price  # type: ignore

        if self.direction == PositionDirection.SHORT:
            return bar.high >= self.stop_price  # type: ignore

        return False

    def _fixed_sl(self) -> float:
        """
        Compute stop loss for fixed mode.

        Returns:
            float: Stop loss price
        """
        if self.direction == PositionDirection.LONG:
            return self.entry_price * (1 - self.fixed_pct)

        return self.entry_price * (1 + self.fixed_pct)

    def _structural_sl_long(self, last_swing_low: float) -> float:
        """
        Compute stop loss for structural LONG (below last swing low).
        """
        buffer = self.entry_price * self.buffer_pct
        return last_swing_low - buffer

    def _structural_sl_short(self, last_swing_high: float) -> float:
        """
        Compute stop loss for structural SHORT (above last swing high).
        """
        buffer = self.entry_price * self.buffer_pct
        return last_swing_high + buffer

    def _atr_sl(self, atr: float) -> float:
        """
        Compute stop loss for atr mode: LONG SL = entry - k_sl*ATR, SHORT SL = entry + k_sl*ATR.
        """
        if self.direction == PositionDirection.LONG:
            return self.entry_price - self.k_sl * atr
        return self.entry_price + self.k_sl * atr

    def _bos_sl(self, signal_bar: Bar) -> float:
        """
        Compute stop loss for bos mode (beyond signal candle extreme).

        Args:
            signal_bar: Break of Structure signal candle

        Returns:
            float: Stop loss price with buffer
        """
        buffer = self.entry_price * self.buffer_pct

        if self.direction == PositionDirection.LONG:
            return signal_bar.low - buffer

        return signal_bar.high + buffer

    def _validate_stop_price(
        self,
        stop_price: Optional[float],
        *,
        last_swing_high: Optional[float],
        last_swing_low: Optional[float],
        signal_bar: Optional[Bar],
    ) -> None:
        """
        Unified validation of stop: validity and side relative to entry.
        """
        if self.entry_price is None:
            raise ValueError("entry_price is missing before stop validation")

        if stop_price is None or (isinstance(stop_price, float) and not math.isfinite(stop_price)):
            raise ValueError(
                f"Invalid stop_price (mode={self.mode}, dir={self.direction}, entry={self.entry_price}, stop={stop_price}, "
                f"last_swing_high={last_swing_high}, last_swing_low={last_swing_low})"
            )

        if stop_price <= 0:
            raise ValueError(
                f"stop_price must be > 0 (mode={self.mode}, dir={self.direction}, entry={self.entry_price}, stop={stop_price})"
            )

        if self.direction == PositionDirection.LONG and stop_price >= self.entry_price:
            raise ValueError(
                f"LONG stop must be below entry (mode={self.mode}, entry={self.entry_price}, stop={stop_price}, last_swing_low={last_swing_low})"
            )

        if self.direction == PositionDirection.SHORT and stop_price <= self.entry_price:
            raise ValueError(
                f"SHORT stop must be above entry (mode={self.mode}, entry={self.entry_price}, stop={stop_price}, last_swing_high={last_swing_high})"
            )
