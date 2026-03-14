from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from entry_exit_rules.entry_exit import PositionDirection


@dataclass(frozen=True)
class RiskConfig:
    """
    Risk configuration for sizing positions (percent-of-equity).

    Attributes:
        risk_pct: fraction of equity to risk per trade (e.g. 0.01 = 1%).
        max_position_size: optional hard cap on quantity (float for fractional).
        min_stop_distance: optional minimal stop distance; if stop is tighter, skip the trade.
        max_leverage: optional cap so notional <= equity * max_leverage.
        use_buying_power_cap: if True, cap risk_cash by buying_power_cash in size_position.
    """
    risk_pct: float
    max_position_size: Optional[float] = None
    min_stop_distance: Optional[float] = None
    max_leverage: Optional[float] = None
    use_buying_power_cap: bool = False


def size_position(
    *,
    direction: "PositionDirection",
    entry_price: float,
    sl_price: float,
    risk_config: RiskConfig,
    equity: float,
    buying_power_cash: Optional[float] = None,
    round_func: Callable[[float], float] = lambda x: x,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute position quantity from percent-of-equity risk and stop distance.

    risk_cash = equity * risk_pct
    stop_distance = abs(entry - sl)
    qty = risk_cash / stop_distance

    Supports fractional sizing (round_func), max_position_size, min_stop_distance, max_leverage.
    Hard failures -> ValueError (broken data).
    Soft refusals -> (None, reason).
    Returns (qty, refusal_reason).
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    if sl_price <= 0:
        raise ValueError(f"sl_price must be > 0, got {sl_price}")
    if equity <= 0:
        raise ValueError(f"equity must be > 0, got {equity}")
    if risk_config.risk_pct <= 0 or risk_config.risk_pct > 1:
        raise ValueError(f"risk_pct must be in (0, 1], got {risk_config.risk_pct}")

    stop_distance = abs(entry_price - sl_price)
    if stop_distance <= 0:
        raise ValueError("stop_distance must be > 0 (entry and SL cannot coincide).")

    if risk_config.min_stop_distance is not None and stop_distance < risk_config.min_stop_distance:
        return None, "stop_distance below min_stop_distance"

    risk_cash = equity * risk_config.risk_pct
    if risk_config.use_buying_power_cap and buying_power_cash is not None:
        risk_cash = min(risk_cash, buying_power_cash)

    if risk_cash <= 0:
        return None, "no risk budget available"

    raw_qty = risk_cash / stop_distance
    qty = round_func(raw_qty)

    if risk_config.max_position_size is not None:
        qty = min(qty, risk_config.max_position_size)

    if risk_config.max_leverage is not None and risk_config.max_leverage > 0:
        max_qty_by_leverage = (equity * risk_config.max_leverage) / entry_price
        qty = min(qty, max_qty_by_leverage)

    if qty <= 0:
        return None, "sized quantity is zero after caps"

    return float(qty), None
