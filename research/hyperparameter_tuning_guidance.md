# Hyperparameter Tuning Guidance

## Purpose

This note separates project parameters into three buckets:

1. `Tune / research`
2. `Freeze / leave as policy or canonical default`
3. `Inactive for now / do not spend tuning budget until wired into production`

The goal is practical governance:

- avoid tuning everything
- avoid overfitting operational controls
- focus research time on parameters that actually change signal quality and trade economics

Important caveat:

- Before doing serious parameter research, the entry timing model in the strategy should be finalized. The current live path still follows the legacy V1 signal/entry convention from the BOS logic, so alpha-parameter research should not be treated as final until execution timing is made fully realistic.

Relevant code references:

- active runtime config: `code/main.py`
- BOS / TP / trailing rules: `code/entry_exit_rules/entry_exit.py`
- risk policy: `code/risk_management/risk.py`
- stop-loss modes: `code/stop_loss/stop_loss.py`
- swing detection: `code/swing_high_low_detection/swing_high_low_detection.py`
- RSI filter: `code/RSI/momentum_confirmation_rsi.py`
- EMA regime filter: `code/ma_regime_filter/ma_regime_filter.py`
- ADX filter: `code/adx_filter/adx_filter.py`
- ATR module: `code/atr_module/atr_module.py`

## 1. Tune / Research

These parameters directly affect either:

- which trades exist at all
- where the stop sits
- how reward is harvested
- how aggressively momentum confirmation blocks entries

These are the parameters most likely to change strategy edge.

### A. Signal construction

Current runtime values in `code/main.py`:

- `N_candidates = [5, 10, 20]`
- `N_confirmation = 3`
- `min_move_threshold = 0.0`
- `min_bars_between_swings = 3`

Why tune:

- These define what counts as a swing.
- They directly change BOS frequency, latency, and structure quality.
- This is core strategy logic, not a harmless implementation detail.

Recommendation:

- Yes, research these.
- Do walk-forward, not single in-sample optimization.
- Optimize for robustness and trade quality, not just total return.

### B. Stop-loss architecture

Current runtime values in `code/main.py`:

- `sl_mode = "fixed"`
- `fixed_pct = 0.0100`
- `buffer_pct = 0.000`

Additional available stop parameters in `code/stop_loss/stop_loss.py`:

- `k_sl = 2.0`
- `atr_period = 14`
- modes: `fixed`, `structural`, `bos`, `atr`

Why tune:

- Stop location determines both risk per trade and whether the trade survives normal noise.
- This is one of the strongest PnL drivers in a breakout system.
- A fixed 1% stop is easy to explain, but not automatically correct for ETHUSD 15m.

Recommendation:

- Yes, research stop mode first.
- Compare `fixed` vs `structural` vs `bos` vs `atr`.
- If `fixed` remains in play, tune `fixed_pct`.
- If `atr` remains in play, then tune `k_sl`; keep `atr_period=14` initially fixed.

### C. Take-profit and trade management

Current runtime values in `code/main.py`:

- `tp_mode = RR_BASED`
- `tp_mult = 3.0`
- `partial_exit_pct = 0.5`
- `partial_exit_at_r = 1.0`
- `k_trail = 2.0`
- `atr_period_trailing = 14`

Why tune:

- These define payoff shape.
- They determine whether the strategy behaves like:
  - a quick scalp
  - a classic breakout runner
  - a hybrid system
- Management parameters often matter as much as entry filters.

Recommendation:

- Yes, research these.
- Suggested sequence:
  1. validate `tp_mode`
  2. validate `tp_mult`
  3. validate whether partial exits help at all
  4. only then fine-tune `k_trail`
- Keep `atr_period_trailing=14` fixed at first.

### D. RSI momentum confirmation

Current runtime values in `code/main.py`:

- `rsi_enabled = True`
- `rsi_period = 14`
- `rsi_mode = THRESHOLD`
- `rsi_long_threshold = 55`
- `rsi_short_threshold = 45`

Available but currently inactive alternatives:

- `CROSS` with `50 / 50`
- `TREND_RANGE` with `bull_support = 45`, `bear_resistance = 55`

Why tune:

- RSI is an entry gate, so it changes trade selection rather than trade management.
- Thresholds like `55 / 45` are reasonable priors, but still strategy-specific.
- The correct setting depends on whether your BOS entries need mild confirmation or only very strong impulse.

Recommendation:

- Yes, research RSI mode and thresholds.
- Start with:
  - `THRESHOLD: 50/50`
  - `THRESHOLD: 55/45`
  - `CROSS: 50/50`
- Only test `TREND_RANGE` after regime timing is production-ready.

## 2. Freeze / Do Not Optimize Aggressively

These parameters are better treated as policy controls, conservative assumptions, or canonical indicator defaults.

### A. Risk policy

Current runtime values in `code/main.py` / `code/risk_management/risk.py`:

- `risk_pct = 0.01`
- `max_position_size = None`
- `min_stop_distance = None`
- `max_leverage = None`
- `use_buying_power_cap = False`

Why freeze:

- `risk_pct` is portfolio governance, not alpha.
- Optimizing `risk_pct` for return is usually a mistake; it just changes leverage on the same signal.
- This should be chosen based on risk tolerance and drawdown limits.

Recommendation:

- Do not optimize `risk_pct` for strategy edge.
- Keep `1%` as a defendable conservative default unless the risk committee wants another budget.

### B. Same-bar ambiguity rule

Current runtime value:

- `same_bar_rule = WORST_CASE`

Why freeze:

- This is a backtest realism assumption, not a signal parameter.
- `WORST_CASE` is conservative and easier to defend to management.

Recommendation:

- Leave as is unless lower-timeframe execution modeling is added.

### C. Canonical indicator periods

Defaults in code:

- `ATR period = 14`
- `ADX period = 14`
- `RSI period = 14`

Why freeze initially:

- These are standard Wilder defaults.
- They are widely used and easy to defend.
- They should not be the first tuning target.

Recommendation:

- Leave these fixed during first-pass research.
- Only revisit if later evidence shows strong sensitivity.

### D. Technical / operational parameters

Current runtime values:

- `qty_decimals = 6`
- `cooldown_bars = 0`
- `max_trades_per_day = None`

Why freeze:

- `qty_decimals` is implementation, not alpha.
- `cooldown_bars` and `max_trades_per_day` are operational guardrails.

Recommendation:

- Do not tune unless there is a specific execution or compliance reason.

## 3. Inactive for Now / Do Not Spend Research Budget Yet

These parameters exist in modules, but are not currently first-class production filters in the active `main.py` trading pipeline.

### A. EMA regime filter

Default:

- `ema_period = 200`

Evidence:

- there is an exported walk-forward report in `research/regime_filter/ema_sweep_report.md`
- the report supports `200` as a stable center-of-range choice, not as a proven alpha generator

Important note:

- the current runtime path does not use EMA regime as a general entry filter
- it is only consulted by RSI if RSI is switched to `TREND_RANGE`, which is not the active mode

Recommendation:

- Do not spend more tuning time here now.
- If EMA is reintroduced as a mandatory production filter, then reuse the existing walk-forward framework and revisit.

### B. ADX filter

Defaults:

- `adx_period = 14`
- `adx_threshold = 25`

Important note:

- ADX is implemented, but not currently active in the live `main.py` entry path.

Recommendation:

- Do not tune ADX until it is actually wired into production decisions.

## 4. Executive Summary

### High-priority research

- swing/BOS parameters
- stop-loss mode and distance
- TP / partial / trailing management
- RSI gate mode and thresholds

### Leave mostly fixed

- `risk_pct = 1%`
- `same_bar_rule = WORST_CASE`
- canonical indicator periods like `14`
- technical settings such as quantity precision

### Do not tune yet

- EMA regime parameters
- ADX parameters

Reason:

- they are either not active in production or not the current bottleneck

## 5. Recommended Research Order

1. Finalize realistic execution timing.
2. Research BOS swing parameters.
3. Research stop-loss family and distance.
4. Research TP / partial / trailing logic.
5. Research RSI gate.
6. Only after that, revisit dormant filters like EMA / ADX if they are promoted into the live path.

## 6. Management-Friendly Positioning

The defensible message is:

- not all parameters are equal
- some are business/risk policy and should not be optimized
- some are standard market defaults and are acceptable as priors
- only the parameters that define signal quality and trade economics should absorb research budget

That is the correct way to avoid both under-research and overfitting.
