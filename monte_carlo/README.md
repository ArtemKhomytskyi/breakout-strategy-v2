# Monte Carlo Risk Module

A pre-trade risk filter that sits between the V2 BOS signal generator and live execution.
Before any trade is placed, this module simulates thousands of possible futures for that
specific trade and asks: *given what the market looks like right now, what does the distribution
of outcomes look like?* The answer feeds an ACCEPT / REDUCE / REJECT decision.

```
V2 signal → TradeCandidate + MarketState → Monte Carlo engine → ACCEPT / REDUCE / REJECT
```

---

## What it does

1. **Estimates current volatility** (`stats.py`) from the most recent 20 bars of close prices.
2. **Generates N price paths** (`path_generator.py`) — either GBM (parametric) or Bootstrap
   (resample from actual historical returns).
3. **Replays the trade** (`trade_replay.py`) on each path, applying the exact V2 exit rules:
   SL, partial take-profit, ATR-based trailing stop, max holding bars.
4. **Aggregates risk metrics** (`metrics.py`): prob_loss, VaR, CVaR, Kelly fraction, Sharpe,
   profit factor, MAE/MFE, skewness.
5. **Makes a decision** (`decision.py`): compares metrics against configurable thresholds and
   returns ACCEPT (full size), REDUCE (half size by default), or REJECT (skip trade).

All 5 steps run every time `run_monte_carlo_analysis()` is called from `engine.py`.

---

## Files at a glance

| File | Purpose |
|---|---|
| `engine.py` | Top-level API — call this from V2 code |
| `config.py` | `MCConfig`: simulation settings (num_simulations, path_method, …) |
| `decision.py` | `DecisionConfig` thresholds + `make_trade_decision()` |
| `stats.py` | Compute sigma and drift from recent close prices |
| `path_generator.py` | GBM and Bootstrap path generation |
| `trade_replay.py` | Single-path trade replay (SL / partial TP / trailing / MAX_BARS) |
| `metrics.py` | Aggregate N simulation results into `RiskMetrics` |
| `optimizer.py` | Walk-forward + Optuna threshold optimisation (Stage F) |
| `market_state.py` | `MarketState` dataclass (input) |
| `trade_candidate.py` | `TradeCandidate` dataclass (input) |
| `main.py` | Standalone diagnostic runner — **start here** |

---

## Quick start

### 1. Run the diagnostic

No live data, no QuantConnect, no V2 strategy needed:

```bash
cd monte_carlo
python main.py
```

Runs 6 labelled scenarios (good LONG, high-RR rejection, SHORT, passive vs active,
GBM vs Bootstrap, crash regime) and prints all metrics to the terminal.

### 2. Run the test suite

```bash
# From the breakout-strategy-v2/ root
cd ..
pytest tests/MC_test_engine.py tests/MC_test_metrics.py \
       tests/MC_test_decision.py tests/MC_test_trade_replay.py -v
```

All MC tests are prefixed `MC_test_`. Run `pytest tests/ -v` for the full suite.

### 3. Call from V2 code

```python
from monte_carlo.engine import run_monte_carlo_analysis
from monte_carlo.trade_candidate import TradeCandidate
from monte_carlo.market_state import MarketState

candidate = TradeCandidate(
    direction="LONG",
    entry_price=current_price,
    stop_loss=sl_price,
    partial_tp_price=partial_tp_price,
    partial_close_fraction=0.5,
    trailing_mode="ATR_BASED",
    atr=current_atr,
    trailing_atr_multiple=2.0,
    max_holding_bars=50,
    planned_size=quantity,
    risk_pct=0.01,
    regime="NEUTRAL",
)

market_state = MarketState(
    recent_close_prices=recent_closes[-30:],  # at least rolling_window+1 bars
    recent_returns=recent_log_returns[-30:],
    atr=current_atr,
    sigma=0.0,        # engine recomputes from recent_close_prices
    drift=0.0,
    regime="NEUTRAL",
    timestamp=str(current_time),
)

decision = run_monte_carlo_analysis(
    candidate,
    market_state,
    passive_mode=True,   # ALWAYS start True — see go-live section below
)

if decision.action == "REJECT":
    skip_trade()
elif decision.action == "REDUCE":
    quantity *= decision.size_factor
```

---

## What data it uses

### Inputs (per trade, at signal time)

**`TradeCandidate`** — everything V2 already knows about the trade:

| Field | What it is |
|---|---|
| `direction` | `'LONG'` or `'SHORT'` |
| `entry_price` | Planned entry price — must equal the current market price (S0) |
| `stop_loss` | Absolute SL price |
| `partial_tp_price` | Price to trigger the partial exit |
| `partial_close_fraction` | Fraction to close at partial TP (e.g. 0.5 = 50%) |
| `trailing_mode` | `'ATR_BASED'` or `'OFF'` |
| `atr` | Current ATR in price units |
| `trailing_atr_multiple` | Trail distance = atr × multiple |
| `max_holding_bars` | Forced exit if trade runs this long |
| `planned_size` | Position size |
| `risk_pct` | Risk per trade as a fraction of equity |
| `regime` | Market regime label (used for sigma multiplier) |

> **Critical**: `entry_price` must match `market_state.recent_close_prices[-1]`.
> `generate_paths()` starts every simulated path at the last close price (S0).
> If `entry_price` differs from S0, PnL calculations will be wrong.

**`MarketState`** — a snapshot of current market conditions:

| Field | What it is | Minimum |
|---|---|---|
| `recent_close_prices` | Last N closing prices | `rolling_window + 1` bars (default: 21) |
| `recent_returns` | Log returns of those prices | Same length minus 1 |
| `atr` | Current ATR | Any positive float |
| `sigma` | Set to 0.0 — engine recomputes | — |
| `drift` | Set to 0.0 — engine recomputes | — |
| `regime` | Regime label for sigma multiplier | String |
| `timestamp` | ISO timestamp for logging | String |

### No external data files

The module uses only the prices and returns you pass in. It does not read CSV files,
connect to an API, or require any database. The path generator creates synthetic future
prices from the statistics of your recent_close_prices.

The optimizer (`optimizer.py`) uses `HistoricalTrade` objects which you populate with
your backtest or live trade results — see the **Fine-tuning** section.

---

## Understanding the output

```
decision.action               # 'ACCEPT' | 'REDUCE' | 'REJECT'
decision.size_factor          # 1.0 | reduce_size_factor | 0.0
decision.recommended_size_factor  # Kelly-based suggestion (informational)
decision.reason               # Human-readable explanation of which gate fired
decision.prob_loss            # P(simulated pnl_r < 0)
decision.var_r                # 5th-percentile outcome in R multiples
decision.cvar_r               # Mean of worst 5% tail
decision.expected_pnl_r       # Mean simulated return in R
decision.kelly_fraction       # Optimal position fraction (Kelly criterion)
decision.profit_factor        # sum(wins) / sum(losses)
```

### What each metric means in practice

**`prob_loss`**: Fraction of simulated paths where the trade lost money.
In a zero-drift simulation, a symmetric trade (SL = TP distance in log space)
produces prob_loss ≈ 0.50. Higher than 0.65 triggers the default reject gate.
*Note: a high-RR trade (e.g. SL=3%, TP=7%) naturally has prob_loss ≈ 0.69
even with positive EV — the system is conservative about loss frequency.*

**`var_r`**: The worst 5th-percentile outcome across all simulations, expressed
in R multiples. If `var_r = -1.0`, the worst 5% of simulations all hit the SL.
Since SL exits always cost exactly -1R, `var_r` is often -1.0R for SL-dominant
trades. It becomes more informative when trailing stops produce partial losses.

**`kelly_fraction`**: The Kelly criterion's optimal position size.
`f* = (win_rate × avg_win_r − loss_rate × avg_loss_r) / avg_win_r`.
A kelly_fraction below 0.05 means the simulation sees almost no positive
expectancy — the default gate rejects the trade.

**`recommended_size_factor`**: The Kelly fraction capped to [0, 1]. This is the
theoretically optimal fraction of your risk budget to deploy. Even on ACCEPT
decisions, the Kelly recommendation may suggest using less than full size.

---

## Go-live sequence (passive mode → active mode)

### Step 1: Start in passive mode (default)

`passive_mode=True` (the default) runs the full pipeline but **always returns ACCEPT**.
The real decision is computed and logged but not applied. This creates an audit trail
without affecting execution.

The INFO log line looks like:
```
MC | dir=LONG | sims=10000 | prob_loss=0.512 | ... | action=ACCEPT [PASSIVE]
MC PASSIVE — would have REJECT: prob_loss=0.668 >= reject threshold=0.650.
```

### Step 2: Collect 30-50 real trade outcomes

Run the V2 strategy live for at least 30-50 trades with `passive_mode=True`.
For each trade, record the simulated `prob_loss` from the log and the actual outcome.

### Step 3: Validate

Compare simulated metrics to actual outcomes:
- Simulated `prob_loss` should correlate with actual win/loss rate
- Simulated `var_r` should be comparable to actual worst outcomes
- Simulated `expected_pnl_r` should be in the right ballpark

If the correlation is reasonable, the model is calibrated for your strategy.
If it diverges significantly, review `trade_replay.py` assumptions (close-only
approximation, ATR trailing multiple, partial fraction).

### Step 4: Switch to active mode

```python
decision = run_monte_carlo_analysis(candidate, market_state, passive_mode=False)
```

### Step 5: Run the optimizer (once you have 50+ trades)

```python
from monte_carlo.optimizer import run_walk_forward_optimization, load_trades_from_csv

trades = load_trades_from_csv("path/to/your/historical_trades.csv")
best_config, report = run_walk_forward_optimization(trades, n_trials=200, n_folds=5)
print(report)
```

---

## Fine-tuning parameters

### Primary levers — DecisionConfig (thresholds)

These control when trades are rejected or reduced. Conservative defaults suitable for go-live:

```python
from monte_carlo.decision import DecisionConfig

config = DecisionConfig(
    reject_prob_loss=0.65,   # REJECT if 65%+ of sims lose. Lower = more rejections.
    reduce_prob_loss=0.55,   # REDUCE if 55-65% of sims lose.
    reject_var_r=-3.0,       # REJECT if 5th-pct outcome is worse than -3R.
    reduce_var_r=-2.0,       # REDUCE if 5th-pct outcome is between -2R and -3R.
    min_expected_pnl_r=-0.5, # REJECT if mean simulated outcome is below -0.5R.
    min_kelly_fraction=0.05, # REJECT if Kelly < 5% (near-zero edge).
    reduce_size_factor=0.5,  # Position multiplier on REDUCE (0.5 = half size).
)
```

**Tuning direction:**
- Getting too many REJECT on trades that actually win → loosen `reject_prob_loss` (e.g. 0.70)
  or tighten `min_kelly_fraction` (e.g. 0.02)
- Getting too many false passes → tighten `reject_prob_loss` (e.g. 0.60) or lower `reject_var_r`
- Use the optimizer to find the thresholds that maximize out-of-sample Sharpe

### Secondary levers — MCConfig (simulation settings)

```python
from monte_carlo.config import MCConfig

config = MCConfig(
    num_simulations=10_000,  # More sims = stable metrics but slower. 1000 for testing.
    horizon_bars=500,        # Max bars to simulate per path. Should exceed max_holding_bars.
    drift_mode="zero",       # 'zero' (conservative) or 'historical' (uses recent drift).
    rolling_window=20,       # Bars used to estimate sigma. Shorter = more reactive.
    path_method="GBM",       # 'GBM' (fast, normal tails) or 'BOOTSTRAP' (fat tails).
    bootstrap_lookback=252,  # How many recent bars to resample from in BOOTSTRAP mode.
    regime_sigma_multipliers={"CRASH": 1.5, "HIGH_VOL": 1.25},  # Widen tails in bad regimes.
    var_confidence=0.95,     # 95% confidence → 5th-percentile VaR.
    random_seed=None,        # Set an integer for reproducibility, None for fresh each run.
)
```

**Tuning direction:**
- VaR estimates jumping around between runs → increase `num_simulations` (10_000 minimum live)
- Sigma feels stale or too smooth → decrease `rolling_window` (try 10-15)
- Strategy runs on high-volatility instruments (crypto, small cap) → use `path_method='BOOTSTRAP'`
- Sigma too low during volatile periods → add `regime_sigma_multipliers`

### Regime sigma multipliers

If your `MarketState.regime` is populated, you can automatically widen tails in dangerous periods:

```python
# In MCConfig:
regime_sigma_multipliers={
    "CRASH":    1.5,   # Sigma × 1.5 during crash conditions
    "HIGH_VOL": 1.25,  # Sigma × 1.25 during elevated vol
    "NEUTRAL":  1.0,   # No change (implicit default)
}
```

Set `market_state.regime` to match one of these keys at signal time. If the regime
is not in the dict, sigma is used unchanged.

### Path method: GBM vs Bootstrap

| | GBM | Bootstrap |
|---|---|---|
| Returns drawn from | Normal distribution | Actual historical returns (with replacement) |
| Captures fat tails | No | Yes |
| Captures volatility clustering | No | Partially (if cluster exists in sample) |
| Min history needed | None | ≥ 10 bars (falls back to GBM otherwise) |
| Speed | Slightly faster | Similar |
| Best for | Broad scenario generation, low data | Tail risk estimation, 60+ bars of history |

**Rule of thumb**: Use `BOOTSTRAP` when you have 60+ recent bars and care most about
worst-case tail outcomes (VaR, CVaR). Use `GBM` for initial testing or when history is sparse.

---

## Optimizer — walk-forward threshold tuning

Once you have 50+ historical trades with known outcomes, the optimizer finds
`DecisionConfig` thresholds that maximize out-of-sample Sharpe.

### Prepare your data

Create a CSV with these columns (see `optimizer.py:load_trades_from_csv` for full spec):

```
direction, entry_price, stop_loss, partial_tp_price, partial_close_fraction,
trailing_mode, atr, trailing_atr_multiple, max_holding_bars, planned_size, risk_pct,
regime, actual_pnl_r, actual_exit_reason, trade_timestamp,
recent_close_prices (JSON array), recent_returns (JSON array), sigma, drift
```

`actual_pnl_r` is the real outcome in R multiples. `recent_close_prices` is the
market state at signal time (not at exit time).

### Run the optimizer

```python
from monte_carlo.optimizer import load_trades_from_csv, run_walk_forward_optimization

trades = load_trades_from_csv("data/historical_trades.csv")

best_config, report = run_walk_forward_optimization(
    trades,
    n_trials=200,          # Optuna trials per fold. 100-200 is usually enough.
    n_folds=5,             # Walk-forward folds. Needs len(trades)/n_folds >= 15.
    min_trade_fraction=0.30,  # Reject configs that filter > 70% of trades.
)

print(f"Best validation Sharpe: {report['best_val_sharpe']:.3f}")
print(f"Best config: {best_config}")
```

### What the optimizer searches

| Parameter | Search range | Notes |
|---|---|---|
| `reject_prob_loss` | 0.55 – 0.85 | |
| `reduce_prob_loss` | 0.40 – (reject − 0.05) | Always below reject |
| `reject_var_r` | −5.0 – −1.5 | |
| `reduce_var_r` | (reject + 0.5) – −1.0 | Always less negative than reject |
| `min_kelly_fraction` | 0.0 – 0.15 | |
| `reduce_size_factor` | 0.25 – 0.75 | |

### Interpreting results

The `report` dict contains:
```python
report['fold_results']      # per-fold train Sharpe, val Sharpe, n_val_trades
report['best_val_sharpe']   # best out-of-sample Sharpe across all folds
report['n_total_trades']    # total trades used
```

If `best_val_sharpe` is significantly below the in-sample Sharpe, you are overfitting.
Use more data or fewer folds. If it is positive and stable across folds, the
optimized config generalizes.

---

## Dependencies

```
numpy       (path generation, statistics)
pandas      (stats.py rolling window)
optuna      (optimizer only — not required for the main engine)
pytest      (tests only)
```

Install:
```bash
pip install numpy pandas optuna pytest
```

Optuna is optional. If not installed, the optimizer tests are skipped automatically.

---

## Key design decisions

**Zero drift by default.** The engine simulates with `drift=0` even if the recent
market has trended. This is conservative: we are stress-testing the trade, not
predicting direction. Using historical drift could make optimistic trades look safer
than they are.

**Close-only simulation.** Trade replay checks SL and TP only at bar closes.
Real markets can breach SL intrabar without closing there. This means the simulated
SL hit rate is slightly lower than reality (undercounts intrabar wicks). The real
SL hit rate in live trading will be slightly higher than `prob_sl_hit` shows.

**Passive mode first.** The engine defaults to `passive_mode=True`. Never let the
MC filter control execution before validating that its risk estimates correlate with
real outcomes. At least 30-50 live trades in passive mode is the recommended minimum.

**Conservative thresholds.** The defaults reject trades that 65%+ of simulations
lose, regardless of EV. A 2.3:1 RR trade naturally has ~69% of paths hit SL in a
zero-drift simulation — this would be rejected by default. The optimizer is the right
tool to tune thresholds once you have real trade data showing that high-prob_loss /
high-EV trades are profitable in your specific strategy.
