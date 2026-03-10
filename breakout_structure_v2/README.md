# Breakout V2 Strategy

BOS (Break of Structure) breakout strategy with EMA regime filter research.

## Structure

```
code/
├── main.py                    # QuantConnect algorithm
├── entry_exit_rules/
├── ma_regime_filter/
├── risk_management/
├── stop_loss/
├── swing_high_low_detection/
└── research/
    ├── dataset_preparation/   # 15m dataset builder
    └── regime_filter_grid_search/  # EMA sweep research
data/
├── raw/
└── processed/
```

## Dataset Preparation

```bash
py code/research/dataset_preparation/build_15m_dataset.py
```

Requires `historical data just in case/BTCUSDT/**/trading_data.csv`. Outputs:
- `data/raw/btc_1m_combined.csv`
- `data/processed/btc_15m.csv`

## EMA Regime Filter Research

```bash
py code/research/regime_filter_grid_search/run_ema_sweep.py
```

Runs walk-forward EMA sweep (ema_period ∈ [100, 150, 200, 250, 300]). Saves:
- `code/research/regime_filter_grid_search/ema_sweep_results.csv`
- `code/research/regime_filter_grid_search/ema_sweep_report.md`

## Requirements

```
pandas>=2.0.0
```
