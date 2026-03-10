# EMA Regime Filter Grid Search Report

## Summary Table (OOS, averaged across walk-forward steps)

| ema_period | number_of_trades | win_rate | expectancy | profit_factor | sharpe_ratio | max_drawdown | average_R_per_trade |
|------------|-----------------|---------|------------|---------------|--------------|--------------|---------------------|
| baseline | 16.9 | 0.130 | -39.87 | 0.612 | -1329.948 | -511.63 | -0.481 |
| 100 | 12.1 | 0.097 | -53.19 | 0.401 | -1193.576 | -450.72 | -0.614 |
| 150 | 11.9 | 0.101 | -51.93 | 0.414 | -1190.564 | -446.53 | -0.597 |
| 200 | 11.5 | 0.099 | -52.63 | 0.404 | -1011.301 | -428.09 | -0.602 |
| 250 | 11.4 | 0.099 | -52.86 | 0.401 | -1012.012 | -429.98 | -0.605 |
| 300 | 11.4 | 0.103 | -51.41 | 0.420 | -1008.057 | -416.95 | -0.587 |

## Walk-Forward
- Steps completed: 16
- EMA periods: [100, 150, 200, 250, 300]
- Baseline (no filter) included