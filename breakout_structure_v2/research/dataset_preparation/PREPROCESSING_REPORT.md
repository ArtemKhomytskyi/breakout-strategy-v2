# Preprocessing Report


=== PREPROCESSING REPORT ===

Files loaded: 63
Schema: ['open_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_buy_volume']
Inconsistencies fixed:
  - Column names standardized to snake_case
  - Timestamps parsed with pd.to_datetime
  - Rows sorted chronologically

Duplicates removed: 0
Missing values (before OHLCV drop): {}

Date range 1m: 2020-01-01 00:00:00 to 2025-03-15 23:59:00
Date range 15m: 2020-01-01 00:00:00 to 2025-03-15 23:45:00

Row count 1m: 2735093
Row count 15m: 182343

Assumptions:
  - Timestamps assumed UTC (no conversion)
  - trading_data.csv used (spot), not futures_data.csv
  - num_trades and taker_buy_volume summed in 15m aggregation

Outputs:
  - C:\Users\artkh\Downloads\breakout_structure_v2\data\raw\btc_1m_combined.csv
  - C:\Users\artkh\Downloads\breakout_structure_v2\data\processed\btc_15m.csv
