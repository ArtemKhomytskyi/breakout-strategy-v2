"""
Generate Monte Carlo price paths for future market scenarios.

Uses geometric Brownian motion (GBM) with log returns.
"""

import numpy as np
from config import MCConfig
from market_state import MarketState


def generate_paths(market_state: MarketState, config: MCConfig) -> np.ndarray:
    """Generate Monte Carlo price paths.
    
    Creates num_simulations price paths, each spanning horizon_bars future bars.
    Uses geometric Brownian motion with normal random shocks.
    
    Args:
        market_state: Current market snapshot with starting price, sigma, drift
        config: MCConfig with num_simulations, horizon_bars, random_seed, drift_mode
        
    Returns:
        2D array of shape (num_simulations, horizon_bars + 1) where:
        - First column is the starting price (repeated)
        - Remaining columns are simulated future prices
        - Each row is one complete price path
        
    Raises:
        ValueError: If market_state has insufficient data or invalid parameters
    """
    # Validate config parameters
    if config.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    
    if config.horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    
    # Validate market data
    if not market_state.recent_close_prices:
        raise ValueError("market_state.recent_close_prices is empty")
    
    # Validate all prices are positive
    closes_array = np.asarray(market_state.recent_close_prices)
    if np.any(closes_array <= 0):
        raise ValueError("all prices must be positive")
    
    S0 = float(market_state.recent_close_prices[-1])  # Starting price (most recent close)
    
    sigma = float(market_state.sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")
    
    # Create local random number generator (doesn't affect global state)
    rng = np.random.default_rng(config.random_seed)
    
    # Extract parameters
    drift = 0.0 if config.drift_mode == "zero" else float(market_state.drift)
    if not np.isfinite(drift):
        raise ValueError("drift must be finite")
    
    num_paths = config.num_simulations
    num_steps = config.horizon_bars
    
    # Initialize price matrix: (num_simulations, num_steps + 1)
    # Column 0 is starting price, columns 1:num_steps+1 are future prices
    prices = np.empty((num_paths, num_steps + 1), dtype=float)
    prices[:, 0] = S0  # All paths start at S0
    
    # Generate random normal shocks: (num_simulations, num_steps)
    # Each row is one path, each column is one time step
    random_shocks = rng.standard_normal((num_paths, num_steps))
    
    # Simulate paths using GBM with log returns (vectorized)
    # log_return_t = (drift - 0.5*sigma^2) + sigma * Z_t, where Z_t ~ N(0,1)
    # S_t = S_0 * exp(sum(log_returns))
    
    adjusted_drift = drift - 0.5 * sigma**2
    log_returns = adjusted_drift + sigma * random_shocks
    
    # Compute cumulative log returns for each path
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    
    # Convert to prices: S_t = S_0 * exp(cumulative_log_returns)
    prices[:, 1:] = S0 * np.exp(cumulative_log_returns)
    
    return prices