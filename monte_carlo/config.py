"""Monte Carlo simulation configuration object."""

from dataclasses import dataclass

@dataclass
class MCConfig:
    """Configuration for Monte Carlo simulations.
    
    Attributes:
        num_simulations: Number of Monte Carlo paths to simulate
        horizon_bars: Number of bars to simulate (max path length)
        use_log_returns: Whether to use log returns for calculations
        drift_mode: Method for calculating drift ('historical', 'zero', 'custom')
        sigma_mode: Method for calculating volatility ('historical', 'ewma', 'custom')
        rolling_window: Number of recent bars for volatility/drift estimation
        random_seed: Random seed for reproducibility (None for random)
        var_confidence: Confidence level for Value at Risk (e.g., 0.95)
        cvar_confidence: Confidence level for Conditional Value at Risk (e.g., 0.95)
    """
    num_simulations: int = 10000
    horizon_bars: int = 500
    use_log_returns: bool = True
    drift_mode: str = 'historical'
    sigma_mode: str = 'historical'
    rolling_window: int = 20
    random_seed: int | None = None
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95


# Default configuration instance
default_config = MCConfig()
