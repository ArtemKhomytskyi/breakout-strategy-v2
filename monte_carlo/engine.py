from .models import TradeCandidate, MarketState
from .path_generator import generate_paths
from .trade_replay import replay_trade
from .metrics import calculate_risk_metrics
from .decision import make_trade_decision

def run_monte_carlo_analysis(candidate: TradeCandidate, market_state: MarketState):
    """
    The main entry point that runs the full pipeline:
    1. Generate Paths
    2. Replay Trade
    3. Calculate Metrics
    4. Return Decision
    """
    pass