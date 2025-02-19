from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class CointegrationConfig:
    """Configuration for cointegration portfolios."""
    
    # Test parameters
    significance_level: float = 0.05    # Test significance level
    det_order: Literal[-1, 0, 1] = 0    # Deterministic term in test (-1, 0, or 1)
    n_lags: int = 1                     # Number of lags in VAR test
    
    # Portfolio parameters
    min_history: int = 252              # Minimum price history required
    position_size: float = 1.0          # Base position size
    
    # Optional parameters
    dependent_variable: Optional[str] = None