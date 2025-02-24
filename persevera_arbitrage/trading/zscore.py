from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
from ..cointegration_approach.utils import calculate_zscore

@dataclass
class CaldeiraMouraConfig:
    """Configuration for Caldeira-Moura trading rule."""
    entry_threshold: float = 2.0
    exit_threshold_short: float = 0.75    # Close short when z < 0.75
    exit_threshold_long: float = -0.50    # Close long when z > -0.50
    stop_loss: float = 0.07              # 7% stop loss
    max_holding_days: int = 50
    portfolio_size: int = 20             # Number of pairs in portfolio (1/20 allocation each)
    initial_capital: float = 1_000_000   # Initial capital for position sizing
    lookback_window: int = 252          # Window for z-score calculation (1 year)

class CaldeiraMouraTradingRule:
    """
    Implementation of Caldeira-Moura trading rules for pairs trading.
    
    Based on: Caldeira & Moura (2013) - Selection of a Portfolio of Pairs Based on
    Cointegration: A Statistical Arbitrage Strategy
    
    Key features:
    - Market neutral: Equal dollar amounts in long and short positions
    - No rebalancing after position opened
    - Only complete position opening/closing (no partial positions)
    - Equal weighting with redistribution to remaining open positions
    """
    
    def __init__(self, config: Optional[CaldeiraMouraConfig] = None):
        """Initialize trading rule."""
        self.config = config or CaldeiraMouraConfig()
        self.positions: Dict[str, int] = {}            # {pair_id: direction}
        self.position_days: Dict[str, int] = {}        # {pair_id: holding_period}
        self.entry_prices: Dict[str, pd.Series] = {}   # {pair_id: prices}
        self.position_sizes: Dict[str, float] = {}     # {pair_id: capital_allocated}
        self.available_capital = self.config.initial_capital
        
    def generate_signals(self,
                        spread: pd.Series,
                        prices: pd.DataFrame,
                        pair_id: str,
                        beta: float) -> Tuple[pd.Series, pd.Series]:
        """Generate trading signals and position sizes for a pair.
        
        Args:
            spread: Spread series for the pair
            prices: Price data for the pair (long and short candidates)
            pair_id: Unique identifier for the pair
            beta: Hedge ratio for the pair
            
        Returns:
            Tuple containing:
            - Series with trading signals (1: long-short, -1: short-long, 0: no position)
            - Series with position sizes
        """
        # Calculate z-score using utility function
        zscore = calculate_zscore(spread, lookback=self.config.lookback_window)
        
        signals = pd.Series(0, index=zscore.index)
        sizes = pd.Series(0.0, index=zscore.index)
        
        for i in range(len(zscore)):
            date = zscore.index[i]
            z = zscore.iloc[i]
            
            if pd.isna(z):
                continue
                
            # Check existing position
            if pair_id in self.positions:
                self.position_days[pair_id] += 1
                current_prices = prices.iloc[i]
                entry_prices = self.entry_prices[pair_id]
                
                # Calculate returns for market-neutral position
                # If long-short (1): long first asset, short second
                # If short-long (-1): short first asset, long second
                returns = (current_prices - entry_prices) / entry_prices
                portfolio_return = self.positions[pair_id] * (returns.iloc[0] - returns.iloc[1])
                
                # Check closing conditions
                should_close = (
                    (self.positions[pair_id] == 1 and z > self.config.exit_threshold_long) or
                    (self.positions[pair_id] == -1 and z < self.config.exit_threshold_short) or
                    (abs(portfolio_return) > self.config.stop_loss) or
                    (self.position_days[pair_id] >= self.config.max_holding_days)
                )
                
                if should_close:
                    signals.iloc[i] = 0
                    sizes.iloc[i] = 0
                    self._close_position(pair_id)
                else:
                    signals.iloc[i] = self.positions[pair_id]
                    sizes.iloc[i] = self.position_sizes[pair_id]
                    
            # Check for new position entry
            elif abs(z) > self.config.entry_threshold:
                if len(self.positions) < self.config.portfolio_size:
                    signal = -1 if z > 0 else 1
                    position_size = self._calculate_position_size()
                    if position_size > 0:
                        signals.iloc[i] = signal
                        sizes.iloc[i] = position_size
                        self._open_position(pair_id, signal, prices.iloc[i], position_size)
        
        return signals, sizes
    
    def _calculate_position_size(self) -> float:
        """Calculate position size for new pair entry."""
        n_current = len(self.positions)
        if n_current >= self.config.portfolio_size:
            return 0.0
        
        # Equal weight allocation
        return self.available_capital / (self.config.portfolio_size - n_current)
    
    def _open_position(self, pair_id: str, signal: int, prices: pd.Series, size: float) -> None:
        """Open new position for a pair."""
        self.positions[pair_id] = signal
        self.position_days[pair_id] = 0
        self.entry_prices[pair_id] = prices.copy()
        self.position_sizes[pair_id] = size
        self.available_capital -= size
    
    def _close_position(self, pair_id: str) -> None:
        """Close position for a pair."""
        # Return capital to available pool
        self.available_capital += self.position_sizes[pair_id]
        
        # Clear position tracking
        del self.positions[pair_id]
        del self.position_days[pair_id]
        del self.entry_prices[pair_id]
        del self.position_sizes[pair_id]
        
        # Redistribute capital to remaining positions if any
        if self.positions:
            additional_per_position = self.available_capital / len(self.positions)
            for pid in self.positions:
                self.position_sizes[pid] += additional_per_position
            self.available_capital = 0
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get current portfolio statistics."""
        return {
            'n_positions': len(self.positions),
            'available_capital': self.available_capital,
            'allocated_capital': sum(self.position_sizes.values()),
            'total_capital': self.available_capital + sum(self.position_sizes.values())
        }
    
    def reset(self) -> None:
        """Reset trading rule state."""
        self.positions.clear()
        self.position_days.clear()
        self.entry_prices.clear()
        self.position_sizes.clear()
        self.available_capital = self.config.initial_capital
