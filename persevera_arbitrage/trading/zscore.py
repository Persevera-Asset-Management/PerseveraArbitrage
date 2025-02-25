from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from ..cointegration_approach.utils import calculate_zscore
from ..utils.logging import get_logger

# Get logger from the package
logger = get_logger('persevera_arbitrage.trading.zscore')

@dataclass
class CaldeiraMouraConfig:
    """Configuration for Caldeira-Moura trading rule."""
    entry_threshold: float = 2.0
    exit_threshold_short: float = 0.75   # Close short when z < 0.75
    exit_threshold_long: float = -0.50   # Close long when z > -0.50
    stop_loss: float = 0.07              # 7% stop loss (positive value for easier understanding)
    max_holding_days: int = 50
    initial_capital: float = 1_000_000   # Initial capital for position sizing
    lookback_window: int = 252           # Window for z-score calculation (1 year)
    verbose: bool = True                 # Whether to print trading actions

class CaldeiraMouraTradingRule:
    """
    Implementation of Caldeira-Moura trading rules for pairs trading.
    
    Based on: Caldeira & Moura (2013) - Selection of a Portfolio of Pairs Based on
    Cointegration: A Statistical Arbitrage Strategy
    
    Key features:
    - Market neutral: Equal dollar amounts in long and short positions
    - No rebalancing after position opened
    - Only complete position opening/closing (no partial positions)
    - Full capital allocation to each pair (no portfolio diversification)
    """
    
    def __init__(self, config: Optional[CaldeiraMouraConfig] = None):
        """Initialize trading rule."""
        self.config = config or CaldeiraMouraConfig()
        self.position: int = 0                        # Direction: 1 (long-short) or -1 (short-long) or 0 (no position)
        self.position_days: int = 0                   # Holding period for current position
        self.entry_prices: Optional[pd.Series] = None # Entry prices for current position
        self.position_size: float = 0.0               # Capital allocated to current position
        self.available_capital = self.config.initial_capital
        self.historical_spread: Optional[pd.Series] = None  # Historical spread data for hybrid z-score calculation
    
    def generate_signals(self,
                         simulated_spread: pd.Series,
                         historical_spread: Optional[pd.Series],
                         prices: pd.DataFrame,
                         beta: float) -> Tuple[pd.Series, pd.Series]:
        """Generate trading signals and position sizes for a pair.
        
        Args:
            simulated_spread: Spread series for the pair
            historical_spread: Optional historical spread data for hybrid z-score calculation
            prices: Price data for the pair (long and short candidates)
            beta: Hedge ratio for the pair (used for z-score calculation, not for position sizing)
            
        Returns:
            Tuple containing:
            - Series with trading signals (1: long-short, -1: short-long, 0: no position)
            - Series with position sizes
        """
        # Construct pair name by joining column names with underscore
        pair_name = '_'.join(prices.columns)
        asset1, asset2 = prices.columns[0], prices.columns[1]

        # Store historical spread if provided
        if historical_spread is not None:
            self.historical_spread = historical_spread
            if self.config.verbose:
                logger.info(f"Stored historical spread data for {pair_name} with {len(historical_spread)} data points")
                logger.info(f"Using hedge ratio (beta): {beta:.4f} for spread calculation only")
        
        # Calculate z-score using hybrid approach if historical data available
        if self.historical_spread is not None:
            if self.config.verbose:
                logger.info(f"Using hybrid z-score calculation for {pair_name}")
            zscore = self.calculate_hybrid_zscore(
                simulated_spread=simulated_spread,
                historical_spread=self.historical_spread,
                window=self.config.lookback_window
            )
        else:
            if self.config.verbose:
                logger.info(f"Using standard z-score calculation for {pair_name} (no historical data available)")
            zscore = calculate_zscore(simulated_spread, window=self.config.lookback_window)
        
        signals = pd.Series(0, index=zscore.index)
        sizes = pd.Series(0.0, index=zscore.index)
        
        for i in range(len(zscore)):
            date = zscore.index[i]
            z = zscore.iloc[i]
            
            if pd.isna(z):
                if self.config.verbose:
                    logger.warning(f"{date}: Z-score is NaN for {pair_name}, skipping day")
                continue
                
            # Check existing position
            if self.position != 0:
                self.position_days += 1
                current_prices = prices.iloc[i]
                
                # Calculate returns based on equal dollar allocation
                # We calculate the return on each leg separately and then combine them
                if self.position == 1:
                    # Long asset1, short asset2 with equal dollar amounts
                    leg1_return = (current_prices[0] - self.entry_prices[0]) / self.entry_prices[0]  # Long return
                    leg2_return = (self.entry_prices[1] - current_prices[1]) / self.entry_prices[1]  # Short return
                    portfolio_return = (leg1_return + leg2_return) / 2  # Average of both legs
                else:
                    # Short asset1, long asset2 with equal dollar amounts
                    leg1_return = (self.entry_prices[0] - current_prices[0]) / self.entry_prices[0]  # Short return
                    leg2_return = (current_prices[1] - self.entry_prices[1]) / self.entry_prices[1]  # Long return
                    portfolio_return = (leg1_return + leg2_return) / 2  # Average of both legs
                
                if self.position == 1:
                    position_type = f"LONG {asset1} / SHORT {asset2}"
                else:
                    position_type = f"SHORT {asset1} / LONG {asset2}"
                
                if self.config.verbose:
                    logger.info(f"{date}: Holding {position_type} position (Day {self.position_days}, Z-score: {z:.2f}, Return: {portfolio_return:.2%})")
                
                # Check closing conditions
                should_close = False
                close_reason = ""
                
                if self.position == 1 and z > self.config.exit_threshold_long:
                    should_close = True
                    close_reason = f"Z-score ({z:.2f}) above exit threshold ({self.config.exit_threshold_long})"
                elif self.position == -1 and z < self.config.exit_threshold_short:
                    should_close = True
                    close_reason = f"Z-score ({z:.2f}) below exit threshold ({self.config.exit_threshold_short})"
                elif portfolio_return <= -self.config.stop_loss:  # Use negative of stop_loss since it's stored as positive
                    should_close = True
                    close_reason = f"Stop loss triggered ({portfolio_return:.2%} <= -{self.config.stop_loss:.2%})"
                elif self.position_days >= self.config.max_holding_days:
                    should_close = True
                    close_reason = f"Maximum holding period reached ({self.position_days} days)"
                
                if should_close:
                    if self.config.verbose:
                        logger.info(f"{date}: CLOSING {position_type} position. Reason: {close_reason}")
                    signals.iloc[i] = 0
                    sizes.iloc[i] = 0
                    self._close_position(asset1, asset2, current_prices)
                else:
                    signals.iloc[i] = self.position
                    sizes.iloc[i] = self.position_size
                    
            # Check for new position entry
            elif abs(z) > self.config.entry_threshold:
                # Only open a position if we don't have any active positions
                signal = -1 if z > 0 else 1
                position_size = self.available_capital  # Use all available capital
                
                if signal == 1:
                    position_type = f"LONG {asset1} / SHORT {asset2}"
                else:
                    position_type = f"SHORT {asset1} / LONG {asset2}"
                
                if self.config.verbose:
                    logger.info(f"{date}: OPENING {position_type} position. Z-score: {z:.2f}, Size: ${position_size:,.2f}")
                signals.iloc[i] = signal
                sizes.iloc[i] = position_size
                self._open_position(signal, prices.iloc[i], position_size, asset1, asset2)
            elif self.config.verbose and i % 20 == 0:  # Log every 20 days when no action is taken to reduce verbosity
                logger.info(f"{date}: No trading action for {pair_name}. Z-score: {z:.2f}")
        
        return signals, sizes
    
    def calculate_hybrid_zscore(self, simulated_spread: pd.Series, historical_spread: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate z-scores using a hybrid approach for realistic simulations
        
        Args:
            simulated_spread: Pandas Series of simulated spread values
            historical_spread: Pandas Series of historical spread values
            window: Lookback window size
        
        Returns:
            Pandas Series of z-scores
        """
        # Combine historical and simulated spreads
        combined_spread = pd.concat([historical_spread, simulated_spread])
        combined_spread = combined_spread[~combined_spread.index.duplicated(keep='first')].sort_index()
        
        # Calculate rolling statistics
        rolling_mean = combined_spread.rolling(window=window, min_periods=window).mean()
        rolling_std = combined_spread.rolling(window=window, min_periods=window).std()
        
        # Calculate z-scores
        zscore = (combined_spread - rolling_mean) / rolling_std
        
        # Filter to just simulated period
        return zscore.loc[simulated_spread.index]
    
    def _open_position(self, signal: int, prices: pd.Series, size: float, asset1: str, asset2: str) -> None:
        """Open new position."""
        self.position = signal
        self.position_days = 0
        self.entry_prices = prices.copy()
        self.position_size = size
        self.available_capital = 0  # All capital is now allocated
        
        # Calculate the dollar amount allocated to each leg
        leg_allocation = size / 2  # Equal dollar allocation to each leg
        
        if self.config.verbose:
            if signal == 1:
                position_type = f"LONG {asset1} / SHORT {asset2}"
            else:
                position_type = f"SHORT {asset1} / LONG {asset2}"
                
            logger.info(f"Position opened: {position_type}")
            logger.info(f"Entry prices: {asset1}=${prices[0]:,.2f}, {asset2}=${prices[1]:,.2f}")
            logger.info(f"Position size: ${size:,.2f} (${leg_allocation:,.2f} per leg)")
            logger.info(f"Available capital: ${self.available_capital:,.2f}")
    
    def _close_position(self, asset1: str = None, asset2: str = None, current_prices: pd.Series = None) -> None:
        """Close current position."""
        position_size = self.position_size
        
        if asset1 and asset2:
            if self.position == 1:
                position_type = f"LONG {asset1} / SHORT {asset2}"
            else:
                position_type = f"SHORT {asset1} / LONG {asset2}"
        else:
            position_type = "LONG-SHORT" if self.position == 1 else "SHORT-LONG"
            
        holding_days = self.position_days
        
        # Calculate profit/loss if we have both entry and current prices
        pnl_info = ""
        if self.entry_prices is not None and current_prices is not None:
            # Calculate returns based on equal dollar allocation to each leg
            if self.position == 1:
                # Long asset1, short asset2 with equal dollar amounts
                leg1_return = (current_prices[0] - self.entry_prices[0]) / self.entry_prices[0]  # Long return
                leg2_return = (self.entry_prices[1] - current_prices[1]) / self.entry_prices[1]  # Short return
                portfolio_return = (leg1_return + leg2_return) / 2  # Average of both legs
            else:
                # Short asset1, long asset2 with equal dollar amounts
                leg1_return = (self.entry_prices[0] - current_prices[0]) / self.entry_prices[0]  # Short return
                leg2_return = (current_prices[1] - self.entry_prices[1]) / self.entry_prices[1]  # Long return
                portfolio_return = (leg1_return + leg2_return) / 2  # Average of both legs
                
            pnl_amount = portfolio_return * position_size
            
            # Add detailed leg returns to the P&L info
            if self.position == 1:
                leg_info = f" [LONG {asset1}: {leg1_return:.2%}, SHORT {asset2}: {leg2_return:.2%}]"
            else:
                leg_info = f" [SHORT {asset1}: {leg1_return:.2%}, LONG {asset2}: {leg2_return:.2%}]"
                
            pnl_info = f", P&L: ${pnl_amount:,.2f} ({portfolio_return:.2%}){leg_info}"
        
        # Return capital to available pool
        self.available_capital = position_size
        
        if self.config.verbose:
            logger.info(f"Position closed: {position_type}{pnl_info}")
            # If we have current prices, log them
            if current_prices is not None and asset1 and asset2:
                logger.info(f"Closing prices: {asset1}=${current_prices[0]:,.2f}, {asset2}=${current_prices[1]:,.2f}")
                if self.entry_prices is not None:
                    logger.info(f"Entry prices: {asset1}=${self.entry_prices[0]:,.2f}, {asset2}=${self.entry_prices[1]:,.2f}")
            logger.info(f"Holding period: {holding_days} days")
            logger.info(f"Capital returned: ${position_size:,.2f}")
            logger.info(f"Available capital: ${self.available_capital:,.2f}")
        
        # Clear position tracking
        self.position = 0
        self.position_days = 0
        self.entry_prices = None
        self.position_size = 0.0
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get current portfolio statistics."""
        allocated_capital = self.position_size if self.position != 0 else 0.0
        
        # Get position type with asset names if available
        position_type = "NONE"
        if self.position != 0 and hasattr(self, 'entry_prices') and self.entry_prices is not None and isinstance(self.entry_prices, pd.Series) and len(self.entry_prices) >= 2:
            asset_names = self.entry_prices.index
            if len(asset_names) >= 2:
                asset1, asset2 = asset_names[0], asset_names[1]
                if self.position == 1:
                    position_type = f"LONG {asset1} / SHORT {asset2}"
                else:
                    position_type = f"SHORT {asset1} / LONG {asset2}"
            else:
                position_type = "LONG-SHORT" if self.position == 1 else "SHORT-LONG"
        elif self.position != 0:
            position_type = "LONG-SHORT" if self.position == 1 else "SHORT-LONG"
        
        stats = {
            'has_position': self.position != 0,
            'position_type': position_type,
            'position_days': self.position_days if self.position != 0 else 0,
            'available_capital': self.available_capital,
            'allocated_capital': allocated_capital,
            'total_capital': self.available_capital + allocated_capital
        }
        
        if self.config.verbose:
            logger.info(f"Portfolio stats: {stats}")
        
        return stats
    
    def reset(self) -> None:
        """Reset trading rule state."""
        if self.config.verbose:
            logger.info("Resetting trading rule state")
        
        self.position = 0
        self.position_days = 0
        self.entry_prices = None
        self.position_size = 0.0
        self.historical_spread = None
        self.available_capital = self.config.initial_capital
        
        if self.config.verbose:
            logger.info(f"Trading rule reset. Available capital: ${self.available_capital:,.2f}")
