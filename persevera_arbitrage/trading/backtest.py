from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .zscore import CaldeiraMouraTradingRule, CaldeiraMouraConfig
from ..cointegration_approach.utils import calculate_zscore
from ..utils.logging import get_logger

# Get logger from the package
logger = get_logger('persevera_arbitrage.trading.backtest')

class Backtester:
    """
    Backtester for pairs trading strategies using historical data.
    
    This class allows backtesting of pairs trading strategies on historical price data
    without using simulations. It uses the CaldeiraMouraTradingRule to generate signals
    and track performance.
    """
    
    def __init__(self, config: Optional[CaldeiraMouraConfig] = None):
        """
        Initialize backtester with configuration.
        
        Args:
            config: Configuration for the trading rule
        """
        self.config = config or CaldeiraMouraConfig()
        self.trading_rule = CaldeiraMouraTradingRule(self.config)
        
    def run_backtest(self, 
                    price_data: pd.DataFrame, 
                    hedge_ratio: float,
                    training_window: int = 252,
                    dependent_variable: Optional[str] = None) -> Dict:
        """
        Run backtest on historical price data.
        
        Args:
            price_data: DataFrame with price data for the pair
            hedge_ratio: Hedge ratio for the pair
            training_window: Number of days to use for initial training
            dependent_variable: Which asset is the dependent variable (Y in Y = β*X + ε)
                                If None, uses the first column
        
        Returns:
            Dictionary with backtest results
        """
        if len(price_data.columns) != 2:
            raise ValueError("Price data must contain exactly two assets")
            
        # Ensure we have enough data for training
        if len(price_data) <= training_window:
            raise ValueError(f"Not enough data for training window of {training_window} days")
            
        # Determine dependent and independent variables
        if dependent_variable is None:
            dependent_variable = price_data.columns[0]
            
        if dependent_variable not in price_data.columns:
            raise ValueError(f"Dependent variable {dependent_variable} not found in price data")
            
        # Split data into training and testing periods
        training_data = price_data.iloc[:training_window]
        testing_data = price_data.iloc[training_window:]
        
        if len(testing_data) == 0:
            raise ValueError("No data left for testing after training window")
            
        # Calculate historical spread for training period
        historical_spread = self._calculate_spread(training_data, hedge_ratio, dependent_variable)
        
        # Calculate spread for the entire period (for z-score calculation)
        full_spread = self._calculate_spread(price_data, hedge_ratio, dependent_variable)
        
        # Calculate spread for testing period
        testing_spread = full_spread.loc[testing_data.index]
        
        # Reset trading rule to ensure clean state
        self.trading_rule.reset()
        
        # Generate trading signals
        signals, position_sizes = self.trading_rule.generate_signals(
            simulated_spread=testing_spread,
            historical_spread=historical_spread,
            prices=testing_data
        )
        
        # Get performance metrics
        portfolio_stats = self.trading_rule.get_portfolio_stats()
        equity_curve = self.trading_rule.get_equity_curve()
        trade_history = self.trading_rule.get_trade_history()
        
        # Combine results
        results = {
            'portfolio_stats': portfolio_stats,
            'equity_curve': equity_curve,
            'trade_history': trade_history,
            'signals': signals,
            'position_sizes': position_sizes,
            'pair': '_'.join(price_data.columns),
            'hedge_ratio': hedge_ratio,
            'training_window': training_window,
            'testing_period': (testing_data.index[0], testing_data.index[-1]),
            'n_days': len(testing_data)
        }
        
        return results
    
    def gbvgtg_calculate_spread(self, price_data: pd.DataFrame, hedge_ratio: float, dependent_variable: str) -> pd.Series:
        """
        Calculate spread between two assets using the hedge ratio.
        
        Args:
            price_data: DataFrame with price data
            hedge_ratio: Hedge ratio for the pair
            dependent_variable: Which asset is the dependent variable
            
        Returns:
            Series with spread values
        """
        # Get column indices
        dependent_idx = price_data.columns.get_loc(dependent_variable)
        independent_idx = 1 - dependent_idx  # Assuming only 2 columns
        
        # Calculate spread: Y - β*X
        spread = price_data.iloc[:, dependent_idx] - hedge_ratio * price_data.iloc[:, independent_idx]
        
        return spread 