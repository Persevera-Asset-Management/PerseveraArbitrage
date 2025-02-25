from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
from ..cointegration_approach.utils import calculate_zscore
from ..utils.logging import get_logger

# Get logger from the package
logger = get_logger('persevera_arbitrage.trading.zscore')

@dataclass
class Trade:
    """Class to represent a single trade with its performance metrics."""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    position_type: str = ""  # "LONG-SHORT" or "SHORT-LONG"
    asset1: str = ""
    asset2: str = ""
    entry_prices: Optional[pd.Series] = None
    exit_prices: Optional[pd.Series] = None
    position_size: float = 0.0
    leg1_return: float = 0.0
    leg2_return: float = 0.0
    portfolio_return: float = 0.0
    pnl_amount: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""
    entry_zscore: float = 0.0
    exit_zscore: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for easy serialization."""
        result = {
            'entry_date': self.entry_date.strftime('%Y-%m-%d') if self.entry_date else None,
            'exit_date': self.exit_date.strftime('%Y-%m-%d') if self.exit_date else None,
            'position_type': self.position_type,
            'asset1': self.asset1,
            'asset2': self.asset2,
            'entry_price_asset1': float(self.entry_prices.iloc[0]) if self.entry_prices is not None else None,
            'entry_price_asset2': float(self.entry_prices.iloc[1]) if self.entry_prices is not None else None,
            'exit_price_asset1': float(self.exit_prices.iloc[0]) if self.exit_prices is not None else None,
            'exit_price_asset2': float(self.exit_prices.iloc[1]) if self.exit_prices is not None else None,
            'position_size': self.position_size,
            'leg1_return': self.leg1_return,
            'leg2_return': self.leg2_return,
            'portfolio_return': self.portfolio_return,
            'pnl_amount': self.pnl_amount,
            'holding_days': self.holding_days,
            'exit_reason': self.exit_reason,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore
        }
        return result

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
        
        # Performance tracking
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.max_drawdown: float = 0.0
        self.peak_capital: float = self.config.initial_capital
        
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
                current_trade_return, leg1_return, leg2_return, pnl_amount = self._calculate_portfolio_return(current_prices)
                
                if self.position == 1:
                    position_type = f"LONG {asset1} / SHORT {asset2}"
                else:
                    position_type = f"SHORT {asset1} / LONG {asset2}"
                
                if self.config.verbose:
                    logger.info(f"{date}: Holding {position_type} position (Day {self.position_days}, Z-score: {z:.2f}, Return: {current_trade_return:.2%})")
                
                # Check closing conditions
                should_close = False
                close_reason = ""
                
                if self.position == 1 and z > self.config.exit_threshold_long:
                    should_close = True
                    close_reason = f"Z-score ({z:.2f}) above exit threshold ({self.config.exit_threshold_long})"
                elif self.position == -1 and z < self.config.exit_threshold_short:
                    should_close = True
                    close_reason = f"Z-score ({z:.2f}) below exit threshold ({self.config.exit_threshold_short})"
                elif current_trade_return <= -self.config.stop_loss:  # Use negative of stop_loss since it's stored as positive
                    should_close = True
                    close_reason = f"Stop loss triggered ({current_trade_return:.2%} <= -{self.config.stop_loss:.2%})"
                elif self.position_days >= self.config.max_holding_days:
                    should_close = True
                    close_reason = f"Maximum holding period reached ({self.position_days} days)"
                
                if should_close:
                    if self.config.verbose:
                        logger.info(f"{date}: CLOSING {position_type} position. Reason: {close_reason}")
                    signals.iloc[i] = 0
                    sizes.iloc[i] = 0
                    self._close_position(asset1, asset2, current_prices, close_reason, date, z)
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
                    logger.info(f"{date}: OPENING {position_type} position. Z-score: {z:.2f}, Size: {position_size:,.2f}")
                signals.iloc[i] = signal
                sizes.iloc[i] = position_size
                self._open_position(signal, prices.iloc[i], position_size, asset1, asset2, date, z)
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
    
    def _open_position(self, signal: int, prices: pd.Series, size: float, asset1: str, asset2: str, 
                       date: datetime = None, zscore: float = None) -> None:
        """Open new position."""
        self.position = signal
        self.position_days = 0
        self.entry_prices = prices.copy()
        self.position_size = size
        self.available_capital = 0  # All capital is now allocated
        
        # Calculate the dollar amount allocated to each leg
        leg_allocation = size / 2  # Equal dollar allocation to each leg
        
        # Create a new trade record
        position_type = f"LONG {asset1} / SHORT {asset2}" if signal == 1 else f"SHORT {asset1} / LONG {asset2}"
        
        self.current_trade = Trade(
            entry_date=date if date is not None else datetime.now(),
            position_type=position_type,
            asset1=asset1,
            asset2=asset2,
            entry_prices=prices.copy(),
            position_size=size,
            entry_zscore=zscore if zscore is not None else 0.0
        )
        
        if self.config.verbose:
            logger.info(f"Position opened: {position_type}")
            logger.info(f"Entry prices: {asset1}={prices.iloc[0]:,.2f}, {asset2}={prices.iloc[1]:,.2f}")
            logger.info(f"Position size: {size:,.2f} ({leg_allocation:,.2f} per leg)")
            logger.info(f"Available capital: {self.available_capital:,.2f}")
    
    def _close_position(self, asset1: str = None, asset2: str = None, current_prices: pd.Series = None,
                        close_reason: str = "", date: datetime = None, zscore: float = None) -> None:
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
            current_trade_return, leg1_return, leg2_return, pnl_amount = self._calculate_portfolio_return(current_prices)
                
            # Add detailed leg returns to the P&L info
            if self.position == 1:
                leg_info = f" [LONG {asset1}: {leg1_return:.2%}, SHORT {asset2}: {leg2_return:.2%}]"
            else:
                leg_info = f" [SHORT {asset1}: {leg1_return:.2%}, LONG {asset2}: {leg2_return:.2%}]"
                
            pnl_info = f", P&L: {pnl_amount:,.2f} ({current_trade_return:.2%}){leg_info}"
            
            # Update available capital with P&L
            self.available_capital = position_size + pnl_amount
            
            # Update trade record
            if self.current_trade is not None:
                self.current_trade.exit_date = date if date is not None else datetime.now()
                self.current_trade.exit_prices = current_prices.copy()
                self.current_trade.leg1_return = leg1_return
                self.current_trade.leg2_return = leg2_return
                self.current_trade.portfolio_return = current_trade_return
                self.current_trade.pnl_amount = pnl_amount
                self.current_trade.holding_days = holding_days
                self.current_trade.exit_reason = close_reason
                self.current_trade.exit_zscore = zscore if zscore is not None else None
                
                # Add completed trade to history
                self.trades.append(self.current_trade)
                self.current_trade = None
                
                # Update drawdown tracking
                total_capital = self.available_capital
                if total_capital > self.peak_capital:
                    self.peak_capital = total_capital
                else:
                    drawdown = (self.peak_capital - total_capital) / self.peak_capital
                    if drawdown > self.max_drawdown:
                        self.max_drawdown = drawdown
        else:
            # If we don't have prices to calculate P&L, just return the original position size
            self.available_capital = position_size
        
        if self.config.verbose:
            logger.info(f"Position closed: {position_type}{pnl_info}")
            # If we have current prices, log them
            if current_prices is not None and asset1 and asset2:
                logger.info(f"Closing prices: {asset1}={current_prices.iloc[0]:,.2f}, {asset2}={current_prices.iloc[1]:,.2f}")
                if self.entry_prices is not None:
                    logger.info(f"Entry prices: {asset1}={self.entry_prices.iloc[0]:,.2f}, {asset2}={self.entry_prices.iloc[1]:,.2f}")
            logger.info(f"Holding period: {holding_days} days")
            logger.info(f"Capital returned: {position_size:,.2f}")
            logger.info(f"Available capital: {self.available_capital:,.2f}")
        
        # Clear position tracking
        self.position = 0
        self.position_days = 0
        self.entry_prices = None
        self.position_size = 0.0
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get current portfolio statistics."""
        allocated_capital = self.position_size if self.position != 0 else 0.0
        total_capital = self.available_capital + allocated_capital
        
        # Calculate cumulative P&L and return
        cumulative_pnl = total_capital - self.config.initial_capital
        cumulative_return = cumulative_pnl / self.config.initial_capital
        
        # Get position type with asset names if available
        position_type = "NONE"
        if self.position != 0 and hasattr(self, 'entry_prices') and self.entry_prices is not None and isinstance(self.entry_prices, pd.Series) and len(self.entry_prices) >= 2:
            asset_names = self.entry_prices.index
            if len(asset_names) >= 2:
                asset1, asset2 = asset_names.iloc[0], asset_names.iloc[1]
                if self.position == 1:
                    position_type = f"LONG {asset1} / SHORT {asset2}"
                else:
                    position_type = f"SHORT {asset1} / LONG {asset2}"
            else:
                position_type = "LONG-SHORT" if self.position == 1 else "SHORT-LONG"
        elif self.position != 0:
            position_type = "LONG-SHORT" if self.position == 1 else "SHORT-LONG"
        
        # Calculate trade statistics
        num_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.pnl_amount > 0)
        losing_trades = sum(1 for trade in self.trades if trade.pnl_amount <= 0)
        win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
        
        avg_return = np.mean([trade.portfolio_return for trade in self.trades]) if self.trades else 0.0
        avg_win = np.mean([trade.portfolio_return for trade in self.trades if trade.pnl_amount > 0]) if winning_trades > 0 else 0.0
        avg_loss = np.mean([trade.portfolio_return for trade in self.trades if trade.pnl_amount <= 0]) if losing_trades > 0 else 0.0
        
        avg_holding_days = np.mean([trade.holding_days for trade in self.trades]) if self.trades else 0.0
        
        # Calculate annualized metrics if we have trades
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        calmar_ratio = 0.0
        
        if self.trades:
            # Calculate annualized metrics from trade history
            annualized_return, annualized_volatility, sharpe_ratio = self._calculate_annualized_metrics()
            
            # Calculate Calmar ratio (annualized return / max drawdown)
            if self.max_drawdown < 0:
                calmar_ratio = annualized_return / self.max_drawdown
        
        stats = {
            'has_position': self.position != 0,
            'position_type': position_type,
            'position_days': self.position_days if self.position != 0 else 0,
            'available_capital': self.available_capital,
            'allocated_capital': allocated_capital,
            'total_capital': total_capital,
            'cumulative_pnl': cumulative_pnl,
            'cumulative_return': cumulative_return,
            'max_drawdown': self.max_drawdown,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_days': avg_holding_days,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio
        }
        
        if self.config.verbose:
            logger.info(f"Portfolio stats: {stats}")
            logger.info(f"Cumulative P&L: {cumulative_pnl:,.2f} ({cumulative_return:.2%})")
            if num_trades > 0:
                logger.info(f"Trade stats: {num_trades} trades, {win_rate:.1%} win rate, {avg_holding_days:.1f} avg days")
                logger.info(f"Returns: {avg_return:.2%} avg, {avg_win:.2%} avg win, {avg_loss:.2%} avg loss")
                logger.info(f"Annualized: {annualized_return:.2%} return, {annualized_volatility:.2%} volatility")
                logger.info(f"Risk metrics: Sharpe {sharpe_ratio:.2f}, Calmar {calmar_ratio:.2f}")
                logger.info(f"Max drawdown: {self.max_drawdown:.2%}")
        
        return stats
    
    def _calculate_annualized_metrics(self, risk_free_rate: float = 0.02) -> Tuple[float, float, float]:
        """
        Calculate annualized return, volatility, and Sharpe ratio from trade history.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Tuple containing:
            - annualized_return: Annualized return
            - annualized_volatility: Annualized volatility
            - sharpe_ratio: Sharpe ratio
        """
        if not self.trades:
            return 0.0, 0.0, 0.0
            
        # Create a DataFrame with trade returns and dates
        trade_data = []
        for trade in self.trades:
            if trade.exit_date and trade.entry_date:
                trade_data.append({
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'return': trade.portfolio_return,
                    'holding_days': trade.holding_days
                })
        
        if not trade_data:
            return 0.0, 0.0, 0.0
            
        df = pd.DataFrame(trade_data)
        
        # Sort by exit date to get chronological order
        df = df.sort_values('exit_date')
        
        # Calculate total trading days
        if len(df) > 1:
            first_date = df['entry_date'].min()
            last_date = df['exit_date'].max()
            total_days = (last_date - first_date).days
            if total_days <= 0:
                total_days = sum(df['holding_days'])
        else:
            total_days = sum(df['holding_days'])
        
        # Ensure we have at least one day to avoid division by zero
        total_days = max(1, total_days)
        
        # Calculate annualized metrics
        # Compound the returns to get total return
        total_return = np.prod(1 + df['return']) - 1
        
        # Annualize the return (assuming 252 trading days per year)
        annualized_return = (1 + total_return) ** (252 / total_days) - 1
        
        # Calculate volatility of returns
        if len(df) > 1:
            # Standard deviation of returns
            return_volatility = np.std(df['return'])
            
            # Annualize volatility
            annualized_volatility = return_volatility * np.sqrt(252 / (total_days / len(df)))
        else:
            # If only one trade, use the absolute return as volatility
            return_volatility = abs(df['return'].iloc[0])
            annualized_volatility = return_volatility * np.sqrt(252 / total_days)
        
        # Calculate Sharpe ratio
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0.0
        
        return annualized_return, annualized_volatility, sharpe_ratio
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_dicts = [trade.to_dict() for trade in self.trades]
        return pd.DataFrame(trade_dicts)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Calculate and return the equity curve of the strategy.
        
        Returns:
            DataFrame with daily equity values and related metrics
        """
        if not self.trades:
            return pd.DataFrame()
        
        # Get trade history
        trades_df = self.get_trade_history()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Convert date strings to datetime
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Sort by exit date
        trades_df = trades_df.sort_values('exit_date')
        
        # Get the full date range from first entry to last exit
        start_date = trades_df['entry_date'].min()
        end_date = trades_df['exit_date'].max()
        
        # Create a DataFrame with all dates in the range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        equity_curve = pd.DataFrame(index=date_range)
        equity_curve.index.name = 'date'
        
        # Initialize with initial capital
        equity_curve['equity'] = self.config.initial_capital
        equity_curve['trade_active'] = False
        equity_curve['trade_entry'] = False
        equity_curve['trade_exit'] = False
        equity_curve['return'] = 0.0
        equity_curve['drawdown'] = 0.0
        
        # Track current equity
        current_equity = self.config.initial_capital
        peak_equity = current_equity
        
        # Process each trade
        for _, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            pnl = trade['pnl_amount']
            
            # Mark trade entry
            if entry_date in equity_curve.index:
                equity_curve.loc[entry_date, 'trade_entry'] = True
            
            # Mark active trade period
            trade_period = equity_curve.index[(equity_curve.index >= entry_date) & (equity_curve.index <= exit_date)]
            equity_curve.loc[trade_period, 'trade_active'] = True
            
            # Mark trade exit and update equity
            if exit_date in equity_curve.index:
                equity_curve.loc[exit_date, 'trade_exit'] = True
                current_equity += pnl
                
                # Update equity for all days after this exit
                future_days = equity_curve.index[equity_curve.index >= exit_date]
                equity_curve.loc[future_days, 'equity'] = current_equity
                
                # Update peak equity if needed
                if current_equity > peak_equity:
                    peak_equity = current_equity
        
        # Calculate returns and drawdowns
        equity_curve['return'] = equity_curve['equity'].pct_change()
        equity_curve['cumulative_return'] = equity_curve['equity'] / self.config.initial_capital - 1
        
        # Calculate drawdown
        equity_curve['peak_equity'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['peak_equity'] - equity_curve['equity']) / equity_curve['peak_equity']
        
        return equity_curve
    
    def reset(self) -> None:
        """Reset trading rule state."""
        if self.config.verbose:
            logger.info("Resetting trading rule state")
            
            # Log final performance before reset if capital has changed
            current_capital = self.available_capital + (self.position_size if self.position != 0 else 0.0)
            if current_capital != self.config.initial_capital:
                pnl = current_capital - self.config.initial_capital
                pnl_return = pnl / self.config.initial_capital
                logger.info(f"Final performance before reset: P&L={pnl:,.2f} ({pnl_return:.2%})")
        
        self.position = 0
        self.position_days = 0
        self.entry_prices = None
        self.position_size = 0.0
        self.historical_spread = None
        self.available_capital = self.config.initial_capital
        
        # Reset performance tracking
        self.trades = []
        self.current_trade = None
        self.max_drawdown = 0.0
        self.peak_capital = self.config.initial_capital
        
        if self.config.verbose:
            logger.info(f"Trading rule reset. Available capital: {self.available_capital:,.2f}")

    def _calculate_portfolio_return(self, current_prices: pd.Series) -> Tuple[float, float, float, float]:
        """
        Calculate returns based on equal dollar allocation to each leg.
        
        Args:
            current_prices: Current prices for the pair
            
        Returns:
            Tuple containing:
            - current_trade_return: Return of the current trade
            - leg1_return: Return of the first leg
            - leg2_return: Return of the second leg
            - pnl_amount: Dollar amount of profit/loss
        """
        if self.position == 1:
            # Long asset1, short asset2 with equal dollar amounts
            leg1_return = (current_prices / self.entry_prices - 1).iloc[0]  # Long return
            leg2_return = - (current_prices / self.entry_prices - 1).iloc[1]  # Short return
        else:
            # Short asset1, long asset2 with equal dollar amounts
            leg1_return = - (current_prices / self.entry_prices - 1).iloc[0]  # Short return
            leg2_return = (current_prices / self.entry_prices - 1).iloc[1]  # Long return
            
        # Average of both legs
        current_trade_return = (leg1_return + leg2_return) / 2
        
        # Calculate P&L amount
        pnl_amount = current_trade_return * self.position_size
        
        return current_trade_return, leg1_return, leg2_return, pnl_amount
