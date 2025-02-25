import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.zscore import CaldeiraMouraTradingRule, CaldeiraMouraConfig

def generate_synthetic_pair_data(n_historical=252, n_simulation=126, mean_reversion=0.05, 
                                volatility=0.01, beta=0.85, seed=42):
    """
    Generate synthetic price data for a cointegrated pair.
    
    Args:
        n_historical: Number of historical data points
        n_simulation: Number of simulation data points
        mean_reversion: Strength of mean reversion in the spread
        volatility: Volatility of the spread
        beta: Hedge ratio between assets
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - prices_df: DataFrame with price data for both assets
        - historical_spread: Series with historical spread values
        - simulated_spread: Series with simulated spread values
    """
    np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=(n_historical + n_simulation) * 2)  # Extra days for weekends
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    dates = all_dates[-(n_historical + n_simulation):]  # Take the last n_historical + n_simulation days
    
    # Generate spread as a mean-reverting process
    spread = np.zeros(n_historical + n_simulation)
    spread[0] = 0
    
    for i in range(1, len(spread)):
        # Mean-reverting process: next_value = current_value - mean_reversion * (current_value - long_term_mean) + random_shock
        spread[i] = spread[i-1] - mean_reversion * spread[i-1] + np.random.normal(0, volatility)
    
    # Convert to pandas Series
    spread_series = pd.Series(spread, index=dates)
    
    # Generate price for asset1 as a random walk
    asset1_returns = np.random.normal(0.0005, 0.01, n_historical + n_simulation)  # Small positive drift
    asset1_price = 100 * np.cumprod(1 + asset1_returns)
    
    # Generate price for asset2 based on asset1 and the spread
    # If spread = asset1 - beta * asset2, then asset2 = (asset1 - spread) / beta
    asset2_price = (asset1_price - spread) / beta
    
    # Create DataFrame with prices
    prices_df = pd.DataFrame({
        'asset1': asset1_price,
        'asset2': asset2_price
    }, index=dates)
    
    # Split into historical and simulation periods
    historical_dates = dates[:n_historical]
    simulation_dates = dates[n_historical:]
    
    historical_spread = spread_series.loc[historical_dates]
    simulated_spread = spread_series.loc[simulation_dates]
    
    return prices_df, historical_spread, simulated_spread

def run_trading_simulation():
    """Run a complete trading simulation using the CaldeiraMouraTradingRule."""
    
    # 1. Generate synthetic data for a cointegrated pair
    print("Generating synthetic pair data...")
    prices_df, historical_spread, simulated_spread = generate_synthetic_pair_data(
        n_historical=252,  # 1 year of historical data
        n_simulation=126,  # 6 months of simulation
        mean_reversion=0.05,
        volatility=0.01,
        beta=0.85
    )
    
    # Split prices into historical and simulation periods
    historical_prices = prices_df.iloc[:252]
    simulation_prices = prices_df.iloc[252:]
    
    # 2. Create trading rule instance with custom configuration
    print("Initializing trading rule...")
    config = CaldeiraMouraConfig(
        entry_threshold=2.0,
        exit_threshold_short=0.75,
        exit_threshold_long=-0.50,
        stop_loss=0.07,
        max_holding_days=50,
        portfolio_size=20,
        initial_capital=1_000_000,
        lookback_window=252
    )
    trading_rule = CaldeiraMouraTradingRule(config)
    
    # 3. Generate trading signals
    print("Generating trading signals...")
    pair_id = "SYNTHETIC_PAIR_1"
    beta = 0.85
    
    signals, position_sizes = trading_rule.generate_signals(
        simulated_spread=simulated_spread,
        historical_spread=historical_spread,
        prices=simulation_prices,
        pair_id=pair_id,
        beta=beta
    )
    
    # 4. Calculate returns based on signals
    print("Calculating returns...")
    
    # Calculate daily returns for each asset
    asset1_returns = simulation_prices['asset1'].pct_change().fillna(0)
    asset2_returns = simulation_prices['asset2'].pct_change().fillna(0)
    
    # Calculate strategy returns
    # When signal is 1: long asset1, short asset2
    # When signal is -1: short asset1, long asset2
    strategy_returns = pd.Series(0.0, index=signals.index)
    
    for i in range(len(signals)):
        if signals.iloc[i] == 1:  # Long asset1, short asset2
            strategy_returns.iloc[i] = asset1_returns.iloc[i] - asset2_returns.iloc[i]
        elif signals.iloc[i] == -1:  # Short asset1, long asset2
            strategy_returns.iloc[i] = asset2_returns.iloc[i] - asset1_returns.iloc[i]
    
    # Scale returns by position size
    scaled_returns = strategy_returns * position_sizes / config.initial_capital
    
    # Calculate cumulative returns
    cumulative_returns = (1 + scaled_returns).cumprod() - 1
    
    # 5. Calculate performance metrics
    print("Calculating performance metrics...")
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(cumulative_returns)) - 1
    annualized_volatility = scaled_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
    
    # 6. Print results
    print("\n=== Performance Metrics ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Number of Trades: {(signals.diff() != 0).sum()}")
    
    # 7. Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Prices
    plt.subplot(3, 1, 1)
    plt.plot(simulation_prices.index, simulation_prices['asset1'], label='Asset 1')
    plt.plot(simulation_prices.index, simulation_prices['asset2'], label='Asset 2')
    plt.title('Asset Prices')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Spread and Z-score
    plt.subplot(3, 1, 2)
    
    # Calculate z-score for plotting
    if pair_id in trading_rule.historical_spreads:
        z_score = trading_rule.calculate_hybrid_zscore(
            simulated_spread=simulated_spread,
            historical_spread=trading_rule.historical_spreads[pair_id],
            window=config.lookback_window
        )
    else:
        from persevera_arbitrage.cointegration_approach.utils import calculate_zscore
        z_score = calculate_zscore(simulated_spread, window=config.lookback_window)
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(simulated_spread.index, simulated_spread, 'b-', label='Spread')
    ax2.plot(z_score.index, z_score, 'r-', label='Z-score')
    ax1.set_ylabel('Spread', color='b')
    ax2.set_ylabel('Z-score', color='r')
    ax1.grid(True)
    
    # Add horizontal lines for entry/exit thresholds
    ax2.axhline(y=config.entry_threshold, color='g', linestyle='--', alpha=0.7, label='Entry (Long)')
    ax2.axhline(y=-config.entry_threshold, color='g', linestyle='--', alpha=0.7, label='Entry (Short)')
    ax2.axhline(y=config.exit_threshold_short, color='m', linestyle='--', alpha=0.7, label='Exit Short')
    ax2.axhline(y=config.exit_threshold_long, color='m', linestyle='--', alpha=0.7, label='Exit Long')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Spread and Z-score')
    
    # Plot 3: Cumulative Returns and Positions
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot cumulative returns
    ax1.plot(cumulative_returns.index, cumulative_returns * 100, 'g-', label='Cumulative Return (%)')
    ax1.set_ylabel('Cumulative Return (%)', color='g')
    ax1.grid(True)
    
    # Plot positions
    ax2.plot(signals.index, signals, 'b-', label='Position')
    ax2.set_ylabel('Position', color='b')
    ax2.set_ylim(-1.5, 1.5)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Strategy Performance')
    
    plt.tight_layout()
    plt.savefig('caldeira_moura_simulation_results.png')
    plt.show()
    
    return trading_rule, signals, position_sizes, cumulative_returns

if __name__ == "__main__":
    trading_rule, signals, position_sizes, cumulative_returns = run_trading_simulation()
    
    # Print final portfolio statistics
    print("\n=== Final Portfolio Statistics ===")
    print(trading_rule.get_portfolio_stats()) 