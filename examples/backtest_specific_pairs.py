import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from persevera_tools.data import get_descriptors, get_index_composition
from persevera_arbitrage.cointegration_approach.pairs_selection import CointegrationPairSelector, PairSelectionCriteria
from persevera_arbitrage.trading import CaldeiraMouraTradingRule, CaldeiraMouraConfig, Backtester
from persevera_arbitrage.cointegration_approach import EngleGrangerPortfolio

# 1. Get index composition
members = get_index_composition(index_code='IBX50', start_date='2024-01-01', end_date='2024-01-01')

# 2. Load price data for all members
price_data = get_descriptors(
    tickers=[*members.columns],
    descriptors='price_close',
    start_date='2023-01-01',
    end_date='2024-01-05'
)
price_data.dropna(axis=1, inplace=True)

# 3. Create selector instance
selector = CointegrationPairSelector()

# 4. Find best pairs
all_pairs = selector.select_pairs(price_data)
passed_pairs = [pair for pair in all_pairs if pair.passed_all_criteria]

# 5. Print information about pairs that passed all criteria
print(f"Found {len(passed_pairs)} cointegrated pairs that passed all criteria")
for pair in passed_pairs:
    print(f"\nPair: {pair.asset1} - {pair.asset2}")
    print(f"Half-life: {pair.half_life:.2f} days")
    print(f"Hurst exponent: {pair.hurst_exponent:.3f}")
    print(f"Hedge ratio: {pair.hedge_ratio:.3f}")

# 6. Create a DataFrame to store backtest results
results_df = pd.DataFrame()

# 7. Run backtest for each pair
for i, pair in enumerate(passed_pairs):
    print(f"\nBacktesting pair {i+1}/{len(passed_pairs)}: {pair.asset1} - {pair.asset2}")
    
    # Get price data for this pair
    pair_data = get_descriptors(
        tickers=[pair.asset1, pair.asset2],
        descriptors='price_close',
        start_date='2023-01-01',
        end_date='2024-03-05'
    )
    
    # Create Engle-Granger portfolio to get the hedge ratio
    eg = EngleGrangerPortfolio()
    res = eg.fit(pair_data, dependent_variable=pair.asset1)
    
    # Create backtester with default configuration
    config = CaldeiraMouraConfig()
    backtester = Backtester(config)
    
    # Run backtest
    backtest_results = backtester.run_backtest(
        price_data=pair_data,
        hedge_ratio=pair.hedge_ratio,
        training_window=252,  # 1 year of data for training
    )
    
    # Extract key metrics
    metrics = {
        'pair': f"{pair.asset1}_{pair.asset2}",
        'half_life': pair.half_life,
        'hurst': pair.hurst_exponent,
        'hedge_ratio': pair.hedge_ratio,
        'annualized_return': backtest_results['portfolio_stats']['annualized_return'],
        'annualized_volatility': backtest_results['portfolio_stats']['annualized_volatility'],
        'sharpe_ratio': backtest_results['portfolio_stats']['sharpe_ratio'],
        'max_drawdown': backtest_results['portfolio_stats']['max_drawdown'],
        'num_trades': backtest_results['portfolio_stats']['num_trades'],
        'win_rate': backtest_results['portfolio_stats']['win_rate']
    }
    
    # Add to results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Plot equity curve
    if 'equity_curve' in backtest_results and not backtest_results['equity_curve'].empty:
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results['equity_curve']['equity'])
        plt.title(f"Equity Curve: {pair.asset1} - {pair.asset2}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"equity_curve_{pair.asset1}_{pair.asset2}.png")
        plt.close()

# 8. Calculate Sharpe ratio and sort results
results_df = results_df.eval('sharpe_ratio = (annualized_return - 0.11) / annualized_volatility')
results_df = results_df.sort_values('sharpe_ratio', ascending=False)

# 9. Display top 20 pairs by Sharpe ratio
print("\nTop 20 pairs by Sharpe ratio:")
print(results_df[['pair', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate']].head(20))

# 10. Run rolling window backtest for the top pair
if not results_df.empty:
    best_pair = results_df.iloc[0]['pair'].split('_')
    print(f"\nRunning rolling window backtest for best pair: {best_pair[0]} - {best_pair[1]}")
    
    # Get price data for best pair
    best_pair_data = get_descriptors(
        tickers=best_pair,
        descriptors='price_close',
        start_date='2022-01-01',  # Get more historical data for rolling windows
        end_date='2024-01-01'
    )
    
    # Create backtester
    backtester = Backtester(config)
    
    # Run rolling window backtest
    rolling_results = backtester.run_rolling_backtest(
        price_data=best_pair_data,
        hedge_ratio=results_df.iloc[0]['hedge_ratio'],
        training_window=252,  # 1 year of data for training
        testing_window=63,    # ~3 months for testing
        step_size=21,         # Move forward ~1 month at a time
        dependent_variable=best_pair[0]
    )
    
    # Print rolling window results
    print("\nRolling Window Backtest Results:")
    print(f"Pair: {rolling_results['pair']}")
    print(f"Number of windows: {rolling_results['n_windows']}")
    print(f"Total trades: {rolling_results['total_trades']}")
    print(f"Win rate: {rolling_results['win_rate']:.2%}")
    print(f"Annualized return: {rolling_results['annualized_return']:.2%}")
    print(f"Annualized volatility: {rolling_results['annualized_volatility']:.2%}")
    print(f"Sharpe ratio: {rolling_results['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {rolling_results['max_drawdown']:.2%}")
    
    # Plot combined equity curve from rolling windows
    if 'equity_curve' in rolling_results and not rolling_results['equity_curve'].empty:
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_results['equity_curve']['equity'])
        plt.title(f"Rolling Window Equity Curve: {best_pair[0]} - {best_pair[1]}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"rolling_equity_curve_{best_pair[0]}_{best_pair[1]}.png")
        plt.close() 