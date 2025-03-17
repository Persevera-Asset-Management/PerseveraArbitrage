import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from persevera_tools.data import get_descriptors, get_index_composition
from persevera_arbitrage.cointegration_approach.pairs_selection import CointegrationPairSelector
from persevera_arbitrage.trading import CaldeiraMouraConfig, Backtester
from persevera_arbitrage.cointegration_approach import EngleGrangerPortfolio

# 1. Get index composition
members = get_index_composition(index_code='IBX50', start_date='2024-01-05', end_date='2024-01-05')

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
for pair in passed_pairs[:5]:  # Show top 5 pairs
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
        end_date='2024-01-01'
    )
    
    # Create backtester with default configuration
    config = CaldeiraMouraConfig()
    backtester = Backtester(config)
    
    # Run backtest
    backtest_results = backtester.run_backtest(
        price_data=pair_data,
        hedge_ratio=pair.hedge_ratio,
        training_window=252,  # 1 year of data for training
        dependent_variable=pair.asset1
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
results_df = results_df.eval('sharpe_ratio_adj = (annualized_return - 0.11) / annualized_volatility')
results_df = results_df.sort_values('sharpe_ratio_adj', ascending=False)

# 9. Display top pairs by Sharpe ratio
print("\nTop pairs by Sharpe ratio:")
print(results_df[['pair', 'annualized_return', 'annualized_volatility', 'sharpe_ratio_adj', 'max_drawdown', 'num_trades', 'win_rate']].head(10))

# 10. Analyze the best pair in more detail
if not results_df.empty:
    best_pair_name = results_df.iloc[0]['pair']
    best_pair_idx = next(i for i, pair in enumerate(passed_pairs) 
                         if f"{pair.asset1}_{pair.asset2}" == best_pair_name)
    best_pair = passed_pairs[best_pair_idx]
    
    print(f"\nDetailed analysis of best pair: {best_pair.asset1} - {best_pair.asset2}")
    print(f"Half-life: {best_pair.half_life:.2f} days")
    print(f"Hurst exponent: {best_pair.hurst_exponent:.3f}")
    print(f"Hedge ratio: {best_pair.hedge_ratio:.3f}")
    
    # Get trade history for best pair
    best_pair_results = None
    for i, pair in enumerate(passed_pairs):
        if f"{pair.asset1}_{pair.asset2}" == best_pair_name:
            # Get price data for this pair
            pair_data = get_descriptors(
                tickers=[pair.asset1, pair.asset2],
                descriptors='price_close',
                start_date='2023-01-01',
                end_date='2024-01-01'
            )
            
            # Create backtester with default configuration
            config = CaldeiraMouraConfig(verbose=True)  # Enable verbose mode for detailed logging
            backtester = Backtester(config)
            
            # Run backtest
            best_pair_results = backtester.run_backtest(
                price_data=pair_data,
                hedge_ratio=pair.hedge_ratio,
                training_window=252,
                dependent_variable=pair.asset1
            )
            break
    
    if best_pair_results and 'trade_history' in best_pair_results:
        trade_history = best_pair_results['trade_history']
        if not trade_history.empty:
            print("\nTrade History:")
            print(trade_history[['entry_date', 'exit_date', 'position_type', 
                                'portfolio_return', 'pnl_amount', 'holding_days', 'exit_reason']])
            
            # Plot trade returns distribution
            plt.figure(figsize=(10, 6))
            plt.hist(trade_history['portfolio_return'], bins=20)
            plt.title(f"Trade Returns Distribution: {best_pair.asset1} - {best_pair.asset2}")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"trade_returns_{best_pair.asset1}_{best_pair.asset2}.png")
            plt.close()
            
            # Plot trade durations
            plt.figure(figsize=(10, 6))
            plt.hist(trade_history['holding_days'], bins=20)
            plt.title(f"Trade Duration Distribution: {best_pair.asset1} - {best_pair.asset2}")
            plt.xlabel("Holding Days")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"trade_durations_{best_pair.asset1}_{best_pair.asset2}.png")
            plt.close()
            
            # Plot cumulative returns
            if 'equity_curve' in best_pair_results and not best_pair_results['equity_curve'].empty:
                equity_curve = best_pair_results['equity_curve']
                plt.figure(figsize=(12, 8))
                
                # Plot equity curve
                plt.subplot(2, 1, 1)
                plt.plot(equity_curve['equity'])
                plt.title(f"Equity Curve: {best_pair.asset1} - {best_pair.asset2}")
                plt.ylabel("Equity")
                plt.grid(True)
                
                # Plot drawdowns
                plt.subplot(2, 1, 2)
                if 'drawdown' in equity_curve.columns:
                    plt.fill_between(equity_curve.index, 0, equity_curve['drawdown'], color='red', alpha=0.3)
                    plt.title("Drawdown")
                    plt.ylabel("Drawdown")
                    plt.ylim(0, equity_curve['drawdown'].max() * 1.1)
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"detailed_equity_{best_pair.asset1}_{best_pair.asset2}.png")
                plt.close() 