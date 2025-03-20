import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from persevera_tools.data import get_descriptors, get_index_composition, get_equities_info
from persevera_arbitrage.cointegration_approach.pairs_selection import CointegrationPairSelector, PairSelectionCriteria
from persevera_arbitrage.trading.zscore import CaldeiraMouraTradingRule, CaldeiraMouraConfig
from persevera_arbitrage.cointegration_approach.simulation import CointegrationSimulation
from persevera_arbitrage.cointegration_approach import EngleGrangerPortfolio
from persevera_arbitrage.trading import Backtester

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Dictionary with dates in YYYY-MM-DD format
dates = {
  "rebalance_date": [
    "2024-12-02",
    "2024-08-02",
    "2024-04-04",
    "2023-12-01",
    "2023-08-01",
    "2023-03-30",
    "2022-11-29",
    "2022-07-28",
    "2022-03-29",
    "2021-11-25",
    "2021-07-26",
    "2021-03-24",
    "2020-11-16",
    "2020-07-16",
    "2020-03-16"
  ],
  "training_start_date": [
    "2023-12-01",
    "2023-08-01",
    "2023-03-30",
    "2022-11-29",
    "2022-07-28",
    "2022-03-29",
    "2021-11-25",
    "2021-07-26",
    "2021-03-24",
    "2020-11-16",
    "2020-07-16",
    "2020-03-16",
    "2019-11-07",
    "2019-07-12",
    "2019-03-12"
  ],
  "testing_end_date": [
    "2025-03-14",
    "2024-12-02",
    "2024-08-02",
    "2024-04-04",
    "2023-12-01",
    "2023-08-01",
    "2023-03-30",
    "2022-11-29",
    "2022-07-28",
    "2022-03-29",
    "2021-11-25",
    "2021-07-26",
    "2021-03-24",
    "2020-11-16",
    "2020-07-16"
  ]
}

# Function to plot pair analysis
def plot_pair_analysis(backtest_results, pair, config, period_dir):
    """Generate comprehensive analysis plots for a trading pair"""
    if 'equity_curve' in backtest_results and not backtest_results['equity_curve'].empty:
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 15))
        
        # Asset Prices
        plt.subplot(4, 1, 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(backtest_results['equity_curve']['asset1_price'], 'b-', label=pair.asset1)
        ax2.plot(backtest_results['equity_curve']['asset2_price'], 'r-', label=pair.asset2)
        ax1.set_ylabel(f'{pair.asset1} Price', color='b')
        ax2.set_ylabel(f'{pair.asset2} Price', color='r')
        plt.title(f"Asset Prices: {pair.asset1} - {pair.asset2}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.grid(True)
        
        # Spread and Z-score
        plt.subplot(4, 1, 2)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        if 'spread' in backtest_results:
            ax1.plot(backtest_results['spread'], 'b-', label='Spread')
        ax2.plot(backtest_results['equity_curve']['zscore'], 'r-', label='Z-score')
        ax1.set_ylabel('Spread', color='b')
        ax2.set_ylabel('Z-score', color='r')
        # Add horizontal lines for entry/exit thresholds
        ax2.axhline(y=config.entry_threshold, color='g', linestyle='--', alpha=0.7, label='Entry (Long)')
        ax2.axhline(y=-config.entry_threshold, color='g', linestyle='--', alpha=0.7, label='Entry (Short)')
        ax2.axhline(y=config.exit_threshold_short, color='m', linestyle='--', alpha=0.7, label='Exit Short')
        ax2.axhline(y=config.exit_threshold_long, color='m', linestyle='--', alpha=0.7, label='Exit Long')
        plt.title("Spread and Z-score")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.grid(True)
        
        # Equity curve and Positions
        plt.subplot(4, 1, 3)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(backtest_results['equity_curve']['equity'], 'g-', label='Equity')
        ax2.plot(backtest_results['equity_curve']['position'], 'b-', label='Position')
        ax1.set_ylabel('Equity', color='g')
        ax2.set_ylabel('Position Size', color='b')
        ax2.set_ylim(-1.5, 1.5)
        plt.title("Strategy Performance")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True)
        
        # Returns Distribution
        plt.subplot(4, 1, 4)
        returns = backtest_results['equity_curve']['equity'].pct_change().dropna()
        plt.hist(returns, bins=50, density=True, alpha=0.75)
        plt.title(f"Returns Distribution: {pair.asset1} - {pair.asset2}")
        plt.xlabel("Returns")
        plt.ylabel("Density")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(period_dir, f"analysis_{pair.asset1}_{pair.asset2}.png"))
        plt.close()

# Loop through each date period
all_results = []
all_returns = []

for i in range(len(dates['rebalance_date'])):
    print(f"\n{'='*80}")
    print(f"Processing Period {i+1}/{len(dates['rebalance_date'])}")
    print(f"Rebalance Date: {dates['rebalance_date'][i]}")
    print(f"Training Start: {dates['training_start'][i]}")
    print(f"Testing End: {dates['testing_end'][i]}")
    print(f"{'='*80}")
    
    # Create a directory for this period's results
    period_dir = os.path.join('results', f"period_{i+1}_{dates['rebalance_date'][i]}")
    os.makedirs(period_dir, exist_ok=True)
    
    try:
        # Get IBX100 composition
        members = get_index_composition(index_code='IBX100', 
                                       start_date=dates['rebalance_date'][i], 
                                       end_date=dates['rebalance_date'][i])
        
        # Load price data
        print("Loading price data...")
        price_data = get_descriptors(
            tickers=[*members.columns],
            descriptors='price_close',
            start_date=dates['training_start'][i],
            end_date=dates['rebalance_date'][i]
        )
        price_data.dropna(axis=1, inplace=True)
        
        # Find cointegrated pairs
        print("Finding cointegrated pairs...")
        selector = CointegrationPairSelector()
        all_pairs = selector.select_pairs(price_data)
        
        # Convert to DataFrame
        results = pd.DataFrame(all_pairs)
        results_passed = results[results['passed_all_criteria']].reset_index(drop=True)
        
        # Save pairs that passed all criteria
        results_passed.to_csv(os.path.join(period_dir, 'cointegrated_pairs.csv'))
        
        print(f"Found {len(results_passed)} cointegrated pairs")
        
        # Simulate trading performance
        print("Simulating trading performance...")
        df = pd.DataFrame()
        
        # Run simulations for each pair
        for r in range(len(results_passed)):
            if r % 10 == 0:
                print(f"  Simulating pair {r+1}/{len(results_passed)}")
                
            tickers = [results_passed.iloc[r].asset1, results_passed.iloc[r].asset2]
            pair_price_data = get_descriptors(
                tickers=tickers,
                descriptors='price_close',
                start_date=dates['training_start'][i],
                end_date=dates['rebalance_date'][i]
            )
            
            eg = EngleGrangerPortfolio()
            res = eg.fit(pair_price_data, dependent_variable=results_passed.iloc[r].asset1)
            historical_spread = res.residuals
            
            simulation = CointegrationSimulation(eg, n_periods=21*3, random_seed=40)
            sim = simulation.simulate_multiple_paths(n_paths=100)
            
            for s in range(len(sim)):
                simulated_spread = sim[s][1]
                simulated_prices = sim[s][0]
                
                config = CaldeiraMouraConfig(verbose=False)
                trading_rule = CaldeiraMouraTradingRule(config)
                
                signals, position_sizes = trading_rule.generate_signals(
                    simulated_spread=simulated_spread,
                    historical_spread=historical_spread,
                    prices=simulated_prices,
                )
                
                simulation_results = trading_rule.get_portfolio_stats()
                simulation_results_df = pd.DataFrame(simulation_results, index=[0])
                simulation_results_df.insert(0, 'pair', '_'.join(tickers))
                df = pd.concat([df, simulation_results_df], axis=0, ignore_index=True)
        
        # Save simulation results
        df.to_csv(os.path.join(period_dir, 'simulation_results.csv'))
        
        # Select top 20 pairs based on Sharpe ratio
        selected_pairs = (df
                          .groupby('pair')
                          .agg({'annualized_return': 'median', 'annualized_volatility': 'median'})
                          .eval('sharpe_ratio = (annualized_return - 0.11) / annualized_volatility')
                          .query('sharpe_ratio > 0')
                          .sort_values('sharpe_ratio', ascending=False)
                          .head(20)
                        )
        
        selected_pairs['asset1'] = selected_pairs.index.str.split('_').str[0]
        selected_pairs['asset2'] = selected_pairs.index.str.split('_').str[1]
        
        # Save selected pairs
        selected_pairs.to_csv(os.path.join(period_dir, 'selected_pairs.csv'))
        
        print(f"Selected {len(selected_pairs)} best pairs")
        
        # Run backtests
        print("Running backtests...")
        results_df = pd.DataFrame()
        returns_df = pd.DataFrame()
        
        # Run backtest for each pair
        counter = 1
        for ix, pair in selected_pairs.iterrows():
            print(f"  Backtesting pair {counter}/{len(selected_pairs)}: {pair.asset1} - {pair.asset2}")
            
            testing_data = results_passed[(results_passed.asset1 == pair.asset1) & (results_passed.asset2 == pair.asset2)]
            hedge_ratio = testing_data['hedge_ratio'].values[0]
            half_life = testing_data['half_life'].values[0]
            hurst = testing_data['hurst_exponent'].values[0]
            
            # Get price data for this pair
            pair_data = get_descriptors(
                tickers=[pair.asset1, pair.asset2],
                descriptors='price_close',
                start_date=dates['training_start'][i],
                end_date=dates['testing_end'][i]
            )
            
            # Create backtester with default configuration
            config = CaldeiraMouraConfig()
            backtester = Backtester(config)
            
            # Run backtest
            backtest_results = backtester.run_backtest(
                price_data=pair_data,
                hedge_ratio=hedge_ratio,
                training_window=252,  # 1 year of data for training
            )
            
            # Extract key metrics
            metrics = {
                'pair': f"{pair.asset1}_{pair.asset2}",
                'period': i,
                'rebalance_date': dates['rebalance_date'][i],
                'half_life': half_life,
                'hurst': hurst,
                'hedge_ratio': hedge_ratio,
                'annualized_return': backtest_results['portfolio_stats']['annualized_return'],
                'annualized_volatility': backtest_results['portfolio_stats']['annualized_volatility'],
                'sharpe_ratio': backtest_results['portfolio_stats']['sharpe_ratio'],
                'max_drawdown': backtest_results['portfolio_stats']['max_drawdown'],
                'num_trades': backtest_results['portfolio_stats']['num_trades'],
                'win_rate': backtest_results['portfolio_stats']['win_rate']
            }
            
            # Add to results dataframe
            results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
            
            # Track returns
            if 'return' in backtest_results['equity_curve']:
                returns_df = pd.merge(
                    returns_df, 
                    backtest_results['equity_curve']['return'].to_frame(f"{pair.asset1}_{pair.asset2}"), 
                    left_index=True, right_index=True, 
                    how='outer'
                )
            
            # Generate visualization
            plot_pair_analysis(backtest_results, pair, config, period_dir)
            
            counter += 1
        
        # Save backtest results
        results_df.to_csv(os.path.join(period_dir, 'backtest_results.csv'))
        returns_df.to_csv(os.path.join(period_dir, 'returns.csv'))
        
        # Plot aggregate returns
        if not returns_df.empty:
            plt.figure(figsize=(12, 8))
            ((1 + returns_df.mean(axis=1)).cumprod() - 1).plot()
            plt.title(f"Aggregate Returns - Period {i+1} ({dates['rebalance_date'][i]})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(period_dir, 'aggregate_returns.png'))
            plt.close()
        
        # Add results to overall results
        all_results.append(results_df)
        all_returns.append(returns_df)
        
        print(f"Period {i+1} processing complete.")
        
    except Exception as e:
        print(f"Error processing period {i+1}: {str(e)}")
        continue

# Combine all results and save
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv('results/all_periods_results.csv')
    
    # Summary by period
    period_summary = combined_results.groupby('period').agg({
        'rebalance_date': 'first',
        'annualized_return': 'mean',
        'annualized_volatility': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'num_trades': 'mean',
        'win_rate': 'mean'
    }).reset_index()
    
    period_summary.to_csv('results/period_summary.csv')
    
    print("\nAll periods processed. Results saved to 'results' directory.")
    print("Period summary:")
    print(period_summary)
else:
    print("\nNo results were generated for any period.") 