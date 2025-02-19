# persevera_arbitrage/cointegration/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def plot_portfolio(portfolio_values: pd.Series,
                   zscore: Optional[pd.Series] = None,
                   signals: Optional[pd.Series] = None,
                   thresholds: Tuple[float, float] = (-2.0, 2.0),
                   figsize: tuple = (15, 8)) -> None:
    """Plot portfolio values, z-scores, and signals.
    
    Args:
        portfolio_values: Portfolio value time series
        zscore: Optional z-score series
        signals: Optional trading signals
        thresholds: Z-score thresholds for trading signals (lower, upper)
        figsize: Figure size tuple (width, height)
    """
    try:
        n_plots = 1 + (zscore is not None) + (signals is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot portfolio values
        axes[0].plot(portfolio_values, label='Portfolio Value', color='blue')
        axes[0].set_title('Portfolio Values')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot z-scores if provided
        if zscore is not None:
            idx = 1
            axes[idx].plot(zscore, label='Z-score', color='orange')
            axes[idx].axhline(y=thresholds[1], color='r', linestyle='--', alpha=0.5, 
                            label='Upper Threshold')
            axes[idx].axhline(y=thresholds[0], color='r', linestyle='--', alpha=0.5, 
                            label='Lower Threshold')
            axes[idx].axhline(y=0, color='grey', linestyle='-', alpha=0.3)
            axes[idx].set_title('Z-scores')
            axes[idx].legend()
            axes[idx].grid(True)
            
        # Plot signals if provided
        if signals is not None:
            idx = 2 if zscore is not None else 1
            axes[idx].plot(signals, label='Signals', color='green', drawstyle='steps-post')
            axes[idx].set_title('Trading Signals')
            axes[idx].set_yticks([-1, 0, 1])
            axes[idx].set_yticklabels(['Sell', 'Hold', 'Buy'])
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    finally:
        # Clean up to prevent memory leaks
        plt.close('all')

def plot_pair_analysis(price1: pd.Series,
                       price2: pd.Series,
                       title: Optional[str] = None,
                       figsize: tuple = (15, 10)) -> None:
    """Plot pair analysis including prices and scatter.
    
    Args:
        price1: First asset prices
        price2: Second asset prices
        title: Optional title
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot normalized prices
    norm1 = price1 / price1.iloc[0]
    norm2 = price2 / price2.iloc[0]
    
    ax1.plot(norm1, label=price1.name)
    ax1.plot(norm2, label=price2.name)
    ax1.set_title('Normalized Price Series')
    ax1.legend()
    ax1.grid(True)
    
    # Plot scatter with regression
    sns.regplot(x=price1, y=price2, ax=ax2, scatter_kws={'alpha':0.5})
    ax2.set_title('Price Relationship')
    ax2.set_xlabel(price1.name)
    ax2.set_ylabel(price2.name)
    ax2.grid(True)
    
    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    
    plt.tight_layout()
    plt.show()

def plot_cointegration_test(test_stats: pd.DataFrame, title: str, figsize: tuple = (10, 6)) -> None:
    """Plot cointegration test statistics.
    
    Args:
        test_stats: DataFrame with test statistics
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Plot test statistics
    bar_width = 0.35
    indices = np.arange(len(test_stats.columns))
    
    plt.bar(indices, test_stats.loc['eigen_value'], bar_width, 
            label='Test Statistic', color='blue', alpha=0.6)
    
    plt.bar(indices + bar_width, test_stats.loc['95%'], bar_width,
            label='95% Critical Value', color='red', alpha=0.6)
    
    plt.xlabel('Number of Cointegration Relations')
    plt.ylabel('Test Statistic')
    plt.title(title)
    plt.xticks(indices + bar_width/2, test_stats.columns)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_hedge_ratios(hedge_ratios: pd.DataFrame, figsize: tuple = (12, 6)) -> None:
    """Plot hedge ratios for each asset.
    
    Args:
        hedge_ratios: DataFrame with hedge ratios
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Create bar plot
    sns.barplot(data=hedge_ratios.T)
    
    plt.title('Portfolio Hedge Ratios')
    plt.xlabel('Asset')
    plt.ylabel('Ratio')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()