import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

def get_half_life(data: pd.Series) -> float:
    """Calculate half-life of mean reversion for a time series.
    
    Args:
        data: Time series data
        
    Returns:
        Half-life of mean reversion
    """
    # Calculate lagged data and delta
    lag = data.shift(1).dropna()
    delta = data.diff().dropna()
    
    # Fit linear regression
    reg = LinearRegression(fit_intercept=True)
    reg.fit(lag.values.reshape(-1, 1), delta.values)
    
    if reg.coef_[0] == 0:
        return np.inf
    
    # Calculate half-life
    half_life = -np.log(2) / reg.coef_[0]
    return half_life

def test_stationarity(data: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
    """Test time series for stationarity using ADF test.
    
    Args:
        data: Time series to test
        significance: Significance level for test
        
    Returns:
        Tuple containing:
        - bool: True if series is stationary
        - float: Test p-value
    """
    adf_result = adfuller(data)
    pvalue = adf_result[1]
    return pvalue < significance, pvalue

def calculate_zscore(spread: pd.Series, lookback: int = None) -> pd.Series:
    """Calculate z-score of a spread series.
    
    Args:
        spread: Spread time series
        lookback: Optional rolling window. If None, uses full series
        
    Returns:
        Z-score series
        
    Notes:
        Handles initial lookback period by forward-filling NaN values
    """
    if lookback:
        # Calculate rolling stats
        mean = spread.rolling(window=lookback, min_periods=1).mean()
        std = spread.rolling(window=lookback, min_periods=1).std()
        
        # Handle edge case where std is zero
        std = std.replace(0, np.nan)
        
        # Calculate z-score
        zscore = (spread - mean) / std
        
        # Forward fill initial NaN values with first valid z-score
        zscore = zscore.fillna(method='bfill')
        
    else:
        # Full series calculation
        mean = spread.mean()
        std = spread.std()
        
        if std == 0 or pd.isna(std):
            return pd.Series(np.nan, index=spread.index)
            
        zscore = (spread - mean) / std
        
    return zscore

def get_hurst_exponent(data: pd.Series, max_lags: int = 100):
    """Calculate Hurst exponent to measure mean reversion strength.
    
    Args:
        data: Time series to analyze
        max_lags: Maximum number of lags to consider
        
    Returns:
        Hurst exponent (0-0.5 indicates mean reversion)
    """
    lags = np.arange(2, max_lags)
    tau = np.array([np.std(np.subtract(data[lag:], data[:-lag]))
                    for lag in lags])
    return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0

def calculate_residuals(data: pd.DataFrame, dependent_var: str, hedge_ratio: pd.Series) -> pd.Series:
    """Calculate residuals from cointegration regression.
    
    Args:
        data: Price data for assets
        dependent_var: Dependent variable column
        hedge_ratio: Hedge ratios for assets
        
    Returns:
        Residual series
    """
    independent_vars = data.columns[data.columns != dependent_var]
    return data[dependent_var] - (hedge_ratio[independent_vars] * data[independent_vars]).sum(axis=1)

def check_pair_correlation(price1: pd.Series, price2: pd.Series, threshold: float = 0.7) -> bool:
    """Check if pair has sufficient correlation.
    
    Args:
        price1: First price series
        price2: Second price series
        threshold: Minimum required correlation
        
    Returns:
        True if correlation exceeds threshold
    """
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    correlation = returns1.corr(returns2)
    return abs(correlation) >= threshold