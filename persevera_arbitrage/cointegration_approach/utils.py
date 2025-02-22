import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

def get_half_life(data: pd.Series) -> float:
    """Calculate half-life of mean reversion for a time series.
    
    Args:
        data: Time series data
        
    Returns:
        Half-life of mean reversion in periods. Returns np.inf if no mean reversion.
        
    Notes:
        Uses linear regression on lagged values to estimate mean reversion speed.
    """
    # Calculate lagged data and delta
    lag = data.shift(1).dropna()
    delta = data.diff().dropna()
    
    # Fit linear regression
    reg = LinearRegression(fit_intercept=True)
    reg.fit(lag.values.reshape(-1, 1), delta.values)
    
    # Handle case where there is no mean reversion
    if reg.coef_[0] >= 0:  # Changed from == to >= for numerical stability
        return np.inf
    
    # Calculate half-life
    half_life = -np.log(2) / reg.coef_[0]
    return half_life

def test_stationarity(data: pd.Series,
                      significance: float = 0.05,
                      regression: str = 'c') -> Tuple[bool, float, dict]:
    """Test time series for stationarity using ADF test.
    
    Args:
        data: Time series to test
        significance: Significance level for test
        regression: Type of regression to use in test ('c', 'n', 'ct', 'ctt')
        
    Returns:
        Tuple containing:
        - bool: True if series is stationary
        - float: Test p-value
        - dict: Critical values at different significance levels
    """
    adf_result = adfuller(data, regression=regression)
    pvalue = adf_result[1]
    return pvalue < significance, pvalue

def calculate_zscore(spread: pd.Series,
                     lookback: Optional[int] = None,
                     min_periods: Optional[int] = None) -> pd.Series:
    """Calculate z-score of a spread series.
    
    Args:
        spread: Spread time series
        lookback: Optional rolling window. If None, uses full series
        min_periods: Minimum periods for rolling calculation. Defaults to lookback/2
        
    Returns:
        Z-score series
        
    Notes:
        Handles initial lookback period by forward-filling NaN values
    """
    if lookback:
        # Set minimum periods if not specified
        if min_periods is None:
            min_periods = lookback // 2
            
        # Calculate rolling stats
        mean = spread.rolling(window=lookback, min_periods=min_periods).mean()
        std = spread.rolling(window=lookback, min_periods=min_periods).std()
        
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

def get_hurst_exponent(data: Union[pd.Series, np.ndarray],
                       max_lags: int = 100,
                       min_lags: int = 2) -> float:
    """Calculate Hurst exponent to measure mean reversion strength.
    
    Args:
        data: Time series to analyze (must be numeric)
        max_lags: Maximum number of lags to consider
        min_lags: Minimum number of lags to consider
        
    Returns:
        Hurst exponent (0-0.5 indicates mean reversion)
        
    Notes:
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
    Raises:
        ValueError: If data contains non-numeric values or is too short
    """
    # Convert to numpy array if needed
    if isinstance(data, pd.Series):
        data = data.values
    
    # Ensure data is numeric
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must be numeric")
        
    # Check data length
    if len(data) < max_lags:
        raise ValueError(f"Data length ({len(data)}) must be greater than max_lags ({max_lags})")
    
    # Validate inputs
    if min_lags < 2:
        raise ValueError("min_lags must be at least 2")
    if max_lags <= min_lags:
        raise ValueError("max_lags must be greater than min_lags")
        
    # Remove any NaN values
    data = data[~np.isnan(data)]
    if len(data) < max_lags:
        raise ValueError("Insufficient non-NaN data points")
        
    lags = np.arange(min_lags, max_lags)
    tau = np.array([np.std(np.subtract(data[lag:], data[:-lag]))
                    for lag in lags])
    
    # Avoid log(0) in case of constant series
    if np.any(tau == 0):
        return np.nan
        
    return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0

def calculate_residuals(data: pd.DataFrame, 
                       dependent_var: str, 
                       hedge_ratio: pd.Series,
                       add_constant: bool = False) -> pd.Series:
    """Calculate residuals from cointegration regression.
    
    Args:
        data: Price data for assets
        dependent_var: Dependent variable column
        hedge_ratio: Hedge ratios for assets
        add_constant: Whether to include constant term
        
    Returns:
        Residual series
    """
    independent_vars = data.columns[data.columns != dependent_var]
    
    if add_constant:
        if 'constant' not in hedge_ratio:
            raise ValueError("Hedge ratio missing constant term")
        residuals = (data[dependent_var] - 
                    (hedge_ratio[independent_vars] * data[independent_vars]).sum(axis=1) - 
                    hedge_ratio['constant'])
    else:
        residuals = (data[dependent_var] - 
                    (hedge_ratio[independent_vars] * data[independent_vars]).sum(axis=1))
    
    return residuals

def check_pair_correlation(price1: pd.Series, 
                          price2: pd.Series, 
                          threshold: float = 0.7,
                          method: str = 'pearson') -> Tuple[bool, float]:
    """Check if pair has sufficient correlation.
    
    Args:
        price1: First price series
        price2: Second price series
        threshold: Minimum required correlation
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Tuple containing:
        - bool: True if correlation exceeds threshold
        - float: Correlation coefficient
    """
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
        
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    correlation = returns1.corr(returns2, method=method)
    return abs(correlation) >= threshold, correlation