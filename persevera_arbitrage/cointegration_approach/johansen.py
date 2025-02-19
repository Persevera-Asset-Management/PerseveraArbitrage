from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .base import CointegratedPortfolio
from .config import CointegrationConfig
from .utils import get_half_life, test_stationarity, calculate_zscore, get_hurst_exponent, check_pair_correlation

class JohansenPortfolio(CointegratedPortfolio):
    """Class for portfolio construction using Johansen cointegration test.
    
    The class implements portfolio construction using eigenvectors from 
    the Johansen cointegration test. It includes both eigenvalue and 
    trace statistics tests for cointegration detection.
    """
    
    def __init__(self, config: Optional[CointegrationConfig] = None):
        """Initialize Johansen Portfolio.
        
        Args:
            config: Configuration parameters. If None, uses defaults
        """
        super().__init__()
        self.config = config or CointegrationConfig()
        self.johansen_trace_statistic = None
        self.johansen_eigen_statistic = None
        self.half_life = None
        self.residuals = None  
        self.zscore = None
        
    def fit(self, price_data: pd.DataFrame) -> None:
        """Find cointegration vectors using Johansen test.
        
        Args:
            price_data: DataFrame with asset prices as columns
                
        Raises:
            ValueError: If price data contains NaN/infinite values
        """
        # Validate input data
        if price_data.isnull().any().any():
            raise ValueError("Price data contains NaN values")
        if np.isinf(price_data).any().any():
            raise ValueError("Price data contains infinite values")
        
        # Verify correlations between pairs
        if not self._verify_correlations(price_data):
            print("Warning: Some pairs show low correlation")

        # Set dependent variable if not specified
        dependent_var = self.config.dependent_variable or price_data.columns[0]
        
        self.price_data = price_data.copy()  # Create copy to prevent modifications
        
        # Run Johansen test
        test_result = coint_johansen(
            price_data,
            det_order=self.config.det_order,
            k_ar_diff=self.config.n_lags
        )
        
        # Store eigenvectors ordered by eigenvalues
        self.cointegration_vectors = pd.DataFrame(
            test_result.evec[:, test_result.ind].T,
            columns=price_data.columns
        )
        
        # Calculate hedge ratios normalized to dependent variable
        all_hedge_ratios = pd.DataFrame()

        for vector_idx in range(self.cointegration_vectors.shape[0]):
            # Get current vector
            hedge_ratios = self.cointegration_vectors.iloc[vector_idx].to_dict()
            
            # Normalize ratios (just divide by first element's ratio)
            scaling = hedge_ratios[dependent_var]
            if scaling == 0:
                raise ValueError(f"The hedge ratio for the dependent variable {dependent_var} is zero. Cannot normalize.")
            
            for ticker in hedge_ratios:
                hedge_ratios[ticker] = hedge_ratios[ticker] / scaling
            
            # Combine all ratios
            all_hedge_ratios = pd.concat([
                all_hedge_ratios,
                pd.DataFrame(hedge_ratios)
            ])
                
        self.hedge_ratios = all_hedge_ratios
        
        # Store test statistics if we have ≤12 variables
        if price_data.shape[1] <= 12:
            self._store_test_statistics(test_result, price_data.columns)
            
        # Calculate portfolio and analyze properties
        portfolio = self.construct_mean_reverting_portfolio(price_data)
        
        self.half_life = get_half_life(portfolio)
        self.zscore = calculate_zscore(portfolio)
        
        # Validate portfolio characteristics
        self._validate_portfolio(portfolio)

    def _verify_correlations(self, price_data: pd.DataFrame, threshold: float = 0.7) -> bool:
        """Verify correlations between all pairs."""
        assets = price_data.columns
        all_correlated = True
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                if not check_pair_correlation(
                    price_data[assets[i]], 
                    price_data[assets[j]], 
                    threshold
                ):
                    all_correlated = False
                    print(f"Warning: Low correlation between {assets[i]} and {assets[j]}")
        
        return all_correlated

    def _validate_portfolio(self, portfolio: pd.Series) -> None:
        """Validate portfolio characteristics.
        
        Args:
            portfolio: Portfolio time series to validate
        """
        # Check stationarity
        is_stationary, pvalue = test_stationarity(portfolio)
        if not is_stationary:
            print(f"Warning: Portfolio may not be stationary (p-value: {pvalue})")
            
        # Check mean reversion strength
        hurst = get_hurst_exponent(portfolio)
        if hurst > 0.5:
            print(f"Warning: Portfolio shows weak mean reversion (Hurst: {hurst})")
            
        # Check half-life is reasonable
        if self.half_life > self.config.min_history/2:
            print(f"Warning: Long half-life detected: {self.half_life:.1f} periods")
   
    def _store_test_statistics(self, test_result, columns) -> None:
            """Store eigenvalue and trace test statistics.
            
            Args:
                test_result: Results from Johansen test
                columns: Column names for statistics DataFrame
            """
            # Eigenvalue test statistics
            self.johansen_eigen_statistic = pd.DataFrame(
                test_result.max_eig_stat_crit_vals.T,
                columns=columns,
                index=['90%', '95%', '99%']
            )
            self.johansen_eigen_statistic.loc['eigen_value'] = test_result.max_eig_stat.T
            self.johansen_eigen_statistic.sort_index(ascending=False, inplace=True)
            
            # Trace test statistics
            self.johansen_trace_statistic = pd.DataFrame(
                test_result.trace_stat_crit_vals.T,
                columns=columns,
                index=['90%', '95%', '99%']
            )
            self.johansen_trace_statistic.loc['trace_statistic'] = test_result.trace_stat.T
            self.johansen_trace_statistic.sort_index(ascending=False, inplace=True)
    
    def is_cointegrated(self, significance: str = '95%') -> bool:
        """Check if assets are cointegrated at given significance level.
        
        Args:
            significance: Significance level ('90%', '95%', or '99%')
            
        Returns:
            True if cointegrated at specified significance level
            
        Raises:
            ValueError: If test statistics not available (>12 assets)
        """
        if self.johansen_eigen_statistic is None:
            raise ValueError("Test statistics not available for >12 assets")
            
        eigen_stats = self.johansen_eigen_statistic
        trace_stats = self.johansen_trace_statistic
        
        eigen_test = (eigen_stats.loc['eigen_value'] > eigen_stats.loc[significance]).any()
        trace_test = (trace_stats.loc['trace_statistic'] > trace_stats.loc[significance]).any()
        
        return eigen_test and trace_test
    
    def get_position_sizes(self, position_size: Optional[float] = None, vector_index: int = 0) -> pd.Series:
        """Get position sizes for trading.
        
        Args:
            position_size: Base position size. If None, uses config value
            vector_index: Which cointegration vector to use (default=0)
            
        Returns:
            Series with position sizes for each asset
        """
        position_size = position_size or self.config.position_size
        vector = self.cointegration_vectors.iloc[vector_index]
        return vector * position_size
        
    def get_trading_signals(self, zscore_threshold: float = 2.0) -> pd.Series:
        """Generate trading signals based on z-score.
        
        Args:
            zscore_threshold: Z-score threshold for generating signals
            
        Returns:
            Series with trading signals:
                1: Long signal (z-score < -threshold)
                -1: Short signal (z-score > threshold)
                0: No signal
                
        Raises:
            ValueError: If portfolio has not been fit
        """
        if self.zscore is None:
            raise ValueError("Must fit portfolio before generating signals")
            
        signals = pd.Series(0, index=self.zscore.index)
        signals[self.zscore > zscore_threshold] = -1  # Sell signal
        signals[self.zscore < -zscore_threshold] = 1  # Buy signal
        return signals