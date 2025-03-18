from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from dataclasses import dataclass

from .base import CointegratedPortfolio
from .config import CointegrationConfig
from .utils import get_half_life, test_stationarity, calculate_zscore, get_hurst_exponent
from .validations import validate_price_data

@dataclass
class JohansenTestResult:
    """Container for Johansen test results."""
    cointegration_vectors: pd.DataFrame
    hedge_ratios: pd.DataFrame
    eigen_statistic: Optional[pd.DataFrame]  # None if >12 assets
    trace_statistic: Optional[pd.DataFrame]  # None if >12 assets
    residuals: pd.Series
    is_cointegrated: bool
    half_life: Optional[float] = None

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

    def fit(self, price_data: pd.DataFrame) -> JohansenTestResult:
        """Find cointegration vectors using Johansen test.
        
        Args:
            price_data: DataFrame with asset prices as columns
                
        Returns:
            JohansenTestResult containing test statistics and results
            
        Raises:
            ValueError: If price data contains NaN/infinite values
        """
        validate_price_data(price_data, self.config.min_history)

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
        
        # Normalize hedge ratios using vectorized operations
        scaling = self.cointegration_vectors[dependent_var].values
        if np.any(scaling == 0):
            raise ValueError(f"The hedge ratio for the dependent variable {dependent_var} is zero. Cannot normalize.")

        self.hedge_ratios = self.cointegration_vectors.div(scaling, axis=0)
        
        # Store test statistics if we have ≤12 variables
        eigen_stat = None
        trace_stat = None
        if price_data.shape[1] <= 12:
            self._store_test_statistics(test_result, price_data.columns)
            eigen_stat = self.johansen_eigen_statistic
            trace_stat = self.johansen_trace_statistic
            
        # Calculate portfolio and analyze properties
        portfolio = self.construct_mean_reverting_portfolio(price_data)
        
        self.half_life = get_half_life(portfolio)
        
        # Validate portfolio characteristics
        # self._validate_portfolio(portfolio)

        # Create and return test results
        return JohansenTestResult(
            cointegration_vectors=self.cointegration_vectors,
            hedge_ratios=self.hedge_ratios,
            eigen_statistic=eigen_stat,
            trace_statistic=trace_stat,
            residuals=portfolio,
            is_cointegrated=self.is_cointegrated() if eigen_stat is not None else True,
            half_life=self.half_life
        )

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
    