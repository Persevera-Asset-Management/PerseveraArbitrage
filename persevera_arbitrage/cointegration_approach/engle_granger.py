from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

from .base import CointegratedPortfolio
from .config import CointegrationConfig
from .utils import test_stationarity, calculate_zscore, get_half_life
from .validations import validate_price_data

@dataclass
class EngleGrangerTestResult:
    """Container for Engle-Granger test results."""
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratios: pd.DataFrame
    residuals: pd.Series
    is_cointegrated: bool
    half_life: Optional[float] = None

class EngleGrangerPortfolio(CointegratedPortfolio):
    """
    Enhanced implementation of mean-reverting portfolio construction using the 
    two-step Engle-Granger method with additional validation and analysis features.
    
    The Engle-Granger method tests for cointegration by:
    1. Performing OLS regression on price levels
    2. Testing residuals for stationarity using ADF test
    
    Reference:
    Engle, Robert F and Granger, Clive W J. "Co-integration and Error Correction: 
    Representation, Estimation, and Testing," Econometrica, 55(2), 251-276, March 1987
    """
    
    def __init__(self, config: Optional[CointegrationConfig] = None):
        """Initialize Engle-Granger Portfolio.
        
        Args:
            config: Configuration parameters. If None, uses defaults
        """
        super().__init__()
        self.config = config or CointegrationConfig()
        self.test_results: Optional[EngleGrangerTestResult] = None
        self.dependent_variable: Optional[str] = None
        self.zscore: Optional[pd.Series] = None
        
    def fit(self, 
            price_data: pd.DataFrame, 
            dependent_variable: Optional[str] = None,
            add_constant: bool = False) -> EngleGrangerTestResult:
        """Fit the Engle-Granger cointegration model.
        
        Args:
            price_data: Price data with columns containing asset prices
            dependent_variable: Column to use as dependent variable. 
                              If None, uses first column
            add_constant: Whether to add constant term in regression
            
        Returns:
            EngleGrangerTestResult containing test statistics and results
            
        Raises:
            ValueError: If price data contains invalid values or insufficient history
        """
        # Validate input data
        validate_price_data(price_data, self.config.min_history)
        
        # Store parameters
        self.price_data = price_data.copy()
        self.dependent_variable = dependent_variable or price_data.columns[0]
        
        # Perform OLS regression
        hedge_ratios, X, y, residuals = self._get_ols_hedge_ratios(
            price_data=price_data,
            dependent_variable=self.dependent_variable,
            add_constant=add_constant
        )
        
        # Store cointegration vectors
        self._store_cointegration_vectors(hedge_ratios)
        
        # Store hedge ratios to instance variable
        self.hedge_ratios = hedge_ratios
        
        # Perform Engle-Granger test on residuals
        test_result = self._perform_engle_granger_test(residuals)
        
        # Calculate additional metrics
        half_life = get_half_life(residuals)
        
        # Store results
        self.test_results = EngleGrangerTestResult(
            adf_statistic=test_result[0],
            p_value=test_result[1],
            critical_values=test_result[4],
            hedge_ratios=self.hedge_ratios,
            residuals=residuals,
            is_cointegrated=test_result[1] < self.config.significance_level,
            half_life=half_life
        )
        
        # Calculate z-score for trading signals
        self.zscore = calculate_zscore(residuals)
        
        return self.test_results

    def _get_ols_hedge_ratios(self,
                              price_data: pd.DataFrame,
                              dependent_variable: str,
                              add_constant: bool = False) -> Tuple[Dict[str, float], pd.DataFrame, pd.Series, pd.Series]:
        """Get OLS hedge ratios using linear regression.
        
        Args:
            price_data: Price data for regression
            dependent_variable: Column to use as dependent variable
            add_constant: Whether to add constant term
            
        Returns:
            Tuple containing:
            - Dictionary mapping asset names to hedge ratios
            - X matrix used in regression
            - y vector used in regression
            - Residuals from regression
        """
        # Prepare regression data
        X = price_data.drop(columns=dependent_variable)
        exogenous_variables = X.columns.tolist()
        
        # Reshape X if single variable
        if X.shape[1] == 1:
            X = X.values.reshape(-1, 1)
            
        y = price_data[dependent_variable]
        
        # Fit regression
        model = LinearRegression(fit_intercept=add_constant)
        model.fit(X, y)
        
        # Calculate residuals
        residuals = y - model.predict(X)
        
        # Create hedge ratios dictionary including intercept if used
        hedge_ratios_dict = {dependent_variable: 1.0}
        if add_constant and model.intercept_ != 0:
            hedge_ratios_dict['constant'] = model.intercept_
        hedge_ratios_dict.update(dict(zip(exogenous_variables, model.coef_)))
        
        return hedge_ratios_dict, X, y, residuals

    def _store_cointegration_vectors(self, hedge_ratios: dict) -> None:
        """Store cointegration vectors and hedge ratios.
        
        Args:
            hedge_ratios: Dictionary of hedge ratios from regression
        """
        # Extract ratios for non-dependent variables
        other_ratios = [
            hedge_ratios[ticker] 
            for ticker in self.price_data.columns 
            if ticker != self.dependent_variable
        ]
        
        # Store cointegration vector [1, -β1, -β2, ...]
        self.cointegration_vectors = pd.DataFrame(
            [np.append(1, -1 * np.array(other_ratios))],
            columns=self.price_data.columns
        )
        
        # Store hedge ratios [1, β1, β2, ...] in a different variable
        self.hedge_ratios_df = pd.DataFrame(
            [np.append(1, np.array(other_ratios))],
            columns=self.price_data.columns
        )

    def _perform_engle_granger_test(self, residuals: pd.Series) -> Tuple:
        """Perform Engle-Granger test on residuals.
        
        Args:
            residuals: OLS regression residuals
            
        Returns:
            ADF test results tuple containing:
            - ADF test statistic
            - p-value
            - number of lags used
            - number of observations
            - dictionary of critical values
            - dictionary of IC values if autolag is used
        """
        return adfuller(
            residuals,
            regression='c',     # Include constant in test regression
            maxlag=None,        # Let adfuller determine max lags based on sample size
            autolag='AIC',      # Use Akaike Information Criterion for lag selection
            regresults=False    # Don't return full regression results
        )

    def is_cointegrated(self, significance: Optional[float] = None) -> bool:
        """Check if assets are cointegrated at given significance level.
        
        Args:
            significance: Significance level. If None, uses config value
            
        Returns:
            True if cointegrated at specified significance
            
        Raises:
            ValueError: If model not yet fit
        """
        if self.test_results is None:
            raise ValueError("Must fit model before checking cointegration")
            
        significance = significance or self.config.significance_level
        return self.test_results.p_value < significance