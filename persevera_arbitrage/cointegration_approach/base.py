from abc import ABC
import pandas as pd
from typing import Optional, Tuple

class CointegratedPortfolio(ABC):
    """Base class for portfolios formed using cointegration methods."""
    
    def __init__(self):
        """Initialize the base class."""
        self.price_data = None  # Price data used to fit the model
        self.cointegration_vectors = None  # Vectors used for mean-reverting portfolios
        self.hedge_ratios = None  # Hedge ratios for trading
        
    def construct_mean_reverting_portfolio(self, 
                                         price_data: pd.DataFrame,
                                         cointegration_vector: Optional[pd.Series] = None) -> pd.Series:
        """Construct mean-reverting portfolio from price data and cointegration vector.
        
        Args:
            price_data: Price data with columns containing asset prices
            cointegration_vector: Vector used to form mean-reverting portfolio.
                If None, uses vector with maximum eigenvalue from fit()
        
        Returns:
            Mean-reverting portfolio series
        """
        if cointegration_vector is None:
            cointegration_vector = self.cointegration_vectors.iloc[0]
            
        return (cointegration_vector * price_data).sum(axis=1)
    
    def get_scaled_cointegration_vector(self, 
                                      cointegration_vector: Optional[pd.Series] = None) -> pd.Series:
        """Get scaled values of cointegration vector in terms of position sizes.
        
        Args:
            cointegration_vector: Vector to scale. If None, uses first vector from fit()
            
        Returns:
            Scaled cointegration vector showing position sizes
        """
        if cointegration_vector is None:
            cointegration_vector = self.cointegration_vectors.iloc[0]
            
        scaling_coefficient = 1 / cointegration_vector.iloc[0]
        return cointegration_vector * scaling_coefficient