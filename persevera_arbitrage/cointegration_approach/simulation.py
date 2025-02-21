from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from .engle_granger import EngleGrangerPortfolio

class CointegrationSimulation:
    """
    Bootstrap simulation of cointegrated price series using estimated parameters.
    
    The simulation process:
    1. Uses actual estimated hedge ratio from portfolio
    2. Preserves mean-reversion characteristics of the spread
    3. Resamples spread innovations from historical data
    4. Maintains the cointegration relationship
    
    The process follows:
    1. Extract historical spread and estimate AR(1) coefficient
    2. Calculate historical spread innovations
    3. Simulate new spread using historical innovations
    4. Generate price series maintaining cointegration relationship
    """
    
    def __init__(self,
                 portfolio: EngleGrangerPortfolio,
                 n_periods: int = 1000,
                 return_mean: float = 0.0001,  # Mean daily return
                 return_vol: float = 0.01,     # Return volatility
                 random_seed: Optional[int] = None):
        """
        Initialize bootstrap simulation.
        
        Args:
            portfolio: Fitted EngleGrangerPortfolio instance with estimated parameters
            n_periods: Number of periods to simulate
            return_mean: Mean of returns process
            return_vol: Volatility of returns innovations
            random_seed: Random seed for reproducibility
        """
        if portfolio.test_results is None:
            raise ValueError("Portfolio must be fitted before simulation")
            
        self.portfolio = portfolio
        self.n_periods = n_periods
        self.return_mean = return_mean
        self.return_vol = return_vol
        
        # Extract parameters from portfolio
        self.price_data = portfolio.price_data
        self.hedge_ratio = portfolio.hedge_ratios[portfolio.price_data.columns[1]]
        self.historical_spread = portfolio.test_results.residuals
        
        # Estimate AR(1) coefficient from historical spread
        self.phi_spread = self._estimate_ar_coefficient()
        
        # Calculate historical innovations
        self.spread_innovations = self._get_spread_innovations()
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _estimate_ar_coefficient(self) -> float:
        """Estimate AR(1) coefficient from historical spread."""
        spread_lag = self.historical_spread.shift(1).dropna()
        spread = self.historical_spread[1:]
        
        # Simple OLS regression
        beta = np.cov(spread, spread_lag)[0,1] / np.var(spread_lag)
        return min(abs(beta), 0.9999)  # Ensure stationarity
    
    def _get_spread_innovations(self) -> np.ndarray:
        """Extract innovations from historical spread."""
        spread = self.historical_spread.values
        innovations = np.zeros(len(spread)-1)
        
        for t in range(1, len(spread)):
            innovations[t-1] = spread[t] - self.phi_spread * spread[t-1]
            
        return innovations
            
    def simulate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate cointegrated price series using bootstrap.
        
        Returns:
            Tuple containing:
            - DataFrame with simulated price series
            - Series with simulated spread
        """
        # 1. Simulate returns using AR(1) process
        returns = np.zeros(self.n_periods)
        returns[0] = np.random.normal(self.return_mean, self.return_vol)
        
        for t in range(1, self.n_periods):
            returns[t] = (self.return_mean + 
                         np.random.normal(0, self.return_vol))
            
        # 2. Get first price series by cumulating returns
        p1 = np.zeros(self.n_periods)
        p1[0] = self.price_data.iloc[-1].iloc[1]  # Use last price from price_data
        
        for t in range(1, self.n_periods):
            p1[t] = p1[t-1] * np.exp(returns[t])
            
        # 3. Simulate spread using bootstrapped innovations
        spread = np.zeros(self.n_periods)
        spread[0] = self.historical_spread.iloc[-1]  # Start from last historical spread
        
        # Resample innovations with replacement
        sampled_innovations = np.random.choice(
            self.spread_innovations,
            size=self.n_periods-1,
            replace=True
        )
        
        for t in range(1, self.n_periods):
            spread[t] = (self.phi_spread * spread[t-1] + 
                        sampled_innovations[t-1])
            
        # 4. Get second price using cointegration relation and hedge ratio
        p2 = p1 * self.hedge_ratio + spread
        
        # Create DataFrame with prices
        dates = pd.bdate_range(
            start=self.price_data.iloc[-1].name,
            periods=self.n_periods,
            freq='D'
        )
        
        prices = pd.DataFrame({
            self.price_data.columns[0]: p2,
            self.price_data.columns[1]: p1
        }, index=dates)
        
        return prices, pd.Series(spread, index=dates)
    
    def simulate_multiple_paths(self, n_paths: int) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Generate multiple cointegrated price series paths.
        
        Args:
            n_paths: Number of paths to simulate
            
        Returns:
            List of tuples containing:
            - DataFrame with simulated price series for each path
            - Series with simulated spread for each path
        """
        return [self.simulate() for _ in range(n_paths)]
    
    def get_estimated_parameters(self) -> dict:
        """Get parameters estimated from historical data."""
        return {
            'hedge_ratio': self.hedge_ratio,
            'phi_spread': self.phi_spread,
            'spread_mean': np.mean(self.historical_spread),
            'spread_std': np.std(self.spread_innovations),
            'half_life': -np.log(2) / np.log(abs(self.phi_spread))
        }
