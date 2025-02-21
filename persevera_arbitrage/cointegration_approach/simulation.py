from typing import Tuple, Optional
import numpy as np
import pandas as pd
from .engle_granger import EngleGrangerPortfolio  # Import the EngleGrangerPortfolio

class CointegrationSimulation:
    """
    Simulate cointegrated price series pairs following Lin (2006) methodology.
    
    The simulation process follows these steps:
    1. Simulate returns of first asset using stationary AR(1) process:
       r1_t = mu + phi_r * r1_{t-1} + eps1_t
       
    2. Get first price series by cumulating returns:
       P1_t = P1_0 * exp(sum(r1_1 to r1_t))
       
    3. Simulate spread using stationary AR(1) process:
       spread_t = phi_s * spread_{t-1} + eps2_t
       
    4. Get second price using cointegration relation:
       P2_t = beta * P1_t + spread_t
    """
    
    def __init__(self,
                 portfolio: EngleGrangerPortfolio,  # Accept EngleGrangerPortfolio instance
                 n_periods: int = 1000,
                 beta: float = 1.0,
                 phi_returns: float = 0.2,    # AR(1) coefficient for returns
                 phi_spread: float = 0.7,     # AR(1) coefficient for spread
                 return_mean: float = 0.0001,  # Mean daily return
                 return_vol: float = 0.01,    # Return volatility
                 spread_vol: float = 0.1,     # Spread volatility
                 random_seed: Optional[int] = None):
        """
        Initialize simulation parameters.
        
        Args:
            portfolio: An instance of EngleGrangerPortfolio containing price data and hedge ratios
            n_periods: Number of periods to simulate
            beta: Cointegration coefficient
            phi_returns: AR(1) coefficient for returns process (0 < phi_r < 1)
            phi_spread: AR(1) coefficient for spread process (0 < phi_s < 1)
            return_mean: Mean of returns process
            return_vol: Volatility of returns innovations
            spread_vol: Volatility of spread innovations
            random_seed: Random seed for reproducibility
        """
        if not (0 <= phi_returns < 1 and 0 <= phi_spread < 1):
            raise ValueError("AR coefficients must be between 0 and 1 for stationarity")
            
        self.portfolio = portfolio  # Store the portfolio instance
        self.n_periods = n_periods
        self.beta = beta
        self.phi_returns = phi_returns
        self.phi_spread = phi_spread
        self.return_mean = return_mean
        self.return_vol = return_vol
        self.spread_vol = spread_vol
        
        # Extract necessary parameters from the portfolio
        self.price_data = portfolio.price_data
        self.hedge_ratios = portfolio.hedge_ratios
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def simulate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate cointegrated price series.
        
        Returns:
            Tuple containing:
            - DataFrame with simulated price series
            - Series with true spread
        """
        # 1. Simulate returns using AR(1) process
        returns = np.zeros(self.n_periods)
        returns[0] = np.random.normal(self.return_mean, self.return_vol)
        
        for t in range(1, self.n_periods):
            returns[t] = (self.return_mean + 
                         self.phi_returns * (returns[t-1] - self.return_mean) +
                         np.random.normal(0, self.return_vol))
            
        # 2. Get first price series by cumulating returns
        p2 = np.zeros(self.n_periods)
        p2[0] = self.price_data.iloc[-1].iloc[1]  # Use last price from price_data
        
        for t in range(1, self.n_periods):
            p2[t] = p2[t-1] * np.exp(returns[t])
            
        # 3. Simulate spread using AR(1) process
        spread = np.zeros(self.n_periods)
        spread[0] = np.random.normal(0, self.spread_vol)
        
        for t in range(1, self.n_periods):
            spread[t] = (self.phi_spread * spread[t-1] + 
                        np.random.normal(0, self.spread_vol))
            
        # 4. Get second price using cointegration relation and hedge ratio
        p1 = p2 * self.hedge_ratios[self.price_data.columns[1]] + spread
        
        # Create DataFrame with prices
        dates = pd.bdate_range(
            start=self.price_data.iloc[-1].name,
            periods=self.n_periods,
            freq='D'
        )
        
        prices = pd.DataFrame({
            self.price_data.columns[0]: p1,
            self.price_data.columns[1]: p2
        }, index=dates)
        
        return prices, pd.Series(spread, index=dates)
    
    def get_true_parameters(self) -> dict:
        """Get true parameters used in simulation."""
        return {
            'beta': self.beta,
            'phi_returns': self.phi_returns,
            'phi_spread': self.phi_spread,
            'return_mean': self.return_mean,
            'return_vol': self.return_vol,
            'spread_vol': self.spread_vol
        }

    def simulate_multiple_paths(self, n_paths: int) -> list:
        """
        Generate multiple cointegrated price series paths.
        
        Args:
            n_paths: Number of paths to simulate
            
        Returns:
            List of tuples containing:
            - DataFrame with simulated price series for each path
            - Series with true spread for each path
        """
        results = []
        
        for _ in range(n_paths):
            prices, spread = self.simulate()
            results.append((prices, spread))
        
        return results
