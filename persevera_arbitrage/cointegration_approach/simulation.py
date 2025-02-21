from typing import Tuple, Optional
import numpy as np
import pandas as pd

class CointegrationSimulation:
    """
    Simulate cointegrated price series pairs.
    
    Generates two price series that are cointegrated through the following process:
    1. Generate first price series using random walk with drift
    2. Generate cointegration error (residuals) using AR(1) process
    3. Calculate second price series using cointegration equation
    
    The cointegration relationship is:
    P2_t = beta * P1_t + error_t
    
    where:
    - P1_t = P1_{t-1} + drift + eps1_t  (random walk with drift)
    - error_t = phi * error_{t-1} + eps2_t  (AR(1) process)
    - eps1_t ~ N(0, price_vol^2)  (price shocks)
    - eps2_t ~ N(0, error_vol^2)  (error shocks)
    """
    
    def __init__(self,
                 price_data: pd.DataFrame,
                 n_periods: int = 1000,
                 beta: float = 1.0,
                 phi: float = 0.7,
                 drift: float = 0.0001,
                 price_vol: float = 0.01,
                 error_vol: float = 0.1,
                 random_seed: Optional[int] = None):
        """
        Initialize simulation parameters.
        
        Args:
            n_periods: Number of periods to simulate
            beta: Cointegration coefficient
            phi: AR(1) coefficient for error process (0 < phi < 1)
            drift: Drift term for price process
            price_vol: Volatility of random shocks to P1
            error_vol: Volatility of shocks to error process
            random_seed: Random seed for reproducibility
        """
        if not 0 <= phi < 1:
            raise ValueError("phi must be between 0 and 1 for stationarity")
            
        if price_data.empty:
            raise ValueError("price_data must be a non-empty DataFrame.")
            
        self.n_periods = n_periods
        self.beta = beta
        self.phi = phi
        self.drift = drift
        self.price_vol = price_vol
        self.error_vol = error_vol
        self.price_data = price_data  # Store price_data as an instance variable
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def simulate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate cointegrated price series.
        
        Returns:
            Tuple containing:
            - DataFrame with simulated price series
            - Series with true cointegration errors
        """
        # Generate random shocks for P1 (eps1)
        price_shocks = np.random.normal(
            loc=self.drift,
            scale=self.price_vol,
            size=self.n_periods
        )
        
        # Generate first price series (random walk with drift)
        p1 = np.zeros(self.n_periods)
        p1[0] = self.price_data.iloc[-1].iloc[0]  # Use self.price_data
        
        for t in range(1, self.n_periods):
            p1[t] = p1[t-1] + price_shocks[t]
            
        # Generate random shocks for error process (eps2)
        error_shocks = np.random.normal(
            loc=0,
            scale=self.error_vol,
            size=self.n_periods
        )
        
        # Generate cointegration errors using AR(1) process
        errors = np.zeros(self.n_periods)
        errors[0] = error_shocks[0]
        
        for t in range(1, self.n_periods):
            errors[t] = self.phi * errors[t-1] + error_shocks[t]
            
        # Generate second price series using cointegration equation
        p2 = self.beta * p1 + errors
        
        # Create DataFrame with prices
        dates = pd.bdate_range(
            start=self.price_data.iloc[-1].name,  # Use self.price_data
            periods=self.n_periods,
            freq='D'
        )
        
        prices = pd.DataFrame({
            self.price_data.columns[0]: p1,
            self.price_data.columns[1]: p2
        }, index=dates)
        
        return prices, pd.Series(errors, index=dates)
    
    def get_true_parameters(self) -> dict:
        """Get true parameters used in simulation."""
        return {
            'beta': self.beta,
            'phi': self.phi,
            'drift': self.drift,
            'price_vol': self.price_vol,
            'error_vol': self.error_vol
        }

    def simulate_multiple_paths(self, n_paths: int) -> list:
        """
        Generate multiple cointegrated price series paths.
        
        Args:
            n_paths: Number of paths to simulate.
            
        Returns:
            List of tuples containing:
            - DataFrame with simulated price series for each path
            - Series with true cointegration errors for each path
        """
        results = []
        
        for _ in range(n_paths):
            prices, errors = self.simulate()
            results.append((prices, errors))
        
        return results
