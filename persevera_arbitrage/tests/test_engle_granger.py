import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from persevera_arbitrage.cointegration_approach import EngleGrangerPortfolio, EngleGrangerTestResult
from persevera_arbitrage.cointegration_approach.config import CointegrationConfig

def generate_cointegrated_series(periods=500, beta=0.7, noise_std=1.0):
    """Generate cointegrated price series for testing."""
    # Generate random walk for first series
    np.random.seed(42)  # For reproducibility
    dates = [datetime.now() + timedelta(days=x) for x in range(periods)]
    price1 = np.random.standard_normal(periods).cumsum()
    
    # Generate cointegrated second series
    noise = np.random.normal(0, noise_std, periods)
    price2 = beta * price1 + noise
    
    return pd.DataFrame({
        'Asset1': price1,
        'Asset2': price2
    }, index=dates)

class TestEngleGrangerPortfolio:
    @pytest.fixture
    def portfolio(self):
        """Create portfolio instance for testing."""
        config = CointegrationConfig(
            significance_level=0.05,
            min_history=100,
            position_size=1.0
        )
        return EngleGrangerPortfolio(config)
    
    @pytest.fixture
    def price_data(self):
        """Generate test price data."""
        return generate_cointegrated_series()
    
    def test_initialization(self, portfolio):
        """Test portfolio initialization."""
        assert portfolio.config is not None
        assert portfolio.test_results is None
        assert portfolio.dependent_variable is None
        assert portfolio.zscore is None
    
    def test_input_validation(self, portfolio):
        """Test input data validation."""
        # Test NaN values
        invalid_data = pd.DataFrame({
            'Asset1': [1, np.nan, 3],
            'Asset2': [1, 2, 3]
        })
        with pytest.raises(ValueError, match="NaN values"):
            portfolio.fit(invalid_data)
        
        # Test insufficient history
        short_data = pd.DataFrame({
            'Asset1': range(50),
            'Asset2': range(50)
        })
        with pytest.raises(ValueError, match="Insufficient price history"):
            portfolio.fit(short_data)
    
    def test_basic_fit(self, portfolio, price_data):
        """Test basic model fitting."""
        results = portfolio.fit(price_data)
        
        # Check results structure
        assert isinstance(results, EngleGrangerTestResult)
        assert hasattr(results, 'adf_statistic')
        assert hasattr(results, 'p_value')
        assert hasattr(results, 'critical_values')
        assert hasattr(results, 'hedge_ratios')
        assert hasattr(results, 'is_cointegrated')
        
        # Check data storage
        assert portfolio.price_data is not None
        assert portfolio.dependent_variable == 'Asset1'  # Default first column
        assert portfolio.zscore is not None
    
    def test_hedge_ratios(self, portfolio, price_data):
        """Test hedge ratio calculation."""
        portfolio.fit(price_data)
        
        # Check hedge ratios structure
        assert isinstance(portfolio.hedge_ratios, pd.DataFrame)
        assert list(portfolio.hedge_ratios.columns) == ['Asset1', 'Asset2']
        assert portfolio.hedge_ratios.iloc[0, 0] == 1.0  # First asset coefficient
        
        # Get position sizes
        positions = portfolio.get_position_sizes()
        assert isinstance(positions, pd.Series)
        assert list(positions.index) == ['Asset1', 'Asset2']
    
    def test_trading_signals(self, portfolio, price_data):
        """Test trading signal generation."""
        portfolio.fit(price_data)
        signals = portfolio.get_trading_signals(zscore_threshold=2.0)
        
        # Check signal properties
        assert isinstance(signals, pd.Series)
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(signals) == len(price_data)
        
        # Test invalid signal generation
        portfolio_unfit = EngleGrangerPortfolio()
        with pytest.raises(ValueError, match="Must fit model"):
            portfolio_unfit.get_trading_signals()
    
    def test_cointegration_check(self, portfolio, price_data):
        """Test cointegration checking."""
        portfolio.fit(price_data)
        
        # Test with default significance
        is_cointegrated = portfolio.is_cointegrated()
        assert isinstance(is_cointegrated, bool)
        
        # Test with custom significance
        is_cointegrated_custom = portfolio.is_cointegrated(significance=0.01)
        assert isinstance(is_cointegrated_custom, bool)
        
        # Test unfit portfolio
        portfolio_unfit = EngleGrangerPortfolio()
        with pytest.raises(ValueError, match="Must fit model"):
            portfolio_unfit.is_cointegrated()
    
    def test_different_dependent_variable(self, portfolio, price_data):
        """Test using different dependent variable."""
        results = portfolio.fit(price_data, dependent_variable='Asset2')
        assert portfolio.dependent_variable == 'Asset2'
        assert isinstance(results.hedge_ratios, pd.DataFrame)
        assert results.hedge_ratios.iloc[0]['Asset2'] == 1.0
    
    def test_add_constant(self, portfolio, price_data):
        """Test regression with constant term."""
        results_with_constant = portfolio.fit(price_data, add_constant=True)
        results_without_constant = portfolio.fit(price_data, add_constant=False)
        
        # Results should be different with constant
        assert not np.allclose(
            results_with_constant.hedge_ratios.values,
            results_without_constant.hedge_ratios.values
        )
    
    def test_position_size_scaling(self, portfolio, price_data):
        """Test position size scaling."""
        portfolio.fit(price_data)
        
        base_positions = portfolio.get_position_sizes()
        scaled_positions = portfolio.get_position_sizes(position_size=2.0)
        
        # Check scaling
        assert np.allclose(scaled_positions.values, base_positions.values * 2.0)

if __name__ == '__main__':
    pytest.main([__file__])