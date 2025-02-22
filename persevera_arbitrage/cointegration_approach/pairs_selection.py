from typing import List, Tuple, Optional
import pandas as pd
import itertools
from dataclasses import dataclass

from .engle_granger import EngleGrangerPortfolio
from .johansen import JohansenPortfolio
from .config import CointegrationConfig
from .utils import get_hurst_exponent

@dataclass
class PairSelectionCriteria:
    """Configuration for pair selection criteria."""
    max_half_life: int = 30           # Maximum acceptable half-life in days
    hurst_threshold: float = 0.5      # Maximum Hurst exponent for mean reversion
    significance_level: float = 0.05   # Statistical significance for cointegration tests
    min_correlation: float = 0.7      # Minimum correlation between pairs

@dataclass
class PairTestResult:
    """Container for pair testing results."""
    asset1: str
    asset2: str
    is_cointegrated_johansen: bool
    is_cointegrated_engle_granger: bool
    half_life: float
    hurst_exponent: float
    correlation: float
    hedge_ratio: float
    passed_all_criteria: bool

class CointegrationPairSelector:
    """
    Selects pairs of assets based on cointegration and mean-reversion criteria.
    
    Selection criteria:
    1. Both Johansen and Engle-Granger tests must indicate cointegration
    2. Spread must show mean-reverting behavior (Hurst exponent < threshold)
    3. Half-life must be below specified threshold
    4. Assets must have sufficient correlation
    """
    
    def __init__(self, 
                 config: Optional[CointegrationConfig] = None,
                 criteria: Optional[PairSelectionCriteria] = None):
        """
        Initialize pair selector.
        
        Args:
            config: Configuration for cointegration tests
            criteria: Criteria for pair selection
        """
        self.config = config or CointegrationConfig()
        self.criteria = criteria or PairSelectionCriteria()
        
        # Initialize test objects
        self.johansen = JohansenPortfolio(config)
        self.engle_granger = EngleGrangerPortfolio(config)
        
    def select_pairs(self, price_data: pd.DataFrame) -> List[PairTestResult]:
        """
        Test all possible pairs in price data for cointegration and mean reversion.
        
        Args:
            price_data: DataFrame with asset prices as columns
            
        Returns:
            List of PairTestResult objects for pairs that pass all criteria
        """
        all_pairs = list(itertools.combinations(price_data.columns, 2))
        results = []
        
        for asset1, asset2 in all_pairs:
            pair_data = price_data[[asset1, asset2]]
            result = self._test_pair(pair_data, asset1, asset2)
            results.append(result)
            
        return results
    
    def _test_pair(self, 
                   pair_data: pd.DataFrame, 
                   asset1: str, 
                   asset2: str) -> PairTestResult:
        """
        Test a single pair for all selection criteria.
        
        Args:
            pair_data: Price data for the pair
            asset1: Name of first asset
            asset2: Name of second asset
            
        Returns:
            PairTestResult containing test results
        """
        # Check correlation
        correlation = pair_data[asset1].corr(pair_data[asset2])
        
        # Test cointegration using Johansen
        self.johansen.fit(pair_data)
        is_cointegrated_johansen = self.johansen.is_cointegrated(significance='95%')
        
        # Test cointegration using Engle-Granger (both directions)
        self.engle_granger.fit(pair_data, dependent_variable=asset1)
        is_cointegrated_eg_1 = self.engle_granger.is_cointegrated(significance=0.05)
        
        self.engle_granger.fit(pair_data, dependent_variable=asset2)
        is_cointegrated_eg_2 = self.engle_granger.is_cointegrated(significance=0.05)
        
        is_cointegrated_engle_granger = is_cointegrated_eg_1 and is_cointegrated_eg_2
        
        # Get spread properties
        spread = self.johansen.construct_mean_reverting_portfolio(pair_data)
        half_life = self.johansen.half_life
        hurst_exponent = get_hurst_exponent(spread)
        
        # Get hedge ratio from Johansen test
        hedge_ratio = abs(self.johansen.hedge_ratios.iloc[0, 1])
        
        # Check if pair passes all criteria
        passed_all_criteria = (
            is_cointegrated_johansen and
            is_cointegrated_engle_granger and
            half_life < self.criteria.max_half_life and
            hurst_exponent < self.criteria.hurst_threshold and
            correlation > self.criteria.min_correlation
        )
        
        return PairTestResult(
            asset1=asset1,
            asset2=asset2,
            is_cointegrated_johansen=is_cointegrated_johansen,
            is_cointegrated_engle_granger=is_cointegrated_engle_granger,
            half_life=half_life,
            hurst_exponent=hurst_exponent,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            passed_all_criteria=passed_all_criteria
        )
    
    def get_best_pairs(self, 
                      price_data: pd.DataFrame, 
                      n_pairs: int = 5) -> List[PairTestResult]:
        """
        Get the best n pairs sorted by half-life.
        
        Args:
            price_data: DataFrame with asset prices as columns
            n_pairs: Number of pairs to return
            
        Returns:
            List of best pairs sorted by half-life
        """
        results = self.select_pairs(price_data)
        passing_pairs = [r for r in results if r.passed_all_criteria]
        
        # Sort by half-life (faster mean reversion is better)
        passing_pairs.sort(key=lambda x: x.half_life)
        
        return passing_pairs[:n_pairs]
