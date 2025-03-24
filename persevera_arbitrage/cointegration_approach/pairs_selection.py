from typing import List, Optional
import pandas as pd
import itertools
from dataclasses import dataclass

from .engle_granger import EngleGrangerPortfolio
from .johansen import JohansenPortfolio
from .config import CointegrationConfig
from .utils import get_hurst_exponent

from persevera_tools.data import get_descriptors

@dataclass
class PairSelectionCriteria:
    """Configuration for pair selection criteria."""
    max_half_life: int = 60           # Maximum acceptable half-life in days
    min_half_life: int = 1            # Maximum acceptable half-life in days
    hurst_threshold: float = 0.5      # Maximum Hurst exponent for mean reversion
    significance_level: float = 0.05  # Statistical significance for cointegration tests
    max_vol_ratio: float = 1.5        # Maximum ratio of volatility of the spread to the volatility of the assets

@dataclass
class PairTestResult:
    """Container for pair testing results."""
    asset1: str
    asset2: str
    is_cointegrated_johansen: bool
    is_cointegrated_engle_granger: bool
    half_life: float
    hurst_exponent: float
    hedge_ratio: float
    passed_all_criteria: bool

class CointegrationPairSelector:
    """
    Selects pairs of assets based on cointegration and mean-reversion criteria.
    
    Selection criteria:
    1. Both Johansen and Engle-Granger tests must indicate cointegration
    2. Spread must show mean-reverting behavior (Hurst exponent < threshold)
    3. Half-life must be below specified threshold
    4. Volatility matching
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
        # Test cointegration using Johansen
        jh_results = self.johansen.fit(pair_data)
        is_cointegrated_johansen = self.johansen.is_cointegrated(significance='95%')
        
        # Test cointegration using Engle-Granger (both directions)
        eg_results_1 = self.engle_granger.fit(pair_data, dependent_variable=asset1)
        is_cointegrated_eg_1 = self.engle_granger.is_cointegrated(significance=0.05)
        
        eg_results_2 = self.engle_granger.fit(pair_data, dependent_variable=asset2)
        is_cointegrated_eg_2 = self.engle_granger.is_cointegrated(significance=0.05)
        
        is_cointegrated_engle_granger = is_cointegrated_eg_1 and is_cointegrated_eg_2
        
        # Due to Engle-Granger's test sensitivity to the ordering of variables, we select the
        # combination that generated the lowest p-value
        if eg_results_1.p_value < eg_results_2.p_value:
            eg_results = eg_results_1
        else:
            eg_results = eg_results_2

        # Calculate the volatility of the assets
        vol_pairs = get_descriptors(tickers=[asset1, asset2], descriptors='volatility_12m', start_date=price_data.index.max(), end_date=price_data.index.max())

        # After performing both tests, we use the residuals from the Engle-Granger test as the spread
        spread = eg_results.residuals
        half_life = eg_results.half_life
        hurst_exponent = get_hurst_exponent(spread)
        hedge_ratio = list(eg_results.hedge_ratios.values())[1]
        vol_ratio = vol_pairs.values.max() / vol_pairs.values.min()
        
        # Check if pair passes all criteria
        passed_all_criteria = (
            is_cointegrated_johansen and
            is_cointegrated_engle_granger and
            self.criteria.min_half_life <= half_life <= self.criteria.max_half_life and
            hurst_exponent < self.criteria.hurst_threshold and
            vol_ratio <= self.criteria.max_vol_ratio
        )
        
        return PairTestResult(
            asset1=asset1,
            asset2=asset2,
            is_cointegrated_johansen=is_cointegrated_johansen,
            is_cointegrated_engle_granger=is_cointegrated_engle_granger,
            half_life=half_life,
            hurst_exponent=hurst_exponent,
            hedge_ratio=hedge_ratio,
            passed_all_criteria=passed_all_criteria,
            vol_ratio=vol_ratio
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

    def get_volatility_metrics(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get volatility metrics for all pairs.
        """
        results = self.select_pairs(price_data)
        return pd.DataFrame([r.volatility_metrics for r in results])

