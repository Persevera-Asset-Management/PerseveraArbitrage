import pandas as pd
import numpy as np

def validate_price_data(price_data: pd.DataFrame, min_history: int) -> None:
    """Validate input price data for NaN and infinite values."""
    if price_data.isnull().any().any():
        raise ValueError("Price data contains NaN values")
    if np.isinf(price_data).any().any():
        raise ValueError("Price data contains infinite values")
    if len(price_data) < min_history:
        raise ValueError(f"Insufficient price history. Need at least {min_history} observations")

def verify_correlations(price_data: pd.DataFrame, threshold: float = 0.7) -> bool:
    """Verify correlations between all pairs."""
    correlation_matrix = price_data.corr()
    low_correlation_pairs = correlation_matrix[correlation_matrix < threshold].stack().index.tolist()
    
    for (i, j) in low_correlation_pairs:
        if i != j:  # Avoid self-correlation
            print(f"Warning: Low correlation between {i} and {j}")
    
    return len(low_correlation_pairs) == 0 