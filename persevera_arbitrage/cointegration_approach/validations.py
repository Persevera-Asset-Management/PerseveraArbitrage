import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_price_data(price_data: pd.DataFrame, min_history: int) -> None:
    """Validate input price data for NaN and infinite values."""
    if price_data.isnull().any().any():
        logger.error("Price data validation failed: NaN values detected")
        raise ValueError("Price data contains NaN values")
    
    if np.isinf(price_data).any().any():
        logger.error("Price data validation failed: infinite values detected")
        raise ValueError("Price data contains infinite values")
    
    if len(price_data) < min_history:
        logger.error(f"Price data validation failed: insufficient history (got {len(price_data)}, need {min_history})")
        raise ValueError(f"Insufficient price history. Need at least {min_history} observations")
    
    logger.debug("Price data validation passed")
