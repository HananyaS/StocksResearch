"""
Utility functions for stock analysis and data processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

def calculate_returns(data: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        data: DataFrame with 'Close' prices
        method: Return calculation method ('simple', 'log', 'arithmetic')
        
    Returns:
        DataFrame with return columns
    """
    if 'Close' not in data.columns:
        raise ValueError("Data must contain 'Close' column")
    
    result = data.copy()
    
    if method == 'simple':
        result['Returns'] = data['Close'].pct_change()
    elif method == 'log':
        result['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    elif method == 'arithmetic':
        result['Returns'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
    else:
        raise ValueError("Method must be 'simple', 'log', or 'arithmetic'")
    
    return result

def calculate_volatility(data: pd.DataFrame, window: int = 20, 
                         annualize: bool = True) -> pd.DataFrame:
    """
    Calculate rolling volatility.
    
    Args:
        data: DataFrame with return data
        window: Rolling window size
        annualize: Whether to annualize volatility
        
    Returns:
        DataFrame with volatility column
    """
    if 'Returns' not in data.columns:
        data = calculate_returns(data)
    
    result = data.copy()
    
    # Calculate rolling standard deviation
    result[f'Volatility_{window}d'] = data['Returns'].rolling(window=window).std()
    
    # Annualize if requested
    if annualize:
        result[f'Volatility_{window}d'] *= np.sqrt(252)
    
    return result

def detect_support_resistance(data: pd.DataFrame, window: int = 20, 
                             threshold: float = 0.02) -> pd.DataFrame:
    """
    Detect potential support and resistance levels.
    
    Args:
        data: DataFrame with price data
        window: Rolling window for local extrema
        threshold: Minimum price change threshold
        
    Returns:
        DataFrame with support/resistance indicators
    """
    result = data.copy()
    
    # Calculate rolling min and max
    result['Rolling_Min'] = data['Low'].rolling(window=window).min()
    result['Rolling_Max'] = data['High'].rolling(window=window).max()
    
    # Detect support levels (price near rolling min)
    result['Support_Level'] = (data['Close'] - result['Rolling_Min']) / result['Rolling_Min'] <= threshold
    
    # Detect resistance levels (price near rolling max)
    result['Resistance_Level'] = (result['Rolling_Max'] - data['Close']) / data['Close'] <= threshold
    
    return result

def calculate_moving_averages(data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    Calculate multiple moving averages.
    
    Args:
        data: DataFrame with 'Close' prices
        periods: List of periods for moving averages
        
    Returns:
        DataFrame with moving average columns
    """
    result = data.copy()
    
    for period in periods:
        result[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
    
    return result

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, 
                             std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with 'Close' prices
        period: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        DataFrame with Bollinger Bands
    """
    result = data.copy()
    
    # Calculate SMA
    result[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    result[f'STD_{period}'] = data['Close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    result[f'BB_Upper_{period}'] = result[f'SMA_{period}'] + (std_dev * result[f'STD_{period}'])
    result[f'BB_Lower_{period}'] = result[f'SMA_{period}'] - (std_dev * result[f'STD_{period}'])
    
    # Calculate bandwidth and %B
    result[f'BB_Bandwidth_{period}'] = (result[f'BB_Upper_{period}'] - result[f'BB_Lower_{period}']) / result[f'SMA_{period}']
    result[f'BB_PercentB_{period}'] = (data['Close'] - result[f'BB_Lower_{period}']) / (result[f'BB_Upper_{period}'] - result[f'BB_Lower_{period}'])
    
    return result

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with 'Close' prices
        period: Period for RSI calculation
        
    Returns:
        DataFrame with RSI column
    """
    result = data.copy()
    
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RSI
    rs = avg_gains / avg_losses
    result[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    return result

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: DataFrame with 'Close' prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with MACD columns
    """
    result = data.copy()
    
    # Calculate EMAs
    result[f'EMA_{fast}'] = data['Close'].ewm(span=fast).mean()
    result[f'EMA_{slow}'] = data['Close'].ewm(span=slow).mean()
    
    # Calculate MACD line
    result[f'MACD_{fast}_{slow}'] = result[f'EMA_{fast}'] - result[f'EMA_{slow}']
    
    # Calculate signal line
    result[f'MACD_Signal_{signal}'] = result[f'MACD_{fast}_{slow}'].ewm(span=signal).mean()
    
    # Calculate histogram
    result[f'MACD_Histogram'] = result[f'MACD_{fast}_{slow}'] - result[f'MACD_Signal_{signal}']
    
    return result

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).
    
    Args:
        data: DataFrame with OHLC data
        period: Period for ATR calculation
        
    Returns:
        DataFrame with ATR column
    """
    result = data.copy()
    
    # Calculate True Range
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift(1))
    low_close = np.abs(data['Low'] - data['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR
    result[f'ATR_{period}'] = true_range.rolling(window=period).mean()
    
    return result

def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize data to a specific range.
    
    Args:
        data: DataFrame to normalize
        method: Normalization method ('minmax', 'zscore', 'decimal')
        
    Returns:
        Normalized DataFrame
    """
    result = data.copy()
    
    for column in data.select_dtypes(include=[np.number]).columns:
        if method == 'minmax':
            min_val = data[column].min()
            max_val = data[column].max()
            if max_val != min_val:
                result[column] = (data[column] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = data[column].mean()
            std_val = data[column].std()
            if std_val != 0:
                result[column] = (data[column] - mean_val) / std_val
        
        elif method == 'decimal':
            max_abs = data[column].abs().max()
            if max_abs != 0:
                result[column] = data[column] / max_abs
        
        else:
            raise ValueError("Method must be 'minmax', 'zscore', or 'decimal'")
    
    return result

def remove_outliers(data: pd.DataFrame, columns: Optional[List[str]] = None, 
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from data.
    
    Args:
        data: DataFrame to clean
        columns: Columns to check for outliers (None for all numeric)
        method: Outlier detection method ('iqr', 'zscore', 'isolation')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for column in columns:
        if column in data.columns:
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                result = result[mask]
            
            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                mask = z_scores < threshold
                result = result[mask]
            
            elif method == 'isolation':
                # Simple isolation forest approach
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                result = result[mask]
            
            else:
                raise ValueError("Method must be 'iqr', 'zscore', or 'isolation'")
    
    return result

def calculate_correlation_matrix(data: pd.DataFrame, 
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns.
    
    Args:
        data: DataFrame with numeric data
        columns: Columns to include in correlation (None for all numeric)
        
    Returns:
        Correlation matrix DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter data to specified columns
    data_subset = data[columns].dropna()
    
    # Calculate correlation matrix
    correlation_matrix = data_subset.corr()
    
    return correlation_matrix

def calculate_rolling_statistics(data: pd.DataFrame, column: str, 
                                window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling statistics for a column.
    
    Args:
        data: DataFrame with data
        column: Column to analyze
        window: Rolling window size
        
    Returns:
        DataFrame with rolling statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column {column} not found in data")
    
    result = data.copy()
    
    # Calculate various rolling statistics
    result[f'{column}_Rolling_Mean'] = data[column].rolling(window=window).mean()
    result[f'{column}_Rolling_Std'] = data[column].rolling(window=window).std()
    result[f'{column}_Rolling_Median'] = data[column].rolling(window=window).median()
    result[f'{column}_Rolling_Min'] = data[column].rolling(window=window).min()
    result[f'{column}_Rolling_Max'] = data[column].rolling(window=window).max()
    result[f'{column}_Rolling_Skew'] = data[column].rolling(window=window).skew()
    result[f'{column}_Rolling_Kurt'] = data[column].rolling(window=window).kurt()
    
    return result

def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format number as currency string.
    
    Args:
        value: Numeric value to format
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    elif currency == 'EUR':
        return f"€{value:,.2f}"
    elif currency == 'GBP':
        return f"£{value:,.2f}"
    else:
        return f"{value:,.2f}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format number as percentage string.
    
    Args:
        value: Numeric value to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimal_places}%}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default 