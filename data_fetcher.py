"""
Stock data fetching module using Yahoo Finance API.
Handles data retrieval, caching, and basic data validation.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Fetches and manages stock data from Yahoo Finance."""
    
    def __init__(self, cache_duration_hours: int = 24):
        """
        Initialize the data fetcher.
        
        Args:
            cache_duration_hours: How long to cache data (default: 24 hours)
        """
        self.cache_duration_hours = cache_duration_hours
        self._cache = {}
        self._cache_timestamps = {}
    
    def get_stock_data(self, symbol: str, period: str = "2y", 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data and additional indicators
        """
        cache_key = f"{symbol}_{period}_{start_date}_{end_date}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached data for {symbol}")
            return self._cache[cache_key]
        
        try:
            logger.info(f"Fetching data for {symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Add basic indicators
            data = self._add_basic_indicators(data)
            
            # Cache the data
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "2y") -> dict:
        """
        Fetch data for multiple stocks simultaneously.
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period)
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {str(e)}")
                results[symbol] = None
        
        return results
    
    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators to the data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with additional indicators
        """
        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Calculate log returns
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate volatility (rolling 20-day)
        data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate price change from previous close
        data['Price_Change'] = data['Close'] - data['Close'].shift(1)
        data['Price_Change_Pct'] = (data['Price_Change'] / data['Close'].shift(1)) * 100
        
        return data
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.cache_duration_hours * 3600)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def get_stock_info(self, symbol: str) -> dict:
        """
        Get basic stock information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

# Add missing import
import numpy as np 