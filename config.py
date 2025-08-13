"""
Configuration file for the Stocks Research Platform.
Contains default settings and parameters for analysis.
"""

import os
from typing import List, Dict, Any

class Config:
    """Configuration class for the Stocks Research Platform."""
    
    # Data fetching settings
    DEFAULT_CACHE_DURATION_HOURS = 24
    DEFAULT_DATA_PERIOD = "2y"
    DEFAULT_SMA_PERIODS = [5, 10, 20, 50, 100, 200]
    DEFAULT_TOUCH_TOLERANCE = 0.01  # 1%
    
    # Analysis settings
    DEFAULT_ANALYSIS_WINDOW = 20
    DEFAULT_OPTIMIZATION_STEP = 5
    DEFAULT_OPTIMIZATION_RANGE = (5, 200)
    
    # Visualization settings
    DEFAULT_FIGURE_SIZE = (12, 8)
    DEFAULT_CHART_HEIGHT = 600
    DEFAULT_DASHBOARD_HEIGHT = 1200
    
    # Export settings
    DEFAULT_EXPORT_FORMAT = 'csv'
    DEFAULT_PLOT_FORMAT = 'html'
    DEFAULT_OUTPUT_DIR = 'output'
    DEFAULT_PLOT_DIR = 'plots'
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # API settings
    YAHOO_FINANCE_RATE_LIMIT_DELAY = 0.1  # seconds between requests
    
    # Technical analysis settings
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_BANDS_PERIOD = 20
    BOLLINGER_BANDS_STD = 2.0
    ATR_PERIOD = 14
    
    # Sample stocks for testing
    SAMPLE_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Technology
        'JPM', 'BAC', 'WFC', 'GS', 'MS',           # Financial
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',       # Healthcare
        'XOM', 'CVX', 'COP', 'EOG', 'SLB',        # Energy
        'SPY', 'QQQ', 'IWM', 'VTI', 'VEA'         # ETFs
    ]
    
    # Market sectors for categorization
    MARKET_SECTORS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'BMY'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'PSX', 'VLO'],
        'Consumer': ['AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX'],
        'Industrial': ['BA', 'CAT', 'MMM', 'GE', 'HON', 'UPS', 'FDX', 'RTX'],
        'Materials': ['LIN', 'APD', 'FCX', 'NEM', 'BLL', 'NUE', 'ALB', 'ECL'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SPG', 'DLR'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'DTE'],
        'Communication': ['META', 'GOOGL', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR']
    }
    
    # Default analysis parameters by stock type
    ANALYSIS_PARAMS = {
        'stocks': {
            'default_periods': [5, 10, 20, 50, 100, 200],
            'tolerance': 0.01,
            'data_period': '2y'
        },
        'etfs': {
            'default_periods': [5, 10, 20, 50, 100, 200],
            'tolerance': 0.008,
            'data_period': '3y'
        },
        'crypto': {
            'default_periods': [5, 10, 20, 50, 100, 200],
            'tolerance': 0.02,
            'data_period': '1y'
        },
        'forex': {
            'default_periods': [5, 10, 20, 50, 100, 200],
            'tolerance': 0.005,
            'data_period': '2y'
        }
    }
    
    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        'excellent_touch_frequency': 0.15,    # 15% or higher
        'good_touch_frequency': 0.10,         # 10% or higher
        'fair_touch_frequency': 0.05,         # 5% or higher
        'excellent_crossover_frequency': 12,   # 12 or more per year
        'good_crossover_frequency': 8,         # 8 or more per year
        'fair_crossover_frequency': 4          # 4 or more per year
    }
    
    # Risk management settings
    RISK_SETTINGS = {
        'max_position_size': 0.05,            # 5% of portfolio
        'stop_loss_percentage': 0.02,         # 2% stop loss
        'take_profit_percentage': 0.06,       # 6% take profit
        'max_drawdown': 0.15,                 # 15% max drawdown
        'correlation_threshold': 0.7          # Maximum correlation between positions
    }
    
    @classmethod
    def get_sma_periods(cls, stock_type: str = 'stocks') -> List[int]:
        """Get default SMA periods for a specific stock type."""
        return cls.ANALYSIS_PARAMS.get(stock_type, {}).get('default_periods', cls.DEFAULT_SMA_PERIODS)
    
    @classmethod
    def get_tolerance(cls, stock_type: str = 'stocks') -> float:
        """Get default tolerance for a specific stock type."""
        return cls.ANALYSIS_PARAMS.get(stock_type, {}).get('tolerance', cls.DEFAULT_TOUCH_TOLERANCE)
    
    @classmethod
    def get_data_period(cls, stock_type: str = 'stocks') -> str:
        """Get default data period for a specific stock type."""
        return cls.ANALYSIS_PARAMS.get(stock_type, {}).get('data_period', cls.DEFAULT_DATA_PERIOD)
    
    @classmethod
    def get_stocks_by_sector(cls, sector: str) -> List[str]:
        """Get list of stocks for a specific sector."""
        return cls.MARKET_SECTORS.get(sector, [])
    
    @classmethod
    def get_all_stocks(cls) -> List[str]:
        """Get all stocks from all sectors."""
        all_stocks = []
        for stocks in cls.MARKET_SECTORS.values():
            all_stocks.extend(stocks)
        return list(set(all_stocks))  # Remove duplicates
    
    @classmethod
    def get_sector_for_stock(cls, symbol: str) -> str:
        """Get the sector for a specific stock symbol."""
        for sector, stocks in cls.MARKET_SECTORS.items():
            if symbol in stocks:
                return sector
        return 'Unknown'
    
    @classmethod
    def get_performance_rating(cls, touch_frequency: float, crossover_frequency: float) -> Dict[str, Any]:
        """Get performance rating based on metrics."""
        rating = {
            'touch_rating': 'Poor',
            'crossover_rating': 'Poor',
            'overall_rating': 'Poor',
            'score': 0
        }
        
        # Rate touch frequency
        if touch_frequency >= cls.PERFORMANCE_THRESHOLDS['excellent_touch_frequency']:
            rating['touch_rating'] = 'Excellent'
            rating['score'] += 3
        elif touch_frequency >= cls.PERFORMANCE_THRESHOLDS['good_touch_frequency']:
            rating['touch_rating'] = 'Good'
            rating['score'] += 2
        elif touch_frequency >= cls.PERFORMANCE_THRESHOLDS['fair_touch_frequency']:
            rating['touch_rating'] = 'Fair'
            rating['score'] += 1
        
        # Rate crossover frequency
        if crossover_frequency >= cls.PERFORMANCE_THRESHOLDS['excellent_crossover_frequency']:
            rating['crossover_rating'] = 'Excellent'
            rating['score'] += 3
        elif crossover_frequency >= cls.PERFORMANCE_THRESHOLDS['good_crossover_frequency']:
            rating['crossover_rating'] = 'Good'
            rating['score'] += 2
        elif crossover_frequency >= cls.PERFORMANCE_THRESHOLDS['fair_crossover_frequency']:
            rating['crossover_rating'] = 'Fair'
            rating['score'] += 1
        
        # Overall rating
        if rating['score'] >= 5:
            rating['overall_rating'] = 'Excellent'
        elif rating['score'] >= 3:
            rating['overall_rating'] = 'Good'
        elif rating['score'] >= 1:
            rating['overall_rating'] = 'Fair'
        
        return rating
    
    @classmethod
    def create_output_directories(cls) -> None:
        """Create necessary output directories."""
        directories = [cls.DEFAULT_OUTPUT_DIR, cls.DEFAULT_PLOT_DIR]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    @classmethod
    def get_config_summary(cls) -> str:
        """Get a summary of the configuration."""
        summary = f"""
=== STOCKS RESEARCH PLATFORM CONFIGURATION ===

Data Settings:
- Default Cache Duration: {cls.DEFAULT_CACHE_DURATION_HOURS} hours
- Default Data Period: {cls.DEFAULT_DATA_PERIOD}
- Default SMA Periods: {cls.DEFAULT_SMA_PERIODS}
- Default Touch Tolerance: {cls.DEFAULT_TOUCH_TOLERANCE*100:.1f}%

Analysis Settings:
- Default Analysis Window: {cls.DEFAULT_ANALYSIS_WINDOW} days
- Optimization Range: {cls.DEFAULT_OPTIMIZATION_RANGE[0]} to {cls.DEFAULT_OPTIMIZATION_RANGE[1]} days
- Optimization Step: {cls.DEFAULT_OPTIMIZATION_STEP} days

Sample Stocks: {len(cls.SAMPLE_STOCKS)} stocks across {len(cls.MARKET_SECTORS)} sectors
Market Sectors: {', '.join(cls.MARKET_SECTORS.keys())}

Performance Thresholds:
- Excellent Touch Frequency: {cls.PERFORMANCE_THRESHOLDS['excellent_touch_frequency']*100:.1f}%
- Good Touch Frequency: {cls.PERFORMANCE_THRESHOLDS['good_touch_frequency']*100:.1f}%
- Excellent Crossover Frequency: {cls.PERFORMANCE_THRESHOLDS['excellent_crossover_frequency']} per year
- Good Crossover Frequency: {cls.PERFORMANCE_THRESHOLDS['good_crossover_frequency']} per year
"""
        return summary

# Create a global config instance
config = Config()

if __name__ == "__main__":
    # Print configuration summary
    print(config.get_config_summary())
    
    # Create output directories
    config.create_output_directories() 