"""
SMA (Simple Moving Average) analysis module.
Analyzes stock tendencies to touch SMA-k as a function of k.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class SMAAnalyzer:
    """Analyzes Simple Moving Averages and stock price interactions."""
    
    def __init__(self):
        """Initialize the SMA analyzer."""
        pass
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average for a given period.
        
        Args:
            data: DataFrame with 'Close' prices
            period: SMA period (k)
            
        Returns:
            Series with SMA values
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        return data['Close'].rolling(window=period).mean()
    
    def calculate_multiple_smas(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate multiple SMAs for different periods.
        
        Args:
            data: DataFrame with 'Close' prices
            periods: List of SMA periods
            
        Returns:
            DataFrame with original data plus SMA columns
        """
        result = data.copy()
        
        for period in periods:
            sma_col = f'SMA_{period}'
            result[sma_col] = self.calculate_sma(data, period)
        
        return result
    
    def analyze_sma_touches(self, data: pd.DataFrame, sma_period: int, 
                           tolerance: float = 0.01) -> Dict:
        """
        Analyze how often a stock touches its SMA-k.
        
        Args:
            data: DataFrame with price and SMA data
            sma_period: SMA period to analyze
            tolerance: Percentage tolerance for "touching" (default: 1%)
            
        Returns:
            Dictionary with touch analysis results
        """
        sma_col = f'SMA_{sma_period}'
        if sma_col not in data.columns:
            data = self.calculate_multiple_smas(data, [sma_period])
        
        # Calculate distance from SMA
        data['Distance_From_SMA'] = ((data['Close'] - data[sma_col]) / data[sma_col]) * 100
        
        # Identify touches (when price is within tolerance of SMA)
        touches = abs(data['Distance_From_SMA']) <= tolerance
        
        # Calculate touch statistics
        total_touches = touches.sum()
        touch_frequency = total_touches / len(data)
        
        # Analyze touch patterns
        touch_analysis = {
            'sma_period': sma_period,
            'total_touches': total_touches,
            'touch_frequency': touch_frequency,
            'touch_percentage': touch_frequency * 100,
            'avg_distance': data['Distance_From_SMA'].mean(),
            'std_distance': data['Distance_From_SMA'].std(),
            'min_distance': data['Distance_From_SMA'].min(),
            'max_distance': data['Distance_From_SMA'].max(),
            'tolerance_used': tolerance
        }
        
        return touch_analysis
    
    def analyze_sma_tendencies(self, data: pd.DataFrame, periods: List[int], 
                             tolerance: float = 0.01) -> pd.DataFrame:
        """
        Analyze SMA touch tendencies across multiple periods.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            tolerance: Percentage tolerance for touches
            
        Returns:
            DataFrame with analysis results for each period
        """
        results = []
        
        for period in periods:
            try:
                analysis = self.analyze_sma_touches(data, period, tolerance)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing SMA {period}: {str(e)}")
                results.append({
                    'sma_period': period,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def find_sma_optimal_periods(self, data: pd.DataFrame, 
                                period_range: Tuple[int, int] = (5, 200),
                                step: int = 5) -> pd.DataFrame:
        """
        Find optimal SMA periods based on touch frequency and consistency.
        
        Args:
            data: DataFrame with price data
            period_range: Tuple of (min_period, max_period)
            step: Step size for period analysis
            
        Returns:
            DataFrame with analysis results for each period
        """
        periods = list(range(period_range[0], period_range[1] + 1, step))
        
        # Analyze all periods
        analysis_df = self.analyze_sma_tendencies(data, periods)
        
        # Add additional metrics
        analysis_df['touch_efficiency'] = analysis_df['touch_frequency'] / analysis_df['sma_period']
        analysis_df['consistency_score'] = 1 / (analysis_df['std_distance'] + 1e-6)
        
        # Sort by touch efficiency
        analysis_df = analysis_df.sort_values('touch_efficiency', ascending=False)
        
        return analysis_df
    
    def calculate_sma_slope(self, data: pd.DataFrame, period: int, 
                           window: int = 20) -> pd.Series:
        """
        Calculate the slope of SMA over a rolling window.
        
        Args:
            data: DataFrame with price data
            period: SMA period
            window: Rolling window for slope calculation
            
        Returns:
            Series with SMA slope values
        """
        sma_col = f'SMA_{period}'
        if sma_col not in data.columns:
            data = self.calculate_multiple_smas(data, [period])
        
        # Calculate slope using linear regression on rolling windows
        slopes = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            if len(window_data) >= 2:
                x = np.arange(len(window_data)).reshape(-1, 1)
                y = window_data[sma_col].values
                
                if not np.isnan(y).any():
                    model = LinearRegression()
                    model.fit(x, y)
                    slopes.iloc[i] = model.coef_[0]
        
        return slopes
    
    def analyze_sma_momentum(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Analyze SMA momentum characteristics.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            
        Returns:
            DataFrame with momentum analysis results
        """
        momentum_results = []
        
        for period in periods:
            try:
                # Calculate SMA and slope
                data_with_sma = self.calculate_multiple_smas(data, [period])
                slopes = self.calculate_sma_slope(data_with_sma, period)
                
                # Analyze slope characteristics
                slope_stats = {
                    'sma_period': period,
                    'avg_slope': slopes.mean(),
                    'slope_std': slopes.std(),
                    'positive_slope_pct': (slopes > 0).mean() * 100,
                    'negative_slope_pct': (slopes < 0).mean() * 100,
                    'max_positive_slope': slopes.max(),
                    'max_negative_slope': slopes.min()
                }
                
                momentum_results.append(slope_stats)
                
            except Exception as e:
                logger.error(f"Error analyzing momentum for SMA {period}: {str(e)}")
                momentum_results.append({
                    'sma_period': period,
                    'error': str(e)
                })
        
        return pd.DataFrame(momentum_results)
    
    def get_sma_summary_stats(self, data: pd.DataFrame, periods: List[int]) -> Dict:
        """
        Get comprehensive summary statistics for multiple SMAs.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        # Calculate all SMAs
        data_with_smas = self.calculate_multiple_smas(data, periods)
        
        # Analyze touches
        touch_analysis = self.analyze_sma_tendencies(data_with_smas, periods)
        
        # Analyze momentum
        momentum_analysis = self.analyze_sma_momentum(data_with_smas, periods)
        
        # Validate that we have valid data
        if touch_analysis.empty:
            logger.warning("No touch analysis data available")
            return {
                'total_periods_analyzed': len(periods),
                'periods_with_errors': len(periods),
                'best_touch_period': periods[0] if periods else 0,
                'best_touch_frequency': 0.0,
                'avg_touch_frequency': 0.0,
                'momentum_analysis': []
            }
        
        # Check for required columns
        required_columns = ['touch_frequency', 'sma_period']
        if not all(col in touch_analysis.columns for col in required_columns):
            logger.warning(f"Missing required columns in touch analysis: {required_columns}")
            return {
                'total_periods_analyzed': len(periods),
                'periods_with_errors': len(periods),
                'best_touch_period': periods[0] if periods else 0,
                'best_touch_frequency': 0.0,
                'avg_touch_frequency': 0.0,
                'momentum_analysis': momentum_analysis.to_dict('records') if not momentum_analysis.empty else []
            }
        
        # Combine results
        summary = {
            'total_periods_analyzed': len(periods),
            'periods_with_errors': len(touch_analysis[touch_analysis.get('error', '').notna()]) if 'error' in touch_analysis.columns else 0,
            'best_touch_period': touch_analysis.loc[touch_analysis['touch_frequency'].idxmax(), 'sma_period'],
            'best_touch_frequency': touch_analysis['touch_frequency'].max(),
            'avg_touch_frequency': touch_analysis['touch_frequency'].mean(),
            'momentum_analysis': momentum_analysis.to_dict('records') if not momentum_analysis.empty else []
        }
        
        return summary 