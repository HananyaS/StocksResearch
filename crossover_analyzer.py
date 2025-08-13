"""
Crossover analysis module for SMA crossovers.
Analyzes the distribution of days between different crossover events.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class CrossoverAnalyzer:
    """Analyzes SMA crossovers and their timing patterns."""
    
    def __init__(self):
        """Initialize the crossover analyzer."""
        pass
    
    def detect_crossovers(self, data: pd.DataFrame, sma_period: int) -> pd.DataFrame:
        """
        Detect when stock price crosses above/below SMA.
        
        Args:
            data: DataFrame with 'Close' prices and SMA data
            sma_period: SMA period to analyze
            
        Returns:
            DataFrame with crossover events
        """
        sma_col = f'SMA_{sma_period}'
        if sma_col not in data.columns:
            raise ValueError(f"SMA column {sma_col} not found in data")
        
        # Create a copy to avoid modifying original data
        analysis_data = data.copy()
        
        # Calculate previous day's position relative to SMA
        analysis_data['Prev_Above_SMA'] = analysis_data['Close'].shift(1) > analysis_data[sma_col].shift(1)
        analysis_data['Current_Above_SMA'] = analysis_data['Close'] > analysis_data[sma_col]
        
        # Detect crossovers
        analysis_data['Bullish_Cross'] = (~analysis_data['Prev_Above_SMA']) & analysis_data['Current_Above_SMA']
        analysis_data['Bearish_Cross'] = analysis_data['Prev_Above_SMA'] & (~analysis_data['Current_Above_SMA'])
        
        # Add crossover type
        analysis_data['Crossover_Type'] = 'None'
        analysis_data.loc[analysis_data['Bullish_Cross'], 'Crossover_Type'] = 'Bullish'
        analysis_data.loc[analysis_data['Bearish_Cross'], 'Crossover_Type'] = 'Bearish'
        
        # Calculate distance from SMA at crossover
        analysis_data['Crossover_Distance'] = np.nan
        crossover_mask = analysis_data['Bullish_Cross'] | analysis_data['Bearish_Cross']
        analysis_data.loc[crossover_mask, 'Crossover_Distance'] = (
            (analysis_data.loc[crossover_mask, 'Close'] - analysis_data.loc[crossover_mask, sma_col]) / 
            analysis_data.loc[crossover_mask, sma_col]
        ) * 100
        
        return analysis_data
    
    def analyze_crossover_timing(self, data: pd.DataFrame, sma_period: int) -> Dict:
        """
        Analyze the timing patterns of crossovers.
        
        Args:
            data: DataFrame with crossover data
            sma_period: SMA period analyzed
            
        Returns:
            Dictionary with timing analysis results
        """
        # Detect crossovers first
        crossover_data = self.detect_crossovers(data, sma_period)
        
        # Get crossover events
        bullish_crosses = crossover_data[crossover_data['Bullish_Cross']]
        bearish_crosses = crossover_data[crossover_data['Bearish_Cross']]
        
        # Calculate days between crossovers
        bullish_intervals = self._calculate_intervals(bullish_crosses.index)
        bearish_intervals = self._calculate_intervals(bearish_crosses.index)
        
        # Analyze timing patterns
        timing_analysis = {
            'sma_period': sma_period,
            'total_bullish_crosses': len(bullish_crosses),
            'total_bearish_crosses': len(bearish_crosses),
            'total_crosses': len(bullish_crosses) + len(bearish_crosses),
            'bullish_intervals': bullish_intervals,
            'bearish_intervals': bearish_intervals,
            'bullish_interval_stats': self._calculate_interval_stats(bullish_intervals),
            'bearish_interval_stats': self._calculate_interval_stats(bearish_intervals),
            'crossover_frequency': (len(bullish_crosses) + len(bearish_crosses)) / len(data) * 252,  # Annualized
            'bullish_crossover_pct': len(bullish_crosses) / (len(bullish_crosses) + len(bearish_crosses)) * 100 if (len(bullish_crosses) + len(bearish_crosses)) > 0 else 0
        }
        
        return timing_analysis
    
    def _calculate_intervals(self, crossover_dates: pd.DatetimeIndex) -> List[int]:
        """Calculate intervals between crossover events in days."""
        if len(crossover_dates) < 2:
            return []
        
        intervals = []
        for i in range(1, len(crossover_dates)):
            interval = (crossover_dates[i] - crossover_dates[i-1]).days
            intervals.append(interval)
        
        return intervals
    
    def _calculate_interval_stats(self, intervals: List[int]) -> Dict:
        """Calculate statistical measures for intervals."""
        if not intervals:
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
        
        return {
            'mean': np.mean(intervals),
            'median': np.median(intervals),
            'std': np.std(intervals),
            'min': np.min(intervals),
            'max': np.max(intervals),
            'count': len(intervals)
        }
    
    def analyze_multiple_sma_crossovers(self, data: pd.DataFrame, 
                                      periods: List[int]) -> pd.DataFrame:
        """
        Analyze crossovers for multiple SMA periods.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            
        Returns:
            DataFrame with crossover analysis for each period
        """
        results = []
        
        for period in periods:
            try:
                analysis = self.analyze_crossover_timing(data, period)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing crossovers for SMA {period}: {str(e)}")
                results.append({
                    'sma_period': period,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def analyze_crossover_distributions(self, data: pd.DataFrame, 
                                      periods: List[int]) -> Dict:
        """
        Analyze the distribution of days between crossovers across multiple periods.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            
        Returns:
            Dictionary with distribution analysis results
        """
        distribution_analysis = {}
        
        for period in periods:
            try:
                # Get crossover timing analysis
                timing_analysis = self.analyze_crossover_timing(data, period)
                
                # Analyze distributions
                period_distributions = {
                    'bullish_distribution': self._analyze_interval_distribution(
                        timing_analysis['bullish_intervals']
                    ),
                    'bearish_distribution': self._analyze_interval_distribution(
                        timing_analysis['bearish_intervals']
                    ),
                    'combined_distribution': self._analyze_interval_distribution(
                        timing_analysis['bullish_intervals'] + timing_analysis['bearish_intervals']
                    )
                }
                
                distribution_analysis[period] = {
                    'timing_analysis': timing_analysis,
                    'distributions': period_distributions
                }
                
            except Exception as e:
                logger.error(f"Error analyzing distributions for SMA {period}: {str(e)}")
                distribution_analysis[period] = {'error': str(e)}
        
        return distribution_analysis
    
    def _analyze_interval_distribution(self, intervals: List[int]) -> Dict:
        """
        Analyze the statistical distribution of intervals.
        
        Args:
            intervals: List of interval values
            
        Returns:
            Dictionary with distribution analysis
        """
        if not intervals:
            return {'error': 'No intervals to analyze'}
        
        try:
            # Basic statistics
            stats_dict = {
                'count': len(intervals),
                'mean': np.mean(intervals),
                'median': np.median(intervals),
                'std': np.std(intervals),
                'skewness': stats.skew(intervals),
                'kurtosis': stats.kurtosis(intervals),
                'min': np.min(intervals),
                'max': np.max(intervals),
                'q25': np.percentile(intervals, 25),
                'q75': np.percentile(intervals, 75)
            }
            
            # Test for normality
            if len(intervals) >= 3:
                try:
                    _, p_value = stats.normaltest(intervals)
                    stats_dict['normality_p_value'] = p_value
                    stats_dict['is_normal'] = p_value > 0.05
                except:
                    stats_dict['normality_p_value'] = np.nan
                    stats_dict['is_normal'] = np.nan
            else:
                stats_dict['normality_p_value'] = np.nan
                stats_dict['is_normal'] = np.nan
            
            return stats_dict
            
        except Exception as e:
            return {'error': str(e)}
    
    def find_crossover_patterns(self, data: pd.DataFrame, 
                               periods: List[int]) -> pd.DataFrame:
        """
        Find patterns in crossover behavior across different SMA periods.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            
        Returns:
            DataFrame with pattern analysis results
        """
        # Analyze all periods
        crossover_analysis = self.analyze_multiple_sma_crossovers(data, periods)
        
        # Add pattern metrics
        pattern_results = []
        
        for _, row in crossover_analysis.iterrows():
            if 'error' not in row or pd.isna(row.get('error')):
                try:
                    # Calculate pattern metrics
                    pattern_metrics = {
                        'sma_period': row['sma_period'],
                        'crossover_frequency': row['crossover_frequency'],
                        'bullish_ratio': row['bullish_crossover_pct'] / 100,
                        'avg_interval': row['bullish_interval_stats']['mean'],
                        'interval_consistency': 1 / (row['bullish_interval_stats']['std'] + 1e-6),
                        'total_crosses': row['total_crosses']
                    }
                    
                    # Add efficiency score (higher frequency with lower intervals is better)
                    if row['bullish_interval_stats']['mean'] > 0:
                        pattern_metrics['efficiency_score'] = (
                            row['crossover_frequency'] / row['bullish_interval_stats']['mean']
                        )
                    else:
                        pattern_metrics['efficiency_score'] = 0
                    
                    pattern_results.append(pattern_metrics)
                    
                except Exception as e:
                    logger.error(f"Error calculating pattern metrics for SMA {row['sma_period']}: {str(e)}")
                    pattern_results.append({
                        'sma_period': row['sma_period'],
                        'error': str(e)
                    })
            else:
                pattern_results.append({
                    'sma_period': row['sma_period'],
                    'error': row['error']
                })
        
        return pd.DataFrame(pattern_results)
    
    def get_crossover_summary(self, data: pd.DataFrame, 
                             periods: List[int]) -> Dict:
        """
        Get comprehensive summary of crossover analysis.
        
        Args:
            data: DataFrame with price data
            periods: List of SMA periods to analyze
            
        Returns:
            Dictionary with crossover summary
        """
        # Get all analyses
        timing_analysis = self.analyze_multiple_sma_crossovers(data, periods)
        distribution_analysis = self.analyze_crossover_distributions(data, periods)
        pattern_analysis = self.find_crossover_patterns(data, periods)
        
        # Validate that we have valid data
        if timing_analysis.empty:
            logger.warning("No timing analysis data available")
            return {
                'total_periods_analyzed': len(periods),
                'periods_with_errors': len(periods),
                'best_crossover_frequency': 0.0,
                'best_crossover_period': periods[0] if periods else 0,
                'avg_crossover_frequency': 0.0,
                'best_bullish_ratio': 0.0,
                'most_consistent_intervals': periods[0] if periods else 0,
                'timing_analysis': [],
                'pattern_analysis': []
            }
        
        # Check for required columns
        required_columns = ['crossover_frequency', 'sma_period']
        if not all(col in timing_analysis.columns for col in required_columns):
            logger.warning(f"Missing required columns in timing analysis: {required_columns}")
            return {
                'total_periods_analyzed': len(periods),
                'periods_with_errors': len(periods),
                'best_crossover_frequency': 0.0,
                'best_crossover_period': periods[0] if periods else 0,
                'avg_crossover_frequency': 0.0,
                'best_bullish_ratio': 0.0,
                'most_consistent_intervals': periods[0] if periods else 0,
                'timing_analysis': timing_analysis.to_dict('records'),
                'pattern_analysis': pattern_analysis.to_dict('records') if not pattern_analysis.empty else []
            }
        
        # Combine results
        summary = {
            'total_periods_analyzed': len(periods),
            'periods_with_errors': len(timing_analysis[timing_analysis.get('error', '').notna()]) if 'error' in timing_analysis.columns else 0,
            'best_crossover_frequency': timing_analysis['crossover_frequency'].max(),
            'best_crossover_period': timing_analysis.loc[timing_analysis['crossover_frequency'].idxmax(), 'sma_period'],
            'avg_crossover_frequency': timing_analysis['crossover_frequency'].mean(),
            'best_bullish_ratio': pattern_analysis['bullish_ratio'].max() if not pattern_analysis.empty and 'bullish_ratio' in pattern_analysis.columns else 0.0,
            'most_consistent_intervals': pattern_analysis.loc[pattern_analysis['interval_consistency'].idxmax(), 'sma_period'] if not pattern_analysis.empty and 'interval_consistency' in pattern_analysis.columns else periods[0] if periods else 0,
            'timing_analysis': timing_analysis.to_dict('records'),
            'pattern_analysis': pattern_analysis.to_dict('records') if not pattern_analysis.empty else []
        }
        
        return summary 