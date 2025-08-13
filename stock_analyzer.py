"""
Main stock analyzer module that orchestrates all analysis components.
Provides a unified interface for SMA and crossover analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import argparse
from datetime import datetime, timedelta

from data_fetcher import StockDataFetcher
from sma_analyzer import SMAAnalyzer
from crossover_analyzer import CrossoverAnalyzer
from visualization import StockVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Main class for comprehensive stock analysis."""
    
    def __init__(self, cache_duration_hours: int = 24):
        """
        Initialize the stock analyzer.
        
        Args:
            cache_duration_hours: Duration to cache stock data
        """
        self.data_fetcher = StockDataFetcher(cache_duration_hours)
        self.sma_analyzer = SMAAnalyzer()
        self.crossover_analyzer = CrossoverAnalyzer()
        self.visualizer = StockVisualizer()
        
        # Default SMA periods to analyze
        self.default_sma_periods = [5, 10, 20, 50, 100, 200]
    
    def analyze_stock(self, symbol: str, sma_periods: Optional[List[int]] = None,
                     period: str = "2y") -> Dict:
        """
        Perform comprehensive analysis of a single stock.
        
        Args:
            symbol: Stock ticker symbol
            sma_periods: List of SMA periods to analyze (default: [5, 10, 20, 50, 100, 200])
            period: Data period to fetch
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if sma_periods is None:
            sma_periods = self.default_sma_periods
        
        logger.info(f"Starting comprehensive analysis of {symbol}")
        
        try:
            # Fetch stock data
            data = self.data_fetcher.get_stock_data(symbol, period)
            logger.info(f"Fetched {len(data)} data points for {symbol}")
            
            # Calculate SMAs
            data_with_smas = self.sma_analyzer.calculate_multiple_smas(data, sma_periods)
            
            # Perform SMA touch analysis (using default tolerance for proximity analysis)
            touch_analysis = self.sma_analyzer.analyze_sma_tendencies(
                data_with_smas, sma_periods, tolerance=0.01  # 1% default for proximity analysis
            )
            
            # Perform crossover analysis
            crossover_analysis = self.crossover_analyzer.analyze_multiple_sma_crossovers(
                data_with_smas, sma_periods
            )
            
            # Analyze crossover distributions
            distribution_analysis = self.crossover_analyzer.analyze_crossover_distributions(
                data_with_smas, sma_periods
            )
            
            # Find optimal SMA periods
            optimal_periods = self.sma_analyzer.find_sma_optimal_periods(
                data_with_smas, (min(sma_periods), max(sma_periods))
            )
            
            # Get summary statistics with error handling
            try:
                sma_summary = self.sma_analyzer.get_sma_summary_stats(data_with_smas, sma_periods)
            except Exception as e:
                logger.warning(f"Error getting SMA summary: {str(e)}")
                sma_summary = {
                    'total_periods_analyzed': len(sma_periods),
                    'periods_with_errors': 0,
                    'best_touch_period': sma_periods[0] if sma_periods else 0,
                    'best_touch_frequency': 0.0,
                    'avg_touch_frequency': 0.0
                }
            
            try:
                crossover_summary = self.crossover_analyzer.get_crossover_summary(data_with_smas, sma_periods)
            except Exception as e:
                logger.warning(f"Error getting crossover summary: {str(e)}")
                crossover_summary = {
                    'total_periods_analyzed': len(sma_periods),
                    'periods_with_errors': 0,
                    'best_crossover_period': sma_periods[0] if sma_periods else 0,
                    'best_crossover_frequency': 0.0,
                    'avg_crossover_frequency': 0.0,
                    'best_bullish_ratio': 0.0,
                    'most_consistent_intervals': sma_periods[0] if sma_periods else 0
                }
            
            # Compile results
            analysis_results = {
                'symbol': symbol,
                'analysis_date': datetime.now().isoformat(),
                'data_period': period,
                'data_points': len(data),
                'sma_periods_analyzed': sma_periods,
                'stock_data': data_with_smas,
                'touch_analysis': touch_analysis,
                'crossover_analysis': crossover_analysis,
                'distribution_analysis': distribution_analysis,
                'optimal_periods': optimal_periods,
                'sma_summary': sma_summary,
                'crossover_summary': crossover_summary
            }
            
            logger.info(f"Analysis completed for {symbol}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            raise
    
    def analyze_multiple_stocks(self, symbols: List[str], 
                               sma_periods: Optional[List[int]] = None,
                               period: str = "2y") -> Dict:
        """
        Analyze multiple stocks and compare results.
        
        Args:
            symbols: List of stock ticker symbols
            sma_periods: List of SMA periods to analyze
            period: Data period to fetch
            
        Returns:
            Dictionary with analysis results for all stocks
        """
        if sma_periods is None:
            sma_periods = self.default_sma_periods
        
        logger.info(f"Starting analysis of {len(symbols)} stocks")
        
        results = {}
        comparison_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Analyzing {symbol}...")
                analysis = self.analyze_stock(symbol, sma_periods, period)
                results[symbol] = analysis
                
                # Extract key metrics for comparison
                comparison_metrics = {
                    'symbol': symbol,
                    'best_touch_period': analysis['sma_summary']['best_touch_period'],
                    'best_touch_frequency': analysis['sma_summary']['best_touch_frequency'],
                    'best_crossover_period': analysis['crossover_summary']['best_crossover_period'],
                    'best_crossover_frequency': analysis['crossover_summary']['best_crossover_frequency'],
                    'avg_touch_frequency': analysis['sma_summary']['avg_touch_frequency'],
                    'avg_crossover_frequency': analysis['crossover_summary']['avg_crossover_frequency']
                }
                comparison_data.append(comparison_metrics)
                
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add overall results
        results['comparison'] = comparison_df
        results['analysis_summary'] = {
            'total_stocks': len(symbols),
            'successful_analyses': len([r for r in results.values() if 'error' not in r]),
            'failed_analyses': len([r for r in results.values() if 'error' in r]),
            'analysis_date': datetime.now().isoformat()
        }
        
        return results
    
    def generate_visualizations(self, analysis_results: Dict, 
                               save_plots: bool = False,
                               output_dir: str = "plots") -> Dict:
        """
        Generate all visualizations for the analysis results.
        
        Args:
            analysis_results: Results from analyze_stock or analyze_multiple_stocks
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with all generated plots
        """
        plots = {}
        
        try:
            if 'symbol' in analysis_results:
                # Single stock analysis
                plots.update(self._generate_single_stock_plots(analysis_results, save_plots, output_dir))
            else:
                # Multiple stocks analysis
                plots.update(self._generate_multiple_stocks_plots(analysis_results, save_plots, output_dir))
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        return plots
    
    def _generate_single_stock_plots(self, analysis: Dict, save_plots: bool, output_dir: str) -> Dict:
        """Generate plots for single stock analysis."""
        plots = {}
        symbol = analysis['symbol']
        
        try:
            # Stock price with SMAs
            plots['price_chart'] = self.visualizer.plot_stock_with_smas(
                analysis['stock_data'], 
                analysis['sma_periods_analyzed'],
                f"{symbol} - Stock Price with SMAs"
            )
            
            # SMA touch analysis
            plots['touch_analysis'] = self.visualizer.plot_sma_touch_analysis(
                analysis['touch_analysis'],
                f"{symbol} - SMA Touch Analysis"
            )
            
            # Crossover timing analysis
            plots['crossover_timing'] = self.visualizer.plot_crossover_timing_analysis(
                analysis['crossover_analysis'],
                f"{symbol} - Crossover Timing Analysis"
            )
            
            # Interval distributions
            plots['interval_distributions'] = self.visualizer.plot_interval_distributions(
                analysis['distribution_analysis'],
                f"{symbol} - Interval Distribution Analysis"
            )
            
            # Optimal periods analysis
            plots['optimal_periods'] = self.visualizer.plot_optimal_periods_analysis(
                analysis['optimal_periods'],
                f"{symbol} - Optimal SMA Periods"
            )
            
            # Comprehensive dashboard
            plots['dashboard'] = self.visualizer.create_dashboard(
                analysis['stock_data'],
                analysis['sma_periods_analyzed'],
                analysis['touch_analysis'],
                analysis['crossover_analysis'],
                f"{symbol} - Analysis Dashboard"
            )
            
            # Save plots if requested
            if save_plots:
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                for plot_name, plot_fig in plots.items():
                    filename = os.path.join(output_dir, f"{symbol}_{plot_name}.html")
                    self.visualizer.save_plot(plot_fig, filename, 'html')
            
        except Exception as e:
            logger.error(f"Error generating plots for {symbol}: {str(e)}")
        
        return plots
    
    def _generate_multiple_stocks_plots(self, analysis: Dict, save_plots: bool, output_dir: str) -> Dict:
        """Generate plots for multiple stocks analysis."""
        plots = {}
        
        try:
            # Comparison plots
            if 'comparison' in analysis and not analysis['comparison'].empty:
                comparison_df = analysis['comparison']
                
                # Touch frequency comparison
                plots['touch_comparison'] = self.visualizer.plot_sma_touch_analysis(
                    comparison_df[['symbol', 'best_touch_frequency', 'avg_touch_frequency']],
                    "Multi-Stock Touch Frequency Comparison"
                )
                
                # Crossover frequency comparison
                plots['crossover_comparison'] = self.visualizer.plot_crossover_timing_analysis(
                    comparison_df[['symbol', 'best_crossover_frequency', 'avg_crossover_frequency']],
                    "Multi-Stock Crossover Frequency Comparison"
                )
            
            # Individual stock plots
            for symbol, stock_analysis in analysis.items():
                if symbol not in ['comparison', 'analysis_summary'] and 'error' not in stock_analysis:
                    stock_plots = self._generate_single_stock_plots(stock_analysis, save_plots, output_dir)
                    plots[f"{symbol}_plots"] = stock_plots
            
        except Exception as e:
            logger.error(f"Error generating multiple stocks plots: {str(e)}")
        
        return plots
    
    def export_results(self, analysis_results: Dict, output_file: str, format: str = 'csv'):
        """
        Export analysis results to file.
        
        Args:
            analysis_results: Analysis results to export
            output_file: Output filename
            format: Export format ('csv', 'excel', 'json')
        """
        try:
            if format == 'csv':
                # Export key metrics to CSV
                if 'symbol' in analysis_results:
                    # Single stock
                    touch_df = analysis_results['touch_analysis']
                    crossover_df = analysis_results['crossover_analysis']
                    
                    # Combine key metrics
                    export_df = pd.merge(touch_df, crossover_df, on='sma_period', suffixes=('_touch', '_crossover'))
                    export_df.to_csv(output_file, index=False)
                else:
                    # Multiple stocks
                    comparison_df = analysis_results['comparison']
                    comparison_df.to_csv(output_file, index=False)
                    
            elif format == 'excel':
                # Export to Excel with multiple sheets
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    if 'symbol' in analysis_results:
                        # Single stock
                        analysis_results['touch_analysis'].to_excel(writer, sheet_name='Touch_Analysis', index=False)
                        analysis_results['crossover_analysis'].to_excel(writer, sheet_name='Crossover_Analysis', index=False)
                        analysis_results['optimal_periods'].to_excel(writer, sheet_name='Optimal_Periods', index=False)
                    else:
                        # Multiple stocks
                        analysis_results['comparison'].to_excel(writer, sheet_name='Comparison', index=False)
                        
            elif format == 'json':
                # Export to JSON
                import json
                with open(output_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
    
    def get_analysis_summary(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable summary of the analysis.
        
        Args:
            analysis_results: Analysis results
            
        Returns:
            Formatted summary string
        """
        if 'symbol' in analysis_results:
            # Single stock summary
            symbol = analysis_results['symbol']
            touch_summary = analysis_results['sma_summary']
            crossover_summary = analysis_results['crossover_summary']
            
            summary = f"""
=== STOCK ANALYSIS SUMMARY ===
Symbol: {symbol}
Analysis Date: {analysis_results['analysis_date']}
Data Period: {analysis_results['data_period']}
Data Points: {analysis_results['data_points']}

SMA TOUCH ANALYSIS:
- Best Touch Period: {touch_summary['best_touch_period']}
- Best Touch Frequency: {touch_summary['best_touch_frequency']:.4f}
- Average Touch Frequency: {touch_summary['avg_touch_frequency']:.4f}

CROSSOVER ANALYSIS:
- Best Crossover Period: {crossover_summary['best_crossover_period']}
- Best Crossover Frequency: {crossover_summary['best_crossover_frequency']:.2f} per year
- Average Crossover Frequency: {crossover_summary['avg_crossover_frequency']:.2f} per year

RECOMMENDATIONS:
- Optimal SMA Period: {touch_summary['best_touch_period']} days
- Most Consistent: {crossover_summary['most_consistent_intervals']} days
"""
        else:
            # Multiple stocks summary
            summary_data = analysis_results['analysis_summary']
            comparison = analysis_results['comparison']
            
            summary = f"""
=== MULTI-STOCK ANALYSIS SUMMARY ===
Total Stocks: {summary_data['total_stocks']}
Successful Analyses: {summary_data['successful_analyses']}
Failed Analyses: {summary_data['failed_analyses']}
Analysis Date: {summary_data['analysis_date']}

TOP PERFORMERS:
{comparison.nlargest(3, 'best_touch_frequency')[['symbol', 'best_touch_frequency', 'best_touch_period']].to_string()}

CROSSOVER LEADERS:
{comparison.nlargest(3, 'best_crossover_frequency')[['symbol', 'best_crossover_frequency', 'best_crossover_period']].to_string()}
"""
        
        return summary


def main():
    """Command line interface for the stock analyzer."""
    parser = argparse.ArgumentParser(description='Stock Analysis Tool')
    parser.add_argument('--symbol', '-s', required=True, help='Stock ticker symbol to analyze')
    parser.add_argument('--sma-periods', '-p', nargs='+', type=int, 
                       default=[5, 10, 20, 50, 100, 200], help='SMA periods to analyze')
    parser.add_argument('--period', '-t', default='2y', help='Data period to fetch')
    parser.add_argument('--output', '-o', default='analysis_results', help='Output filename')
    parser.add_argument('--format', '-f', choices=['csv', 'excel', 'json'], default='csv',
                       help='Output format')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--plot-dir', default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = StockAnalyzer()
        
        # Perform analysis
        logger.info(f"Analyzing {args.symbol}...")
        results = analyzer.analyze_stock(
            args.symbol, 
            args.sma_periods, 
            args.period
        )
        
        # Generate visualizations
        plots = analyzer.generate_visualizations(results, args.save_plots, args.plot_dir)
        
        # Export results
        output_file = f"{args.output}.{args.format}"
        analyzer.export_results(results, output_file, args.format)
        
        # Print summary
        summary = analyzer.get_analysis_summary(results)
        print(summary)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 