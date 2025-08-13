"""
Sample analysis script demonstrating the Stocks Research Platform.
This script shows how to use the platform for comprehensive stock analysis.
"""

import pandas as pd
import numpy as np
from stock_analyzer import StockAnalyzer
from data_fetcher import StockDataFetcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_sample_analysis():
    """Run a sample analysis on multiple stocks."""
    
    logger.info("Starting sample stock analysis...")
    
    # Initialize the analyzer
    analyzer = StockAnalyzer()
    
    # Sample stocks to analyze
    sample_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    # Custom SMA periods for analysis
    custom_sma_periods = [5, 10, 20, 50, 100, 200]
    
    try:
        # Analyze multiple stocks
        logger.info(f"Analyzing {len(sample_stocks)} stocks...")
        results = analyzer.analyze_multiple_stocks(
            sample_stocks, 
            custom_sma_periods, 
            period="2y"
        )
        
        # Display summary
        summary = analyzer.get_analysis_summary(results)
        print("\n" + "="*80)
        print("SAMPLE ANALYSIS SUMMARY")
        print("="*80)
        print(summary)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        plots = analyzer.generate_visualizations(results, save_plots=True, output_dir="sample_plots")
        
        # Export results
        logger.info("Exporting results...")
        analyzer.export_results(results, "sample_analysis_results", "excel")
        
        # Display detailed results for each stock
        print("\n" + "="*80)
        print("DETAILED RESULTS BY STOCK")
        print("="*80)
        
        for symbol in sample_stocks:
            if symbol in results and 'error' not in results[symbol]:
                stock_results = results[symbol]
                print(f"\n📊 {symbol} Analysis:")
                print(f"   - Best Touch Period: {stock_results['sma_summary']['best_touch_period']} days")
                print(f"   - Best Touch Frequency: {stock_results['sma_summary']['best_touch_frequency']:.4f}")
                print(f"   - Best Crossover Period: {stock_results['crossover_summary']['best_crossover_period']} days")
                print(f"   - Crossover Frequency: {stock_results['crossover_summary']['best_crossover_frequency']:.2f}/year")
        
        logger.info("Sample analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Sample analysis failed: {str(e)}")
        raise

def analyze_single_stock_detailed(symbol: str = 'AAPL'):
    """Perform detailed analysis of a single stock."""
    
    logger.info(f"Performing detailed analysis of {symbol}...")
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Extended SMA periods for detailed analysis
    extended_periods = list(range(5, 201, 5))  # 5, 10, 15, ..., 200
    
    try:
        # Perform analysis
        results = analyzer.analyze_stock(
            symbol, 
            extended_periods, 
            period="5y",  # Longer period for more data
            tolerance=0.005  # 0.5% tolerance
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: {symbol}")
        print(f"{'='*80}")
        
        # SMA Summary
        sma_summary = results['sma_summary']
        print(f"\n📈 SMA TOUCH ANALYSIS:")
        print(f"   - Total Periods Analyzed: {sma_summary['total_periods_analyzed']}")
        print(f"   - Best Touch Period: {sma_summary['best_touch_period']} days")
        print(f"   - Best Touch Frequency: {sma_summary['best_touch_frequency']:.4f}")
        print(f"   - Average Touch Frequency: {sma_summary['avg_touch_frequency']:.4f}")
        
        # Crossover Summary
        crossover_summary = results['crossover_summary']
        print(f"\n🔄 CROSSOVER ANALYSIS:")
        print(f"   - Best Crossover Period: {crossover_summary['best_crossover_period']} days")
        print(f"   - Best Crossover Frequency: {crossover_summary['best_crossover_frequency']:.2f} per year")
        print(f"   - Average Crossover Frequency: {crossover_summary['avg_crossover_frequency']:.2f} per year")
        print(f"   - Most Consistent Intervals: {crossover_summary['most_consistent_intervals']} days")
        
        # Top 5 optimal periods
        optimal_periods = results['optimal_periods'].head(5)
        print(f"\n🏆 TOP 5 OPTIMAL SMA PERIODS:")
        for _, row in optimal_periods.iterrows():
            print(f"   - {row['sma_period']:3d} days: Efficiency={row['touch_efficiency']:.6f}, "
                  f"Consistency={row['consistency_score']:.4f}")
        
        # Generate and save visualizations
        logger.info("Generating detailed visualizations...")
        plots = analyzer.generate_visualizations(
            results, 
            save_plots=True, 
            output_dir=f"{symbol}_detailed_plots"
        )
        
        # Export detailed results
        analyzer.export_results(results, f"{symbol}_detailed_analysis", "excel")
        
        logger.info(f"Detailed analysis of {symbol} completed!")
        
        return results
        
    except Exception as e:
        logger.error(f"Detailed analysis of {symbol} failed: {str(e)}")
        raise

def compare_stock_characteristics():
    """Compare characteristics across different stocks."""
    
    logger.info("Comparing stock characteristics...")
    
    # Different types of stocks
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    retail_stocks = ['AMZN', 'WMT', 'TGT', 'COST', 'HD']
    financial_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
    
    all_stocks = tech_stocks + retail_stocks + financial_stocks
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    try:
        # Analyze all stocks
        results = analyzer.analyze_multiple_stocks(all_stocks, period="2y")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for symbol in all_stocks:
            if symbol in results and 'error' not in results[symbol]:
                stock_results = results[symbol]
                
                # Determine stock category
                if symbol in tech_stocks:
                    category = 'Technology'
                elif symbol in retail_stocks:
                    category = 'Retail'
                else:
                    category = 'Financial'
                
                comparison_data.append({
                    'Symbol': symbol,
                    'Category': category,
                    'Best_Touch_Period': stock_results['sma_summary']['best_touch_period'],
                    'Best_Touch_Frequency': stock_results['sma_summary']['best_touch_frequency'],
                    'Best_Crossover_Period': stock_results['crossover_summary']['best_crossover_period'],
                    'Best_Crossover_Frequency': stock_results['crossover_summary']['best_crossover_frequency'],
                    'Avg_Touch_Frequency': stock_results['sma_summary']['avg_touch_frequency'],
                    'Avg_Crossover_Frequency': stock_results['crossover_summary']['avg_crossover_frequency']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison results
        print(f"\n{'='*100}")
        print("STOCK CHARACTERISTICS COMPARISON")
        print(f"{'='*100}")
        
        # By category
        for category in ['Technology', 'Retail', 'Financial']:
            cat_data = comparison_df[comparison_df['Category'] == category]
            print(f"\n📊 {category} Stocks:")
            print(cat_data[['Symbol', 'Best_Touch_Period', 'Best_Touch_Frequency', 
                           'Best_Crossover_Frequency']].to_string(index=False))
        
        # Top performers
        print(f"\n🏆 TOP PERFORMERS BY CATEGORY:")
        
        # Best touch frequency by category
        for category in ['Technology', 'Retail', 'Financial']:
            cat_data = comparison_df[comparison_df['Category'] == category]
            if not cat_data.empty:
                best_touch = cat_data.loc[cat_data['Best_Touch_Frequency'].idxmax()]
                print(f"   - {category} (Best Touch): {best_touch['Symbol']} "
                      f"({best_touch['Best_Touch_Frequency']:.4f})")
        
        # Best crossover frequency by category
        for category in ['Technology', 'Retail', 'Financial']:
            cat_data = comparison_df[comparison_df['Category'] == category]
            if not cat_data.empty:
                best_crossover = cat_data.loc[cat_data['Best_Crossover_Frequency'].idxmax()]
                print(f"   - {category} (Best Crossover): {best_crossover['Symbol']} "
                      f"({best_crossover['Best_Crossover_Frequency']:.2f}/year)")
        
        # Export comparison
        comparison_df.to_csv("stock_comparison.csv", index=False)
        logger.info("Stock comparison completed and exported!")
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"Stock comparison failed: {str(e)}")
        raise

def analyze_sma_optimization():
    """Analyze SMA optimization across different market conditions."""
    
    logger.info("Analyzing SMA optimization...")
    
    # Test different SMA periods
    test_periods = list(range(5, 301, 5))  # 5 to 300 days
    
    # Test stocks
    test_stocks = ['AAPL', 'SPY', 'QQQ']  # Apple, S&P 500 ETF, NASDAQ ETF
    
    analyzer = StockAnalyzer()
    
    try:
        optimization_results = {}
        
        for symbol in test_stocks:
            logger.info(f"Optimizing SMAs for {symbol}...")
            
            # Get stock data
            data_fetcher = StockDataFetcher()
            data = data_fetcher.get_stock_data(symbol, period="5y")
            
            # Find optimal periods
            optimal_periods = analyzer.sma_analyzer.find_sma_optimal_periods(
                data, (5, 300), step=5
            )
            
            optimization_results[symbol] = optimal_periods
            
            # Display top 10 optimal periods
            print(f"\n🔍 {symbol} - Top 10 Optimal SMA Periods:")
            top_10 = optimal_periods.head(10)
            for _, row in top_10.iterrows():
                print(f"   - {row['sma_period']:3d} days: "
                      f"Efficiency={row['touch_efficiency']:.6f}, "
                      f"Consistency={row['consistency_score']:.4f}")
        
        # Find common optimal periods across stocks
        common_periods = set()
        for symbol, periods_df in optimization_results.items():
            top_periods = set(periods_df.head(20)['sma_period'].tolist())
            if not common_periods:
                common_periods = top_periods
            else:
                common_periods = common_periods.intersection(top_periods)
        
        print(f"\n🎯 Common Optimal Periods Across All Stocks:")
        if common_periods:
            for period in sorted(common_periods):
                print(f"   - {period} days")
        else:
            print("   - No common optimal periods found")
        
        # Export optimization results
        with pd.ExcelWriter("sma_optimization_results.xlsx", engine='openpyxl') as writer:
            for symbol, periods_df in optimization_results.items():
                periods_df.to_excel(writer, sheet_name=symbol[:31], index=False)  # Excel sheet name limit
        
        logger.info("SMA optimization analysis completed!")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"SMA optimization analysis failed: {str(e)}")
        raise

def main():
    """Main function to run all sample analyses."""
    
    print("🚀 STOCKS RESEARCH PLATFORM - SAMPLE ANALYSIS")
    print("=" * 60)
    
    try:
        # Run basic sample analysis
        print("\n1️⃣ Running basic sample analysis...")
        run_sample_analysis()
        
        # Run detailed single stock analysis
        print("\n2️⃣ Running detailed single stock analysis...")
        analyze_single_stock_detailed('AAPL')
        
        # Compare stock characteristics
        print("\n3️⃣ Comparing stock characteristics...")
        compare_stock_characteristics()
        
        # Analyze SMA optimization
        print("\n4️⃣ Analyzing SMA optimization...")
        analyze_sma_optimization()
        
        print("\n✅ All sample analyses completed successfully!")
        print("\n📁 Check the generated files:")
        print("   - sample_analysis_results.xlsx")
        print("   - AAPL_detailed_analysis.xlsx")
        print("   - stock_comparison.csv")
        print("   - sma_optimization_results.xlsx")
        print("   - sample_plots/ (directory)")
        print("   - AAPL_detailed_plots/ (directory)")
        
    except Exception as e:
        logger.error(f"Sample analysis failed: {str(e)}")
        print(f"\n❌ Sample analysis failed: {str(e)}")

if __name__ == "__main__":
    main() 