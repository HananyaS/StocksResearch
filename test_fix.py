#!/usr/bin/env python3
"""
Test script to verify that the notna() error has been fixed.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        print("🔄 Testing imports...")
        
        from stock_analyzer import StockAnalyzer
        print("✅ StockAnalyzer imported successfully")
        
        from sma_analyzer import SMAAnalyzer
        print("✅ SMAAnalyzer imported successfully")
        
        from crossover_analyzer import CrossoverAnalyzer
        print("✅ CrossoverAnalyzer imported successfully")
        
        from data_fetcher import StockDataFetcher
        print("✅ StockDataFetcher imported successfully")
        
        from visualization import StockVisualizer
        print("✅ StockVisualizer imported successfully")
        
        print("\n🎉 All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_analyzer_creation():
    """Test that analyzers can be created without errors."""
    try:
        print("\n🔄 Testing analyzer creation...")
        
        from stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        print("✅ StockAnalyzer created successfully")
        
        from sma_analyzer import SMAAnalyzer
        sma_analyzer = SMAAnalyzer()
        print("✅ SMAAnalyzer created successfully")
        
        from crossover_analyzer import CrossoverAnalyzer
        crossover_analyzer = CrossoverAnalyzer()
        print("✅ CrossoverAnalyzer created successfully")
        
        print("\n🎉 All analyzers created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer creation error: {str(e)}")
        return False

def test_data_structures():
    """Test that data structures can be created without errors."""
    try:
        print("\n🔄 Testing data structure creation...")
        
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        print("✅ Sample data created successfully")
        
        # Test SMA calculation
        from sma_analyzer import SMAAnalyzer
        sma_analyzer = SMAAnalyzer()
        
        sma_data = sma_analyzer.calculate_multiple_smas(data, [20, 50])
        print("✅ SMA calculation successful")
        
        # Test touch analysis
        touch_analysis = sma_analyzer.analyze_sma_tendencies(sma_data, [20, 50])
        print("✅ Touch analysis successful")
        
        # Test summary creation
        summary = sma_analyzer.get_sma_summary_stats(sma_data, [20, 50])
        print("✅ Summary creation successful")
        
        print("\n🎉 All data structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data structure error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING STOCKS RESEARCH PLATFORM FIXES")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Analyzer Creation Test", test_analyzer_creation),
        ("Data Structure Test", test_data_structures)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The notna() error has been fixed.")
        print("\nYou can now run:")
        print("   python quick_start.py")
        print("   streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1) 