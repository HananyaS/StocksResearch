#!/usr/bin/env python3
"""
Quick Start Script for the Stocks Research Platform
Run this script to get started with basic stock analysis.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analyzer import StockAnalyzer
from config import config

def quick_analysis():
    """Run a quick analysis on a popular stock."""
    
    print("🚀 STOCKS RESEARCH PLATFORM - QUICK START")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Quick analysis on Apple
    symbol = "AAPL"
    print(f"\n📊 Analyzing {symbol}...")
    
    try:
        # Perform analysis with default settings
        results = analyzer.analyze_stock(symbol)
        
        # Display key results
        print(f"\n✅ Analysis completed for {symbol}!")
        print(f"\n📈 Key Findings:")
        print(f"   - Best Touch Period: {results['sma_summary']['best_touch_period']} days")
        print(f"   - Touch Frequency: {results['sma_summary']['best_touch_frequency']:.4f}")
        print(f"   - Best Crossover Period: {results['crossover_summary']['best_crossover_period']} days")
        print(f"   - Crossover Frequency: {results['crossover_summary']['best_crossover_frequency']:.2f}/year")
        
        # Generate basic visualization
        print(f"\n📊 Generating visualization...")
        plots = analyzer.generate_visualizations(results, save_plots=True, output_dir="quick_start_plots")
        
        # Export results
        print(f"\n💾 Exporting results...")
        analyzer.export_results(results, f"{symbol}_quick_analysis", "csv")
        
        print(f"\n🎉 Quick analysis completed!")
        print(f"📁 Check the generated files:")
        print(f"   - {symbol}_quick_analysis.csv")
        print(f"   - quick_start_plots/ directory")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("💡 Make sure you have an internet connection and the required packages installed.")
        return None

def install_requirements():
    """Install required packages."""
    
    print("📦 Installing required packages...")
    
    try:
        import subprocess
        import sys
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install requirements: {str(e)}")
        print("💡 Please install the requirements manually:")
        print("   pip install -r requirements.txt")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'scipy', 'scikit-learn', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def show_usage_examples():
    """Show usage examples."""
    
    print("\n📚 USAGE EXAMPLES")
    print("=" * 30)
    
    print("\n1️⃣ Command Line Analysis:")
    print("   python stock_analyzer.py --symbol AAPL --sma-periods 20 50 200")
    
    print("\n2️⃣ Web Interface:")
    print("   streamlit run app.py")
    
    print("\n3️⃣ Python Script:")
    print("   from stock_analyzer import StockAnalyzer")
    print("   analyzer = StockAnalyzer()")
    print("   results = analyzer.analyze_stock('AAPL')")
    
    print("\n4️⃣ Sample Analysis:")
    print("   python sample_analysis.py")
    
    print("\n5️⃣ Configuration:")
    print("   python config.py")

def main():
    """Main function."""
    
    print("🎯 Welcome to the Stocks Research Platform!")
    print("This platform helps you analyze stock movements and SMA patterns.")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        install_choice = input("Would you like to install requirements now? (y/n): ").lower()
        if install_choice == 'y':
            if install_requirements():
                print("✅ Dependencies installed! You can now run the analysis.")
            else:
                print("❌ Failed to install dependencies. Please install manually.")
                return
        else:
            print("💡 Please install dependencies manually and try again.")
            return
    
    # Show menu
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. 🚀 Quick Analysis (AAPL)")
        print("2. 📚 Show Usage Examples")
        print("3. ⚙️  Show Configuration")
        print("4. 🔍 Check Dependencies")
        print("5. 📦 Install Requirements")
        print("6. 🚪 Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            quick_analysis()
            
        elif choice == '2':
            show_usage_examples()
            
        elif choice == '3':
            print(config.get_config_summary())
            
        elif choice == '4':
            check_dependencies()
            
        elif choice == '5':
            install_requirements()
            
        elif choice == '6':
            print("\n👋 Goodbye! Happy trading!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy trading!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("💡 Please check your setup and try again.") 