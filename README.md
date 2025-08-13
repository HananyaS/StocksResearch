# Stocks Research Application

A comprehensive platform for analyzing stock movements and mathematical research, specifically focused on SMA (Simple Moving Average) analysis and crossover patterns.

## Features

### Core Analysis
- **SMA Analysis**: Calculate and analyze Simple Moving Averages for different periods (k)
- **Crossover Detection**: Identify when stock prices cross above/below SMA-k
- **Tendency Analysis**: Analyze how stocks tend to touch SMA-k as a function of k
- **Distribution Analysis**: Study the distribution of days between different crossover events

### Technical Capabilities
- Real-time stock data fetching via Yahoo Finance API
- Interactive visualizations with Plotly and Matplotlib
- Statistical analysis and pattern recognition
- Web-based interface with Streamlit
- Export capabilities for further analysis

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### Command Line Analysis
```bash
python stock_analyzer.py --symbol AAPL --sma_periods 20,50,200
```

## Project Structure

- `app.py` - Main Streamlit web application
- `stock_analyzer.py` - Core analysis engine
- `sma_analyzer.py` - SMA-specific analysis functions
- `crossover_analyzer.py` - Crossover detection and analysis
- `visualization.py` - Charting and plotting functions
- `data_fetcher.py` - Stock data retrieval
- `utils.py` - Utility functions and helpers

## Analysis Examples

- Compare how different stocks behave relative to various SMA periods
- Analyze the frequency and timing of SMA crossovers
- Study the distribution of days between crossover events
- Identify patterns in stock price movements around moving averages

## Contributing

Feel free to contribute by adding new analysis methods, improving visualizations, or enhancing the statistical models. 