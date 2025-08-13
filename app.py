"""
Streamlit web application for Stocks Research Platform.
Provides an easy-to-use interface for SMA and crossover analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import os

# Import our analysis modules
from stock_analyzer import StockAnalyzer
from data_fetcher import StockDataFetcher

# Configure page
st.set_page_config(
    page_title="Stocks Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h3 {
        color: #f0f0f0;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .metric-card h2 {
        color: white;
        font-size: 2rem;
        margin: 0;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .analysis-section {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stocks Research Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced SMA Analysis & Crossover Research")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    # SMA periods
    st.sidebar.subheader("SMA Analysis")
    st.sidebar.markdown("""
    **What This Analyzes:**
    - **SMA Crossovers**: When price crosses above/below SMA
    - **Touch Patterns**: How often price touches different SMA periods
    - **Crossover Timing**: Days between crossover events
    - **Optimal Periods**: Best SMA periods for analysis
    """)
    
    sma_periods_input = st.sidebar.text_input(
        "SMA Periods (comma-separated)", 
        value="5,10,20,50,100,200",
        help="Enter SMA periods separated by commas"
    )
    
    # Parse SMA periods
    try:
        sma_periods = [int(p.strip()) for p in sma_periods_input.split(",")]
    except:
        st.sidebar.error("Invalid SMA periods format. Using default periods.")
        sma_periods = [5, 10, 20, 50, 100, 200]
    
    # Data period
    period = st.sidebar.selectbox(
        "Data Period",
        options=["1y", "2y", "5y", "10y", "max"],
        index=1,
        help="Select the time period for analysis"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Analyze Stock", type="primary")
    
    # Main content area
    if analyze_button and symbol:
        with st.spinner("Analyzing stock data..."):
            try:
                # Initialize analyzer
                analyzer = StockAnalyzer()
                
                # Perform analysis
                results = analyzer.analyze_stock(
                    symbol.upper(), 
                    sma_periods, 
                    period
                )
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.symbol = symbol.upper()
                
                st.success(f"✅ Analysis completed for {symbol.upper()}!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please check the stock symbol and try again.")
                return
    
    # Display results if available
    if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)
    
    # Quick start guide
    if not hasattr(st.session_state, 'analysis_results'):
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Enter Stock Symbol**
            - Use standard ticker symbols (e.g., AAPL, MSFT, TSLA)
            - Symbols are case-insensitive
            """)
        
        with col2:
            st.markdown("""
            **2. Configure Analysis**
            - Select SMA periods to analyze
            - Choose data time period
            - Adjust touch tolerance
            """)
        
        with col3:
            st.markdown("""
            **3. Run Analysis**
            - Click "Analyze Stock" button
            - View comprehensive results
            - Explore interactive charts
            """)
        
        st.markdown("---")
        st.markdown("### 📊 What This Platform Analyzes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **SMA Crossover Analysis:**
            - **Actual Price Crosses**: When price crosses above/below SMA
            - **Crossover Frequency**: How often crossovers occur
            - **Timing Patterns**: Days between crossover events
            - **Bullish vs Bearish**: Direction of crossovers
            """)
        
        with col2:
            st.markdown("""
            **Touch Pattern Analysis:**
            - **Touch Frequency**: How often price touches SMA-k
            - **Distance Patterns**: Price distance from SMA
            - **Optimal Periods**: Best SMA periods for analysis
            - **Consistency Metrics**: Reliability of patterns
            """)

def display_analysis_results(results):
    """Display comprehensive analysis results."""
    
    symbol = results['symbol']
    touch_analysis = results['touch_analysis']
    crossover_analysis = results['crossover_analysis']
    sma_summary = results['sma_summary']
    crossover_summary = results['crossover_summary']
    
    # Header with stock info
    st.markdown(f"## 📊 Analysis Results for {symbol}")
    st.markdown("*Analyzing SMA crossovers, touch patterns, and optimal periods*")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Touch Period</h3>
            <h2>{sma_summary['best_touch_period']} days</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Touch Frequency</h3>
            <h2>{sma_summary['best_touch_frequency']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Crossover Period</h3>
            <h2>{crossover_summary['best_crossover_period']} days</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Crossover Frequency</h3>
            <h2>{crossover_summary['best_crossover_frequency']:.1f}/year</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Stock price chart with SMAs
    st.markdown("### 📈 Stock Price with SMAs")
    
    # Create price chart
    price_fig = create_price_chart(results['stock_data'], results['sma_periods_analyzed'], symbol)
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 SMA Touch Analysis", 
        "🔄 Crossover Analysis", 
        "📈 Optimal Periods", 
        "📋 Summary Report"
    ])
    
    with tab1:
        display_sma_touch_analysis(touch_analysis, symbol)
    
    with tab2:
        display_crossover_analysis(crossover_analysis, symbol)
    
    with tab3:
        display_optimal_periods_analysis(results['optimal_periods'], symbol)
    
    with tab4:
        display_summary_report(results, symbol)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            export_to_csv(results)
    
    with col2:
        if st.button("📈 Export to Excel"):
            export_to_excel(results)
    
    with col3:
        if st.button("📄 Export to JSON"):
            export_to_json(results)

def create_price_chart(data, sma_periods, symbol):
    """Create interactive stock price chart with SMAs."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{symbol} - Stock Price with SMAs", "Volume"),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#2ca02c',
            decreasing_line_color='#d62728'
        ),
        row=1, col=1
    )
    
    # Add SMAs
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, period in enumerate(sma_periods):
        sma_col = f'SMA_{period}'
        if sma_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[sma_col],
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(width=2, color=colors[i % len(colors)]),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # Volume bars
    if 'Volume' in data.columns:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Stock Price Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def display_sma_touch_analysis(touch_analysis, symbol):
    """Display SMA touch analysis results."""
    
    st.markdown(f"### {symbol} - SMA Touch Analysis")
    
    # Create touch analysis charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Touch Frequency by Period", "Touch Percentage",
                       "Average Distance from SMA", "Distance Standard Deviation"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Touch frequency
    fig.add_trace(
        go.Scatter(
            x=touch_analysis['sma_period'],
            y=touch_analysis['touch_frequency'],
            mode='lines+markers',
            name='Touch Frequency',
            line=dict(color='#1f77b4', width=3)
        ),
        row=1, col=1
    )
    
    # Touch percentage
    fig.add_trace(
        go.Bar(
            x=touch_analysis['sma_period'],
            y=touch_analysis['touch_percentage'],
            name='Touch Percentage',
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    
    # Average distance
    fig.add_trace(
        go.Scatter(
            x=touch_analysis['sma_period'],
            y=touch_analysis['avg_distance'],
            mode='lines+markers',
            name='Avg Distance',
            line=dict(color='#7f7f7f', width=3)
        ),
        row=2, col=1
    )
    
    # Distance std
    fig.add_trace(
        go.Scatter(
            x=touch_analysis['sma_period'],
            y=touch_analysis['std_distance'],
            mode='lines+markers',
            name='Distance Std',
            line=dict(color='#2ca02c', width=3)
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Touch analysis table
    st.markdown("#### Touch Analysis Details")
    st.dataframe(touch_analysis, use_container_width=True)

def display_crossover_analysis(crossover_analysis, symbol):
    """Display crossover analysis results."""
    
    st.markdown(f"### {symbol} - Crossover Analysis")
    
    # Create crossover timing charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Crossover Frequency", "Bullish vs Bearish Crosses",
                       "Average Interval Between Crosses", "Total Crosses by Period"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Crossover frequency
    fig.add_trace(
        go.Scatter(
            x=crossover_analysis['sma_period'],
            y=crossover_analysis['crossover_frequency'],
            mode='lines+markers',
            name='Crossover Frequency',
            line=dict(color='#1f77b4', width=3)
        ),
        row=1, col=1
    )
    
    # Bullish vs Bearish
    fig.add_trace(
        go.Bar(
            x=crossover_analysis['sma_period'],
            y=crossover_analysis['total_bullish_crosses'],
            name='Bullish Crosses',
            marker_color='#2ca02c'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=crossover_analysis['sma_period'],
            y=crossover_analysis['total_bearish_crosses'],
            name='Bearish Crosses',
            marker_color='#d62728'
        ),
        row=1, col=2
    )
    
    # Average interval
    fig.add_trace(
        go.Scatter(
            x=crossover_analysis['sma_period'],
            y=crossover_analysis['bullish_interval_stats'].apply(lambda x: x['mean']),
            mode='lines+markers',
            name='Avg Interval',
            line=dict(color='#7f7f7f', width=3)
        ),
        row=2, col=1
    )
    
    # Total crosses
    fig.add_trace(
        go.Bar(
            x=crossover_analysis['sma_period'],
            y=crossover_analysis['total_crosses'],
            name='Total Crosses',
            marker_color='#ff7f0e'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Crossover analysis table
    st.markdown("#### Crossover Analysis Details")
    st.dataframe(crossover_analysis, use_container_width=True)

def display_optimal_periods_analysis(optimal_periods, symbol):
    """Display optimal periods analysis."""
    
    st.markdown(f"### {symbol} - Optimal SMA Periods Analysis")
    
    # Create optimal periods charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Touch Efficiency by Period", "Consistency Score",
                       "Touch Frequency vs Period", "Top 10 Efficiency Ranking"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Touch efficiency
    fig.add_trace(
        go.Scatter(
            x=optimal_periods['sma_period'],
            y=optimal_periods['touch_efficiency'],
            mode='lines+markers',
            name='Touch Efficiency',
            line=dict(color='#1f77b4', width=3)
        ),
        row=1, col=1
    )
    
    # Consistency score
    fig.add_trace(
        go.Scatter(
            x=optimal_periods['sma_period'],
            y=optimal_periods['consistency_score'],
            mode='lines+markers',
            name='Consistency Score',
            line=dict(color='#ff7f0e', width=3)
        ),
        row=1, col=2
    )
    
    # Touch frequency vs period
    fig.add_trace(
        go.Scatter(
            x=optimal_periods['sma_period'],
            y=optimal_periods['touch_frequency'],
            mode='lines+markers',
            name='Touch Frequency',
            line=dict(color='#7f7f7f', width=3)
        ),
        row=2, col=1
    )
    
    # Top 10 ranking
    top_10 = optimal_periods.head(10)
    fig.add_trace(
        go.Bar(
            x=top_10['sma_period'],
            y=top_10['touch_efficiency'],
            name='Top 10 Efficiency',
            marker_color='#2ca02c'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 optimal periods table
    st.markdown("#### Top 10 Optimal SMA Periods")
    st.dataframe(top_10.head(10), use_container_width=True)

def display_summary_report(results, symbol):
    """Display comprehensive summary report."""
    
    st.markdown(f"### {symbol} - Analysis Summary Report")
    
    # Analysis metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Analysis Information:**
        - **Symbol:** {results['symbol']}
        - **Analysis Date:** {results['analysis_date']}
        - **Data Period:** {results['data_period']}
        - **Data Points:** {results['data_points']}
        - **SMA Periods Analyzed:** {', '.join(map(str, results['sma_periods_analyzed']))}
        - **Analysis Type:** SMA Crossover Analysis (Actual Price Crosses)
        """)
    
    with col2:
        st.markdown(f"""
        **Key Findings:**
        - **Best Touch Period:** {results['sma_summary']['best_touch_period']} days
        - **Best Touch Frequency:** {results['sma_summary']['best_touch_frequency']:.4f}
        - **Best Crossover Period:** {results['crossover_summary']['best_crossover_period']} days
        - **Best Crossover Frequency:** {results['crossover_summary']['best_crossover_frequency']:.2f} per year
        """)
    
    # Recommendations
    st.markdown("#### 📋 Investment Recommendations")
    
    touch_summary = results['sma_summary']
    crossover_summary = results['crossover_summary']
    
    st.markdown(f"""
    **SMA Strategy:**
    - **Primary SMA Period:** {touch_summary['best_touch_period']} days
        - This period shows the highest frequency of price touches
        - Touch frequency: {touch_summary['best_touch_frequency']:.4f}
        - Consistency score: {results['optimal_periods'].loc[results['optimal_periods']['sma_period'] == touch_summary['best_touch_period'], 'consistency_score'].iloc[0]:.4f}
    
    **Crossover Strategy:**
    - **Optimal Crossover Period:** {crossover_summary['best_crossover_period']} days
        - Highest crossover frequency: {crossover_summary['best_crossover_frequency']:.2f} per year
        - Most consistent intervals: {crossover_summary['most_consistent_intervals']} days
    
    **Risk Management:**
    - Use {touch_summary['best_touch_period']}-day SMA for support/resistance levels
    - Monitor crossovers at {crossover_summary['best_crossover_period']}-day SMA for trend changes
    - Focus on actual price crosses above/below SMA, not arbitrary proximity
    """)

def export_to_csv(results):
    """Export results to CSV."""
    try:
        # Create export data
        export_data = []
        for period in results['sma_periods_analyzed']:
            touch_row = results['touch_analysis'][results['touch_analysis']['sma_period'] == period].iloc[0]
            crossover_row = results['crossover_analysis'][results['crossover_analysis']['sma_period'] == period].iloc[0]
            
            export_data.append({
                'SMA_Period': period,
                'Touch_Frequency': touch_row['touch_frequency'],
                'Touch_Percentage': touch_row['touch_percentage'],
                'Avg_Distance': touch_row['avg_distance'],
                'Crossover_Frequency': crossover_row['crossover_frequency'],
                'Total_Crosses': crossover_row['total_crosses'],
                'Bullish_Crosses': crossover_row['total_bullish_crosses'],
                'Bearish_Crosses': crossover_row['total_bearish_crosses']
            })
        
        export_df = pd.DataFrame(export_data)
        
        # Download button
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{results['symbol']}_analysis.csv",
            mime="text/csv"
        )
        
        st.success("✅ CSV export ready!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_to_excel(results):
    """Export results to Excel."""
    try:
        # Create Excel file with multiple sheets
        from io import BytesIO
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results['touch_analysis'].to_excel(writer, sheet_name='Touch_Analysis', index=False)
            results['crossover_analysis'].to_excel(writer, sheet_name='Crossover_Analysis', index=False)
            results['optimal_periods'].to_excel(writer, sheet_name='Optimal_Periods', index=False)
        
        output.seek(0)
        
        # Download button
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name=f"{results['symbol']}_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Excel export ready!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_to_json(results):
    """Export results to JSON."""
    try:
        import json
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if key == 'stock_data':
                # Skip large data for JSON export
                json_results[key] = "Data too large for JSON export"
            else:
                json_results[key] = value
        
        json_str = json.dumps(json_results, indent=2, default=str)
        
        # Download button
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{results['symbol']}_analysis.json",
            mime="application/json"
        )
        
        st.success("✅ JSON export ready!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    main() 