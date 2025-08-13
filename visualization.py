"""
Visualization module for stock analysis and SMA research.
Provides comprehensive charting capabilities for analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import logging

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class StockVisualizer:
    """Comprehensive visualization tools for stock analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = {
            'price': '#1f77b4',
            'sma': '#ff7f0e',
            'volume': '#2ca02c',
            'bullish': '#2ca02c',
            'bearish': '#d62728',
            'neutral': '#7f7f7f'
        }
    
    def plot_stock_with_smas(self, data: pd.DataFrame, sma_periods: List[int],
                            title: str = "Stock Price with SMAs",
                            show_volume: bool = True) -> go.Figure:
        """
        Create an interactive plot of stock price with multiple SMAs.
        
        Args:
            data: DataFrame with price and SMA data
            sma_periods: List of SMA periods to plot
            title: Plot title
            show_volume: Whether to show volume bars
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2 if show_volume else 1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, "Volume") if show_volume else (title,),
            row_heights=[0.7, 0.3] if show_volume else [1.0]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Add SMAs
        for period in sma_periods:
            sma_col = f'SMA_{period}'
            if sma_col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[sma_col],
                        mode='lines',
                        name=f'SMA {period}',
                        line=dict(width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add volume if requested
        if show_volume and 'Volume' in data.columns:
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
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_sma_touch_analysis(self, touch_analysis: pd.DataFrame,
                               title: str = "SMA Touch Analysis") -> go.Figure:
        """
        Plot SMA touch frequency analysis.
        
        Args:
            touch_analysis: DataFrame with touch analysis results
            title: Plot title
            
        Returns:
            Plotly figure object
        """
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
                line=dict(color=self.colors['price'], width=3)
            ),
            row=1, col=1
        )
        
        # Touch percentage
        fig.add_trace(
            go.Bar(
                x=touch_analysis['sma_period'],
                y=touch_analysis['touch_percentage'],
                name='Touch Percentage',
                marker_color=self.colors['sma']
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
                line=dict(color=self.colors['neutral'], width=3)
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
                line=dict(color=self.colors['volume'], width=3)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_crossover_timing_analysis(self, timing_analysis: pd.DataFrame,
                                     title: str = "Crossover Timing Analysis") -> go.Figure:
        """
        Plot crossover timing analysis results.
        
        Args:
            timing_analysis: DataFrame with timing analysis results
            title: Plot title
            
        Returns:
            Plotly figure object
        """
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
                x=timing_analysis['sma_period'],
                y=timing_analysis['crossover_frequency'],
                mode='lines+markers',
                name='Crossover Frequency',
                line=dict(color=self.colors['price'], width=3)
            ),
            row=1, col=1
        )
        
        # Bullish vs Bearish
        fig.add_trace(
            go.Bar(
                x=timing_analysis['sma_period'],
                y=timing_analysis['total_bullish_crosses'],
                name='Bullish Crosses',
                marker_color=self.colors['bullish']
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=timing_analysis['sma_period'],
                y=timing_analysis['total_bearish_crosses'],
                name='Bearish Crosses',
                marker_color=self.colors['bearish']
            ),
            row=1, col=2
        )
        
        # Average interval
        fig.add_trace(
            go.Scatter(
                x=timing_analysis['sma_period'],
                y=timing_analysis['bullish_interval_stats'].apply(lambda x: x['mean']),
                mode='lines+markers',
                name='Avg Interval',
                line=dict(color=self.colors['neutral'], width=3)
            ),
            row=2, col=1
        )
        
        # Total crosses
        fig.add_trace(
            go.Bar(
                x=timing_analysis['sma_period'],
                y=timing_analysis['total_crosses'],
                name='Total Crosses',
                marker_color=self.colors['sma']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_interval_distributions(self, distribution_analysis: Dict,
                                  title: str = "Interval Distribution Analysis") -> go.Figure:
        """
        Plot interval distribution analysis.
        
        Args:
            distribution_analysis: Dictionary with distribution analysis results
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        periods = list(distribution_analysis.keys())
        periods = [p for p in periods if 'error' not in distribution_analysis[p]]
        
        if not periods:
            return go.Figure().add_annotation(
                text="No valid distribution data to plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplots for each period
        fig = make_subplots(
            rows=len(periods), cols=2,
            subplot_titles=[f"Period {p} - Bullish" for p in periods] + 
                          [f"Period {p} - Bearish" for p in periods],
            specs=[[{"type": "histogram"}, {"type": "histogram"}] for _ in periods]
        )
        
        for i, period in enumerate(periods):
            try:
                # Bullish distribution
                if 'distributions' in distribution_analysis[period]:
                    bullish_intervals = distribution_analysis[period]['timing_analysis']['bullish_intervals']
                    if bullish_intervals:
                        fig.add_trace(
                            go.Histogram(
                                x=bullish_intervals,
                                name=f'Bullish {period}',
                                marker_color=self.colors['bullish'],
                                opacity=0.7
                            ),
                            row=i+1, col=1
                        )
                
                # Bearish distribution
                if 'distributions' in distribution_analysis[period]:
                    bearish_intervals = distribution_analysis[period]['timing_analysis']['bearish_intervals']
                    if bearish_intervals:
                        fig.add_trace(
                            go.Histogram(
                                x=bearish_intervals,
                                name=f'Bearish {period}',
                                marker_color=self.colors['bearish'],
                                opacity=0.7
                            ),
                            row=i+1, col=2
                        )
                        
            except Exception as e:
                logger.error(f"Error plotting distribution for period {period}: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title=title,
            height=300 * len(periods),
            showlegend=True
        )
        
        return fig
    
    def plot_optimal_periods_analysis(self, optimal_analysis: pd.DataFrame,
                                    title: str = "Optimal SMA Periods Analysis") -> go.Figure:
        """
        Plot optimal SMA periods analysis.
        
        Args:
            optimal_analysis: DataFrame with optimal periods analysis
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Touch Efficiency by Period", "Consistency Score",
                           "Touch Frequency vs Period", "Optimal Period Ranking"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Touch efficiency
        fig.add_trace(
            go.Scatter(
                x=optimal_analysis['sma_period'],
                y=optimal_analysis['touch_efficiency'],
                mode='lines+markers',
                name='Touch Efficiency',
                line=dict(color=self.colors['price'], width=3)
            ),
            row=1, col=1
        )
        
        # Consistency score
        fig.add_trace(
            go.Scatter(
                x=optimal_analysis['sma_period'],
                y=optimal_analysis['consistency_score'],
                mode='lines+markers',
                name='Consistency Score',
                line=dict(color=self.colors['sma'], width=3)
            ),
            row=1, col=2
        )
        
        # Touch frequency vs period
        fig.add_trace(
            go.Scatter(
                x=optimal_analysis['sma_period'],
                y=optimal_analysis['touch_frequency'],
                mode='lines+markers',
                name='Touch Frequency',
                line=dict(color=self.colors['neutral'], width=3)
            ),
            row=2, col=1
        )
        
        # Ranking (top 10)
        top_10 = optimal_analysis.head(10)
        fig.add_trace(
            go.Bar(
                x=top_10['sma_period'],
                y=top_10['touch_efficiency'],
                name='Top 10 Efficiency',
                marker_color=self.colors['volume']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_dashboard(self, data: pd.DataFrame, sma_periods: List[int],
                        touch_analysis: pd.DataFrame,
                        timing_analysis: pd.DataFrame,
                        title: str = "Stock Analysis Dashboard") -> go.Figure:
        """
        Create a comprehensive dashboard with all analysis components.
        
        Args:
            data: Stock price data
            sma_periods: List of SMA periods
            touch_analysis: SMA touch analysis results
            timing_analysis: Crossover timing analysis results
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Create main price chart
        price_fig = self.plot_stock_with_smas(data, sma_periods, "Stock Price with SMAs")
        
        # Create analysis charts
        touch_fig = self.plot_sma_touch_analysis(touch_analysis, "SMA Touch Analysis")
        timing_fig = self.plot_crossover_timing_analysis(timing_analysis, "Crossover Analysis")
        
        # Combine into dashboard
        dashboard = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Stock Price with SMAs", "SMA Touch Analysis", "Crossover Analysis"),
            vertical_spacing=0.05,
            specs=[[{"type": "candlestick"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]],
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Add price chart traces
        for trace in price_fig.data:
            dashboard.add_trace(trace, row=1, col=1)
        
        # Add touch analysis traces
        for trace in touch_fig.data:
            dashboard.add_trace(trace, row=2, col=1)
        
        # Add timing analysis traces
        for trace in timing_fig.data:
            dashboard.add_trace(trace, row=3, col=1)
        
        # Update layout
        dashboard.update_layout(
            title=title,
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )
        
        return dashboard
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save a plot to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format ('html', 'png', 'pdf')
        """
        try:
            if format == 'html':
                fig.write_html(filename)
            elif format == 'png':
                fig.write_image(filename)
            elif format == 'pdf':
                fig.write_image(filename)
            else:
                logger.warning(f"Unsupported format: {format}. Saving as HTML.")
                fig.write_html(filename)
            
            logger.info(f"Plot saved as {filename}")
        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}") 