"""
Interactive visualization dashboard for order book analytics
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..core.orderbook import OrderBook
from ..analytics.spread_analyzer import SpreadAnalyzer
from ..analytics.depth_analyzer import DepthAnalyzer
from ..analytics.liquidity_analyzer import LiquidityAnalyzer


class OrderBookVisualizer:
    """Create interactive visualizations for order book analytics"""
    
    def __init__(self, theme: str = 'plotly_white'):
        self.theme = theme
        self.color_bid = '#00BFFF'  # Deep sky blue
        self.color_ask = '#FF6B6B'  # Coral red
        self.color_mid = '#2E8B57'  # Sea green
        self.color_spread = '#FFA500'  # Orange
        
    def plot_order_book_depth(self, orderbook: OrderBook, 
                            levels: int = 20,
                            title: str = "Order Book Depth") -> go.Figure:
        """Create interactive depth chart"""
        depth = orderbook.get_depth(levels=levels)
        
        if not depth['bids'] or not depth['asks']:
            return self._create_empty_figure("No data available")
        
        # Extract data
        bid_prices = [level['price'] for level in depth['bids']]
        bid_volumes = [level['quantity'] for level in depth['bids']]
        ask_prices = [level['price'] for level in depth['asks']]
        ask_volumes = [level['quantity'] for level in depth['asks']]
        
        # Create figure
        fig = go.Figure()
        
        # Add bid bars
        fig.add_trace(go.Bar(
            x=bid_volumes,
            y=bid_prices,
            name='Bids',
            orientation='h',
            marker_color=self.color_bid,
            opacity=0.7
        ))
        
        # Add ask bars
        fig.add_trace(go.Bar(
            x=ask_volumes,
            y=ask_prices,
            name='Asks',
            orientation='h',
            marker_color=self.color_ask,
            opacity=0.7
        ))
        
        # Add mid price line
        if orderbook.mid_price:
            fig.add_hline(
                y=orderbook.mid_price,
                line_dash="dash",
                line_color=self.color_mid,
                annotation_text=f"Mid: ${orderbook.mid_price:,.2f}",
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Volume",
            yaxis_title="Price",
            barmode='relative',
            height=600,
            template=self.theme,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_analytics_dashboard(self, orderbook: OrderBook,
                               spread_analyzer: SpreadAnalyzer,
                               depth_analyzer: DepthAnalyzer,
                               liquidity_analyzer: LiquidityAnalyzer) -> go.Figure:
        """Create comprehensive analytics dashboard"""
        # Calculate metrics
        spread_metrics = spread_analyzer.calculate_all_metrics()
        depth_metrics = depth_analyzer.calculate_all_metrics()
        liquidity_metrics = liquidity_analyzer.calculate_all_metrics()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Order Book Depth',
                'Spread Analysis',
                'Liquidity Metrics',
                'Depth Profile',
                'Volume Imbalance',
                'Market Impact',
                'Price Levels',
                'Resilience Score',
                'Summary'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'table'}, {'type': 'indicator'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Order Book Depth (Top-left)
        depth = orderbook.get_depth(levels=10)
        if depth['bids'] and depth['asks']:
            bid_prices = [level['price'] for level in depth['bids']]
            bid_volumes = [level['quantity'] for level in depth['bids']]
            ask_prices = [level['price'] for level in depth['asks']]
            ask_volumes = [level['quantity'] for level in depth['asks']]
            
            fig.add_trace(
                go.Bar(x=bid_volumes, y=bid_prices, name='Bids',
                      orientation='h', marker_color=self.color_bid),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=ask_volumes, y=ask_prices, name='Asks',
                      orientation='h', marker_color=self.color_ask),
                row=1, col=1
            )
        
        # 2. Spread Indicator (Top-middle)
        spread_bps = spread_metrics.spread_bps
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=spread_bps,
                title={'text': "Spread (bps)"},
                gauge={
                    'axis': {'range': [0, max(50, spread_bps * 1.5)]},
                    'bar': {'color': self.color_spread},
                    'steps': [
                        {'range': [0, 10], 'color': "green"},
                        {'range': [10, 25], 'color': "yellow"},
                        {'range': [25, 50], 'color': "red"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # 3. Liquidity Score Indicator (Top-right)
        liquidity_score = liquidity_metrics.overall_liquidity_score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=liquidity_score,
                title={'text': "Liquidity Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        # 4. Depth Profile (Middle-left)
        if depth_metrics.cumulative_depth_bps:
            offsets, volumes = zip(*depth_metrics.cumulative_depth_bps)
            fig.add_trace(
                go.Scatter(x=offsets, y=volumes, mode='lines',
                          name='Cumulative Depth', line=dict(color='purple')),
                row=2, col=1
            )
        
        # 5. Volume Imbalance (Middle-middle)
        imbalance_levels = [5, 10, 20]
        imbalances = depth_analyzer.calculate_volume_imbalance_at_levels(imbalance_levels)
        
        fig.add_trace(
            go.Bar(x=list(imbalances.keys()), y=list(imbalances.values()),
                  name='Volume Imbalance', marker_color='orange'),
            row=2, col=2
        )
        
        # 6. Market Impact Curve (Middle-right)
        if liquidity_metrics.market_impact_models:
            quantities = np.linspace(1, 1000, 50)
            for model_name, model_func in liquidity_metrics.market_impact_models.items():
                # Simplified impact calculation
                impacts = [model_func * np.sqrt(q) for q in quantities]
                fig.add_trace(
                    go.Scatter(x=quantities, y=impacts, mode='lines',
                              name=f'{model_name} model'),
                    row=2, col=3
                )
        
        # 7. Price Levels Table (Bottom-left)
        price_levels_data = []
        for i, (bid, ask) in enumerate(zip(depth['bids'][:5], depth['asks'][:5])):
            price_levels_data.append([
                f"Level {i+1}",
                f"${bid['price']:,.2f}",
                f"{bid['quantity']:,.2f}",
                f"${ask['price']:,.2f}",
                f"{ask['quantity']:,.2f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Level', 'Bid Price', 'Bid Qty', 'Ask Price', 'Ask Qty'],
                    fill_color='gray',
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*price_levels_data)) if price_levels_data else [[]]*5,
                    fill_color='white',
                    align='center'
                )
            ),
            row=3, col=1
        )
        
        # 8. Resilience Indicator (Bottom-middle)
        resilience = depth_metrics.depth_resilience
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=resilience,
                title={'text': "Resilience"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"}
                }
            ),
            row=3, col=2
        )
        
        # 9. Summary Metrics Table (Bottom-right)
        summary_data = [
            ['Metric', 'Value'],
            ['Best Bid', f"${orderbook.best_bid or 0:,.2f}"],
            ['Best Ask', f"${orderbook.best_ask or 0:,.2f}"],
            ['Spread', f"{spread_metrics.absolute_spread:,.4f}"],
            ['Spread (bps)', f"{spread_bps:,.1f}"],
            ['Mid Price', f"${orderbook.mid_price or 0:,.2f}"],
            ['Bid Volume', f"{depth_metrics.total_bid_volume:,.2f}"],
            ['Ask Volume', f"{depth_metrics.total_ask_volume:,.2f}"],
            ['Imbalance', f"{depth_metrics.volume_imbalance:+.3f}"],
            ['Kyle Lambda', f"{liquidity_metrics.kyle_lambda:.6f}"],
            ['Amihud Ratio', f"{liquidity_metrics.amihud_ratio:.6f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=summary_data[0],
                    fill_color='darkblue',
                    font=dict(color='white'),
                    align='center'
                ),
                cells=dict(
                    values=[row[0] for row in summary_data[1:]] + [row[1] for row in summary_data[1:]],
                    fill_color=[['white', 'lightgray'] * len(summary_data)],
                    align='center'
                )
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Order Book Analytics Dashboard",
            showlegend=True,
            template=self.theme
        )
        
        return fig
    
    def create_summary_table(self, orderbook: OrderBook,
                        spread_analyzer: SpreadAnalyzer,
                        depth_analyzer: DepthAnalyzer,
                        liquidity_analyzer: LiquidityAnalyzer) -> str:
        """Create formatted summary table using ASCII characters"""
        spread_metrics = spread_analyzer.calculate_all_metrics()
        depth_metrics = depth_analyzer.calculate_all_metrics()
        liquidity_metrics = liquidity_analyzer.calculate_all_metrics()
        
        summary = f"""
    {'='*80}
    ORDER BOOK ANALYTICS SUMMARY
    {'='*80}
    Symbol: {orderbook.symbol}
    Timestamp: {orderbook.last_update_time or 'N/A'}
    {'-'*80}
    PRICE INFORMATION
    {'-'*80}
    Best Bid:     ${orderbook.best_bid or 0:15,.2f}
    Best Ask:     ${orderbook.best_ask or 0:15,.2f}
    Mid Price:    ${orderbook.mid_price or 0:15,.2f}
    Spread:       ${spread_metrics.absolute_spread:15,.4f}
    Spread (bps): {spread_metrics.spread_bps:15,.1f}
    {'-'*80}
    DEPTH METRICS
    {'-'*80}
    Bid Volume:   {depth_metrics.total_bid_volume:15,.2f}
    Ask Volume:   {depth_metrics.total_ask_volume:15,.2f}
    Imbalance:    {depth_metrics.volume_imbalance:15,.3f}
    Liquidity Score: {depth_metrics.liquidity_score:12,.1f}/100
    Resilience:   {depth_metrics.depth_resilience:15,.1f}/100
    {'-'*80}
    LIQUIDITY METRICS
    {'-'*80}
    Kyle's Lambda: {liquidity_metrics.kyle_lambda:14,.6f}
    Amihud Ratio:  {liquidity_metrics.amihud_ratio:14,.6f}
    LCR:           {liquidity_metrics.liquidity_coverage_ratio:15,.3f}
    Price Efficiency: {liquidity_metrics.price_efficiency:12,.3f}
    Overall Score: {liquidity_metrics.overall_liquidity_score:12,.1f}/100
    {'-'*80}
    ALERTS & WARNINGS
    {'-'*80}
    """
        
        # Add alerts
        alerts = []
        if spread_metrics.is_wide_spread:
            alerts.append("⚠️  Wide spread detected")
        if abs(depth_metrics.volume_imbalance) > 0.7:
            alerts.append("⚠️  Severe volume imbalance")
        if depth_metrics.liquidity_score < 30:
            alerts.append("⚠️  Low liquidity score")
        if liquidity_metrics.overall_liquidity_score < 40:
            alerts.append("⚠️  Poor overall liquidity")
        
        if alerts:
            for alert in alerts:
                summary += f"{alert}\n"
        else:
            summary += "No alerts - Market conditions normal\n"
        
        summary += f"{'='*80}"
        
        return summary
    
    def plot_market_impact_curve(self, liquidity_analyzer: LiquidityAnalyzer,
                               max_quantity: float = 10000) -> go.Figure:
        """Plot market impact curves for different models"""
        fig = go.Figure()
        
        # Generate quantity range
        quantities = np.linspace(1, max_quantity, 100)
        
        # Get impact models
        metrics = liquidity_analyzer.calculate_all_metrics()
        
        if metrics.market_impact_models:
            for model_name in metrics.market_impact_models.keys():
                # Calculate impacts (simplified)
                impacts = []
                for q in quantities:
                    # Simple sqrt model for demonstration
                    impact = metrics.market_impact_models.get(model_name, 0) * np.sqrt(q)
                    impacts.append(impact)
                
                fig.add_trace(go.Scatter(
                    x=quantities,
                    y=impacts,
                    mode='lines',
                    name=f'{model_name} Impact',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Market Impact Curves',
            xaxis_title='Order Quantity',
            yaxis_title='Price Impact',
            template=self.theme,
            height=500
        )
        
        return fig
    
    @staticmethod
    def _create_empty_figure(message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig