#!/usr/bin/env python3
"""
Comprehensive demonstration of OrderBook Analytics Engine
"""
import time
import random
import numpy as np
from src.core.orderbook import OrderBook
from src.analytics.spread_analyzer import SpreadAnalyzer
from src.analytics.depth_analyzer import DepthAnalyzer
from src.analytics.liquidity_analyzer import LiquidityAnalyzer
from src.visualization.dashboard import OrderBookVisualizer


def generate_market_data(base_price: float = 50000.0) -> tuple:
    """Generate simulated market data"""
    # Generate bids (descending from base_price)
    bids = []
    current_price = base_price * 0.995  # Start 0.5% below
    for i in range(20):
        price = current_price - i * 0.5
        quantity = random.uniform(0.5, 5.0)
        bids.append((price, quantity))
    
    # Generate asks (ascending from base_price)
    asks = []
    current_price = base_price * 1.005  # Start 0.5% above
    for i in range(20):
        price = current_price + i * 0.5
        quantity = random.uniform(0.5, 5.0)
        asks.append((price, quantity))
    
    return bids, asks


def run_basic_demo():
    """Run basic order book demo"""
    print("=" * 60)
    print("ORDER BOOK ANALYTICS ENGINE - DEMONSTRATION")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Order Book...")
    orderbook = OrderBook(symbol="BTC/USDT")
    
    # Generate and update market data
    print("2. Updating with market data...")
    bids, asks = generate_market_data()
    orderbook.update_bids(bids)
    orderbook.update_asks(asks)
    
    # Display basic info
    print(f"\n3. Order Book Status:")
    print(f"   Symbol: {orderbook.symbol}")
    print(f"   Best Bid: ${orderbook.best_bid:,.2f}")
    print(f"   Best Ask: ${orderbook.best_ask:,.2f}")
    print(f"   Mid Price: ${orderbook.mid_price:,.2f}")
    print(f"   Spread: ${orderbook.get_spread():,.2f}")
    
    # Analyze spreads
    print("\n4. Spread Analysis...")
    spread_analyzer = SpreadAnalyzer(orderbook)
    spread_metrics = spread_analyzer.calculate_all_metrics()
    
    print(f"   Absolute Spread: ${spread_metrics.absolute_spread:,.4f}")
    print(f"   Relative Spread: {spread_metrics.relative_spread:.4f}%")
    print(f"   Spread (bps): {spread_metrics.spread_bps:,.1f}")
    print(f"   Price Impact: ${spread_metrics.price_impact:,.4f}")
    print(f"   Wide Spread Alert: {'YES' if spread_metrics.is_wide_spread else 'NO'}")
    
    # Analyze depth
    print("\n5. Depth Analysis...")
    depth_analyzer = DepthAnalyzer(orderbook)
    depth_metrics = depth_analyzer.calculate_all_metrics()
    
    print(f"   Bid Volume: {depth_metrics.total_bid_volume:,.2f}")
    print(f"   Ask Volume: {depth_metrics.total_ask_volume:,.2f}")
    print(f"   Volume Imbalance: {depth_metrics.volume_imbalance:+.3f}")
    print(f"   Liquidity Score: {depth_metrics.liquidity_score:.1f}/100")
    print(f"   Order Book Slope: {depth_metrics.order_book_slope:.6f}")
    
    # Analyze liquidity
    print("\n6. Advanced Liquidity Analysis...")
    
    # Generate some price history for liquidity analysis
    price_history = [50000 + random.uniform(-100, 100) for _ in range(50)]
    volume_history = [random.uniform(10, 100) for _ in range(50)]
    
    liquidity_analyzer = LiquidityAnalyzer(orderbook, price_history, volume_history)
    liquidity_metrics = liquidity_analyzer.calculate_all_metrics()
    
    print(f"   Kyle's Lambda: {liquidity_metrics.kyle_lambda:.6f}")
    print(f"   Amihud Ratio: {liquidity_metrics.amihud_ratio:.6f}")
    print(f"   LCR: {liquidity_metrics.liquidity_coverage_ratio:.3f}")
    print(f"   Resilience Score: {liquidity_metrics.resilience_score:.1f}/100")
    print(f"   Price Efficiency: {liquidity_metrics.price_efficiency:.3f}")
    print(f"   Overall Liquidity Score: {liquidity_metrics.overall_liquidity_score:.1f}/100")
    
    # Estimate execution quality
    print("\n7. Execution Quality Estimation...")
    
    for quantity in [1.0, 10.0, 100.0]:
        buy_slippage = orderbook.estimate_slippage(quantity, 'buy')
        sell_slippage = orderbook.estimate_slippage(quantity, 'sell')
        
        print(f"\n   Quantity: {quantity} BTC")
        print(f"   Buy VWAP: ${buy_slippage['vwap']:,.2f}")
        print(f"   Buy Slippage: {buy_slippage['slippage_bps']:.1f} bps")
        print(f"   Sell VWAP: ${sell_slippage['vwap']:,.2f}")
        print(f"   Sell Slippage: {sell_slippage['slippage_bps']:.1f} bps")
    
    # Create visualizations
    print("\n8. Creating Visualizations...")
    visualizer = OrderBookVisualizer()
    
    # Save depth chart
    fig_depth = visualizer.plot_order_book_depth(orderbook, levels=15)
    fig_depth.write_html("output/depth_chart.html")
    print("   ✓ Depth chart saved to output/depth_chart.html")
    
    # Create dashboard
    fig_dashboard = visualizer.plot_analytics_dashboard(
        orderbook, spread_analyzer, depth_analyzer, liquidity_analyzer
    )
    fig_dashboard.write_html("output/analytics_dashboard.html")
    print("   ✓ Analytics dashboard saved to output/analytics_dashboard.html")
    
    # Generate summary table
    summary = visualizer.create_summary_table(
        orderbook, spread_analyzer, depth_analyzer, liquidity_analyzer
    )
    print("\n" + summary)
    
    # Save summary to file
    with open("output/summary.txt", "w") as f:
        f.write(summary)
    print("\n   ✓ Summary saved to output/summary.txt")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


def run_realtime_simulation():
    """Run real-time market simulation"""
    print("\n" + "=" * 60)
    print("REAL-TIME MARKET SIMULATION")
    print("=" * 60)
    
    # Initialize
    orderbook = OrderBook(symbol="ETH/USDT")
    spread_analyzer = SpreadAnalyzer(orderbook)
    visualizer = OrderBookVisualizer()
    
    base_price = 2500.0
    
    print("\nSimulating 10 seconds of market data...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for i in range(10):
            # Update price with random walk
            base_price += random.uniform(-10, 10)
            base_price = max(1000, min(5000, base_price))
            
            # Generate new market data
            bids, asks = generate_market_data(base_price)
            orderbook.update_bids(bids)
            orderbook.update_asks(asks)
            
            # Update liquidity analyzer with simulated history
            price_history = [base_price + random.uniform(-50, 50) for _ in range(50)]
            volume_history = [random.uniform(50, 200) for _ in range(50)]
            
            # Calculate metrics
            spread_metrics = spread_analyzer.calculate_all_metrics()
            spread_stats = spread_analyzer.calculate_spread_statistics()
            spread_trend = spread_analyzer.get_spread_trend()
            
            # Display update
            print(f"\nUpdate {i+1}:")
            print(f"  Price: ${base_price:,.2f}")
            print(f"  Spread: {spread_metrics.spread_bps:,.1f} bps")
            print(f"  Trend: {spread_trend}")
            print(f"  Avg Spread: {spread_stats.get('mean', 0):.1f} bps")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    
    print("\nSimulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run demos
    run_basic_demo()
    run_realtime_simulation()
    
    print("\nTo view interactive charts:")
    print("1. Open 'output/depth_chart.html' in your browser")
    print("2. Open 'output/analytics_dashboard.html' for full dashboard")
    print("\nThank you for using OrderBook Analytics Engine!")