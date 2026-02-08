from src.core.orderbook import OrderBook
from src.analytics.spread_analyzer import SpreadAnalyzer
from src.visualization.dashboard import OrderBookVisualizer

# Initialize order book
ob = OrderBook(symbol="BTC/USDT")

# Update with market data
ob.update_bids([(50000, 1.5), (49900, 2.0)])
ob.update_asks([(50100, 1.2), (50200, 3.0)])

# Analyze spreads
analyzer = SpreadAnalyzer(ob)
metrics = analyzer.calculate_all_metrics()
print(f"Spread: {metrics['absolute_spread']}")