"""
Spread analysis for market microstructure
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from ..core.orderbook import OrderBook


@dataclass
class SpreadMetrics:
    """Container for spread metrics"""
    absolute_spread: float = 0.0
    relative_spread: float = 0.0  # percentage
    spread_bps: float = 0.0  # basis points
    effective_spread: float = 0.0
    price_impact: float = 0.0
    spread_volatility: float = 0.0
    is_wide_spread: bool = False


class SpreadAnalyzer:
    """Analyze spread metrics and patterns"""
    
    def __init__(self, orderbook: OrderBook):
        self.orderbook = orderbook
        self.history: List[SpreadMetrics] = []
        self.max_history = 1000
    
    def calculate_all_metrics(self) -> SpreadMetrics:
        """Calculate comprehensive spread metrics"""
        metrics = SpreadMetrics()
        
        # Basic spread calculations
        spread = self.orderbook.get_spread()
        mid_price = self.orderbook.mid_price
        
        if spread is None or mid_price is None or mid_price == 0:
            return metrics
        
        metrics.absolute_spread = spread
        metrics.relative_spread = (spread / mid_price) * 100
        metrics.spread_bps = metrics.relative_spread * 100
        
        # Effective spread (mid-quote deviation)
        if self.orderbook.best_bid and self.orderbook.best_ask:
            effective_spread = 2 * (mid_price - (self.orderbook.best_bid + self.orderbook.best_ask) / 2)
            metrics.effective_spread = abs(effective_spread)
        
        # Price impact estimation
        metrics.price_impact = self.calculate_price_impact()
        
        # Statistical analysis
        if len(self.history) > 1:
            spreads = [m.absolute_spread for m in self.history[-100:]]
            metrics.spread_volatility = np.std(spreads) if spreads else 0
        
        # Alert for wide spreads
        metrics.is_wide_spread = self._detect_wide_spread(metrics)
        
        # Store in history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return metrics
    
    def calculate_price_impact(self, quantity: float = 1.0) -> float:
        """
        Estimate price impact for a given quantity
        Uses sqrt model: impact = lambda * sqrt(quantity)
        """
        if not self.orderbook.mid_price or self.orderbook.mid_price == 0:
            return 0.0
        
        # Simple model based on order book depth
        depth = self.orderbook.get_depth(levels=10)
        
        if not depth['bids'] or not depth['asks']:
            return 0.0
        
        # Calculate average liquidity
        bid_liquidity = sum(level['quantity'] for level in depth['bids'])
        ask_liquidity = sum(level['quantity'] for level in depth['asks'])
        avg_liquidity = (bid_liquidity + ask_liquidity) / 2
        
        if avg_liquidity == 0:
            return 0.0
        
        # Kyle's lambda approximation
        spread = self.orderbook.get_spread()
        if spread is None:
            return 0.0
        
        lambda_est = spread / (avg_liquidity * self.orderbook.mid_price)
        impact = lambda_est * np.sqrt(quantity) * self.orderbook.mid_price
        
        return impact
    
    def _detect_wide_spread(self, metrics: SpreadMetrics) -> bool:
        """Detect abnormally wide spreads"""
        if len(self.history) < 10:
            return False
        
        recent_spreads = [m.spread_bps for m in self.history[-10:]]
        avg_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)
        
        # Alert if spread is 3 sigma above mean
        if std_spread > 0 and metrics.spread_bps > avg_spread + 3 * std_spread:
            return True
        
        # Absolute threshold (e.g., > 50 bps)
        if metrics.spread_bps > 50:
            return True
        
        return False
    
    def calculate_spread_statistics(self, window: int = 100) -> Dict:
        """Calculate statistical measures of spread"""
        if len(self.history) < window:
            window = len(self.history)
        
        if window == 0:
            return {}
        
        spreads = [m.spread_bps for m in self.history[-window:]]
        
        return {
            'mean': float(np.mean(spreads)),
            'median': float(np.median(spreads)),
            'std': float(np.std(spreads)),
            'min': float(np.min(spreads)),
            'max': float(np.max(spreads)),
            'q1': float(np.percentile(spreads, 25)),
            'q3': float(np.percentile(spreads, 75)),
            'iqr': float(np.percentile(spreads, 75) - np.percentile(spreads, 25))
        }
    
    def get_spread_trend(self, periods: int = 20) -> str:
        """Determine spread trend direction"""
        if len(self.history) < periods:
            return "insufficient_data"
        
        spreads = [m.spread_bps for m in self.history[-periods:]]
        
        # Linear regression for trend
        x = np.arange(len(spreads))
        slope, _ = np.polyfit(x, spreads, 1)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"