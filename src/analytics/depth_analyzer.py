"""
Order book depth analysis
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from ..core.orderbook import OrderBook


@dataclass
class DepthMetrics:
    """Container for depth metrics"""
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    volume_imbalance: float = 0.0  # -1 to 1
    cumulative_depth_bps: List[Tuple[float, float]] = None  # (price_offset_bps, cumulative_volume)
    vwap_at_depth: Dict[float, float] = None  # {depth_percentage: vwap}
    liquidity_score: float = 0.0  # 0-100
    order_book_slope: float = 0.0
    depth_resilience: float = 0.0  # 0-100


class DepthAnalyzer:
    """Analyze order book depth and liquidity"""
    
    def __init__(self, orderbook: OrderBook, depth_levels: int = 20):
        self.orderbook = orderbook
        self.depth_levels = depth_levels
    
    def calculate_all_metrics(self) -> DepthMetrics:
        """Calculate comprehensive depth metrics"""
        metrics = DepthMetrics()
        
        # Get current depth
        depth = self.orderbook.get_depth(levels=self.depth_levels)
        
        if not depth['bids'] or not depth['asks']:
            return metrics
        
        metrics.total_bid_volume = depth['total_bid_volume']
        metrics.total_ask_volume = depth['total_ask_volume']
        
        # Volume imbalance
        total_volume = metrics.total_bid_volume + metrics.total_ask_volume
        if total_volume > 0:
            metrics.volume_imbalance = (
                (metrics.total_bid_volume - metrics.total_ask_volume) / total_volume
            )
        
        # Cumulative depth in bps
        metrics.cumulative_depth_bps = self._calculate_cumulative_depth(depth)
        
        # VWAP at various depth percentages
        metrics.vwap_at_depth = self._calculate_vwap_at_depth(depth)
        
        # Order book slope (price sensitivity to volume)
        metrics.order_book_slope = self._calculate_book_slope(depth)
        
        # Liquidity score (0-100)
        metrics.liquidity_score = self._calculate_liquidity_score(metrics)
        
        # Depth resilience
        metrics.depth_resilience = self._calculate_resilience_score(depth)
        
        return metrics
    
    def _calculate_cumulative_depth(self, depth: Dict) -> List[Tuple[float, float]]:
        """Calculate cumulative volume at price offsets in bps"""
        if not self.orderbook.mid_price or self.orderbook.mid_price == 0:
            return []
        
        mid_price = self.orderbook.mid_price
        cumulative_depth = []
        
        # Calculate for bids (negative offsets)
        cumulative_volume = 0
        for level in depth['bids']:
            price_offset_bps = ((level['price'] - mid_price) / mid_price) * 10000
            cumulative_volume += level['quantity']
            cumulative_depth.append((price_offset_bps, cumulative_volume))
        
        # Calculate for asks (positive offsets)
        cumulative_volume = 0
        for level in depth['asks']:
            price_offset_bps = ((level['price'] - mid_price) / mid_price) * 10000
            cumulative_volume += level['quantity']
            cumulative_depth.append((price_offset_bps, cumulative_volume))
        
        # Sort by price offset
        cumulative_depth.sort(key=lambda x: x[0])
        
        return cumulative_depth
    
    def _calculate_vwap_at_depth(self, depth: Dict) -> Dict[float, float]:
        """Calculate VWAP for various depth percentages"""
        vwap_at_depth = {}
        
        # Define depth percentages to analyze
        depth_percentages = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]  # as percentages of total volume
        
        for percentage in depth_percentages:
            # Calculate target volume
            total_volume = depth['total_bid_volume'] + depth['total_ask_volume']
            target_volume = total_volume * (percentage / 100)
            
            # Calculate VWAP for this depth
            vwap = self._calculate_partial_vwap(depth, target_volume)
            vwap_at_depth[percentage] = vwap
        
        return vwap_at_depth
    
    def _calculate_partial_vwap(self, depth: Dict, target_volume: float) -> float:
        """Calculate VWAP for a target volume"""
        total_cost = 0.0
        filled_volume = 0.0
        
        # Fill from bids (for sell orders)
        for level in depth['bids']:
            available = level['quantity']
            if filled_volume + available <= target_volume / 2:  # Split between bids and asks
                total_cost += level['price'] * available
                filled_volume += available
            else:
                remaining = target_volume / 2 - filled_volume
                total_cost += level['price'] * remaining
                filled_volume += remaining
                break
        
        # Fill from asks (for buy orders)
        for level in depth['asks']:
            available = level['quantity']
            if filled_volume + available <= target_volume:
                total_cost += level['price'] * available
                filled_volume += available
            else:
                remaining = target_volume - filled_volume
                total_cost += level['price'] * remaining
                filled_volume += remaining
                break
        
        if filled_volume == 0:
            return 0.0
        
        return total_cost / filled_volume
    
    def _calculate_book_slope(self, depth: Dict) -> float:
        """Calculate order book slope (ΔPrice/ΔVolume)"""
        if not depth['bids'] or not depth['asks']:
            return 0.0
        
        # Calculate average price change per unit volume
        slopes = []
        
        # Bid side slope
        bid_prices = [level['price'] for level in depth['bids']]
        bid_volumes = [level['quantity'] for level in depth['bids']]
        
        if len(bid_prices) > 1:
            price_range = bid_prices[0] - bid_prices[-1]
            volume_sum = sum(bid_volumes)
            if volume_sum > 0:
                slopes.append(price_range / volume_sum)
        
        # Ask side slope
        ask_prices = [level['price'] for level in depth['asks']]
        ask_volumes = [level['quantity'] for level in depth['asks']]
        
        if len(ask_prices) > 1:
            price_range = ask_prices[-1] - ask_prices[0]
            volume_sum = sum(ask_volumes)
            if volume_sum > 0:
                slopes.append(price_range / volume_sum)
        
        return np.mean(slopes) if slopes else 0.0
    
    def _calculate_liquidity_score(self, metrics: DepthMetrics) -> float:
        """Calculate overall liquidity score (0-100)"""
        score = 50.0  # Base score
        
        # Adjust based on volume
        total_volume = metrics.total_bid_volume + metrics.total_ask_volume
        if total_volume > 1000:  # Threshold for "good" volume
            score += 20
        elif total_volume < 100:
            score -= 20
        
        # Adjust based on imbalance
        imbalance_abs = abs(metrics.volume_imbalance)
        if imbalance_abs > 0.7:  # Severe imbalance
            score -= 30
        elif imbalance_abs < 0.2:  # Balanced
            score += 10
        
        # Adjust based on order book slope (lower is better)
        if metrics.order_book_slope > 0:
            if metrics.order_book_slope < 0.001:  # Very liquid
                score += 20
            elif metrics.order_book_slope > 0.01:  # Illiquid
                score -= 20
        
        # Clamp between 0 and 100
        return max(0, min(100, score))
    
    def _calculate_resilience_score(self, depth: Dict) -> float:
        """Calculate depth resilience score (0-100)"""
        if len(depth['bids']) < 3 or len(depth['asks']) < 3:
            return 0.0
        
        # Check depth distribution
        bid_volumes = [level['quantity'] for level in depth['bids']]
        ask_volumes = [level['quantity'] for level in depth['asks']]
        
        # Calculate concentration (Herfindahl index)
        def herfindahl_index(volumes):
            total = sum(volumes)
            if total == 0:
                return 1.0  # Worst case
            return sum((v / total) ** 2 for v in volumes)
        
        bid_concentration = herfindahl_index(bid_volumes)
        ask_concentration = herfindahl_index(ask_volumes)
        
        # Lower concentration (closer to uniform) is better
        resilience = 100 * (1 - (bid_concentration + ask_concentration) / 2)
        
        return max(0, min(100, resilience))
    
    def calculate_volume_imbalance_at_levels(self, levels: List[int]) -> Dict[int, float]:
        """Calculate volume imbalance at specific levels"""
        imbalances = {}
        
        for level_count in levels:
            depth = self.orderbook.get_depth(levels=level_count)
            
            total_bid = depth['total_bid_volume']
            total_ask = depth['total_ask_volume']
            total = total_bid + total_ask
            
            if total > 0:
                imbalance = (total_bid - total_ask) / total
            else:
                imbalance = 0.0
            
            imbalances[level_count] = imbalance
        
        return imbalances