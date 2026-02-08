"""
Advanced liquidity analysis including market impact models
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy import stats
from ..core.orderbook import OrderBook


@dataclass
class LiquidityMetrics:
    """Container for advanced liquidity metrics"""
    kyle_lambda: float = 0.0  # Permanent impact coefficient
    amihud_ratio: float = 0.0  # Illiquidity measure
    liquidity_coverage_ratio: float = 0.0  # LCR
    resilience_score: float = 0.0  # 0-100
    price_efficiency: float = 0.0  # 0-1
    market_impact_models: Dict[str, float] = None  # Different impact models
    overall_liquidity_score: float = 0.0  # 0-100


class LiquidityAnalyzer:
    """Advanced liquidity analysis with market impact models"""
    
    def __init__(self, orderbook: OrderBook, 
                 price_history: Optional[List[float]] = None,
                 volume_history: Optional[List[float]] = None):
        self.orderbook = orderbook
        self.price_history = price_history or []
        self.volume_history = volume_history or []
        self.max_history = 1000
    
    def calculate_all_metrics(self) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics"""
        metrics = LiquidityMetrics()
        metrics.market_impact_models = {}
        
        # Kyle's lambda (permanent price impact)
        metrics.kyle_lambda = self.calculate_kyle_lambda()
        
        # Amihud illiquidity ratio
        metrics.amihud_ratio = self.calculate_amihud_ratio()
        
        # Liquidity Coverage Ratio
        metrics.liquidity_coverage_ratio = self.calculate_lcr()
        
        # Resilience score
        metrics.resilience_score = self.calculate_resilience_score()
        
        # Price efficiency
        metrics.price_efficiency = self.calculate_price_efficiency()
        
        # Market impact models
        metrics.market_impact_models = self.calculate_market_impact_models()
        
        # Overall composite score
        metrics.overall_liquidity_score = self.calculate_composite_score(metrics)
        
        return metrics
    
    def calculate_kyle_lambda(self, window: int = 20) -> float:
        """
        Calculate Kyle's lambda (permanent price impact coefficient)
        lambda = cov(ΔPrice, Volume) / var(Volume)
        """
        if len(self.price_history) < window or len(self.volume_history) < window:
            return 0.0
        
        # Use recent data
        prices = self.price_history[-window:]
        volumes = self.volume_history[-window:]
        
        if len(prices) < 2:
            return 0.0
        
        # Convert to numpy arrays
        prices_np = np.array(prices)
        volumes_np = np.array(volumes)
        
        # Calculate returns and volume changes
        returns = np.diff(prices_np) / prices_np[:-1]
        volume_changes = np.diff(volumes_np)
        
        if len(returns) != len(volume_changes) or len(returns) < 2:
            return 0.0
        
        # Calculate Kyle's lambda
        covariance = np.cov(returns, volume_changes)[0, 1]
        volume_variance = np.var(volume_changes)
        
        if volume_variance == 0:
            return 0.0
        
        kyle_lambda = covariance / volume_variance
        
        return float(kyle_lambda)
    
    def calculate_amihud_ratio(self, window: int = 20) -> float:
        """
        Calculate Amihud illiquidity ratio
        Ratio = average(|Return| / Volume)
        """
        if len(self.price_history) < window or len(self.volume_history) < window:
            return 0.0
        
        prices = self.price_history[-window:]
        volumes = self.volume_history[-window:]
        
        if len(prices) < 2:
            return 0.0
        
        # Convert to numpy arrays
        prices_np = np.array(prices)
        volumes_np = np.array(volumes)
        
        # Calculate absolute returns
        returns = np.abs(np.diff(prices_np) / prices_np[:-1])
        
        # Use volumes for the periods corresponding to returns
        volumes_for_returns = volumes_np[1:]  # Align with returns
        
        # Filter zero volumes
        valid_indices = volumes_for_returns > 0
        if not np.any(valid_indices):
            return 0.0
        
        returns = returns[valid_indices]
        volumes_for_returns = volumes_for_returns[valid_indices]
        
        # Calculate Amihud ratio
        ratios = returns / volumes_for_returns
        amihud_ratio = np.mean(ratios)
        
        return float(amihud_ratio)
    
    def calculate_lcr(self, stress_volume: float = 10000.0) -> float:
        """
        Calculate Liquidity Coverage Ratio
        LCR = Available liquidity / Stress volume need
        """
        depth = self.orderbook.get_depth(levels=20)
        
        total_liquidity = depth['total_bid_volume'] + depth['total_ask_volume']
        
        if stress_volume <= 0:
            return 0.0
        
        lcr = total_liquidity / stress_volume
        
        return float(lcr)
    
    def calculate_resilience_score(self) -> float:
        """Calculate market resilience score (0-100)"""
        # Combine multiple factors
        factors = []
        
        # 1. Depth concentration
        depth = self.orderbook.get_depth(levels=10)
        bid_volumes = [level['quantity'] for level in depth['bids']]
        ask_volumes = [level['quantity'] for level in depth['asks']]
        
        if bid_volumes and ask_volumes:
            # Check if liquidity is concentrated at top
            top_bid_ratio = bid_volumes[0] / sum(bid_volumes) if sum(bid_volumes) > 0 else 0
            top_ask_ratio = ask_volumes[0] / sum(ask_volumes) if sum(ask_volumes) > 0 else 0
            
            # Lower concentration at top is better
            concentration_score = 100 * (1 - (top_bid_ratio + top_ask_ratio) / 2)
            factors.append(concentration_score)
        
        # 2. Spread stability (if we have history)
        if len(self.price_history) > 10:
            recent_prices = self.price_history[-10:]
            price_volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
            volatility_score = 100 * (1 - min(price_volatility * 100, 1))
            factors.append(volatility_score)
        
        # 3. Volume profile
        if self.volume_history:
            recent_volumes = self.volume_history[-10:]
            recent_volumes_np = np.array(recent_volumes)
            if np.mean(recent_volumes_np) > 0:
                volume_stability = 1 - (np.std(recent_volumes_np) / np.mean(recent_volumes_np))
                volume_score = 100 * volume_stability
                factors.append(volume_score)
        
        if not factors:
            return 50.0
        
        return float(np.mean(factors))
    
    def calculate_price_efficiency(self) -> float:
        """
        Calculate price efficiency (0-1)
        How quickly prices reflect information
        """
        if len(self.price_history) < 20:
            return 0.5
        
        # Convert to numpy array
        prices_np = np.array(self.price_history[-20:])
        returns = np.diff(prices_np) / prices_np[:-1]
        
        if len(returns) < 2:
            return 0.5
        
        # Variance ratio test (simplified)
        # Efficient markets should have returns close to random walk
        var_1day = np.var(returns)
        
        # Calculate 2-day returns
        if len(returns) >= 4:
            returns_2day = (prices_np[2:] - prices_np[:-2]) / prices_np[:-2]
            var_2day = np.var(returns_2day) / 2  # Normalize to daily
            
            if var_1day > 0:
                variance_ratio = var_2day / var_1day
                # Efficiency is closeness to 1
                efficiency = 1 - abs(variance_ratio - 1)
                return float(max(0, min(1, efficiency)))
        
        return 0.5
    
    def calculate_market_impact_models(self, quantity: float = 100.0) -> Dict[str, float]:
        """Calculate different market impact models"""
        models = {}
        
        if not self.orderbook.mid_price or self.orderbook.mid_price == 0:
            return models
        
        mid_price = self.orderbook.mid_price
        
        # 1. Linear model: impact = a * quantity
        depth = self.orderbook.get_depth(levels=10)
        total_liquidity = depth['total_bid_volume'] + depth['total_ask_volume']
        
        if total_liquidity > 0:
            a = 1 / total_liquidity
            models['linear'] = a * quantity * mid_price
        
        # 2. Square root model: impact = b * sqrt(quantity)
        spread = self.orderbook.get_spread()
        if spread is not None:
            b = spread / (2 * np.sqrt(total_liquidity)) if total_liquidity > 0 else 0
            models['sqrt'] = b * np.sqrt(quantity) * mid_price
        
        # 3. Kyle's model: impact = lambda * quantity
        kyle_lambda = self.calculate_kyle_lambda()
        models['kyle'] = kyle_lambda * quantity * mid_price
        
        return models
    
    def calculate_composite_score(self, metrics: LiquidityMetrics) -> float:
        """Calculate overall liquidity score from components"""
        scores = []
        weights = []
        
        # Kyle's lambda (lower is better)
        if metrics.kyle_lambda > 0:
            # Normalize: 100 * exp(-lambda * scale)
            kyle_score = 100 * np.exp(-metrics.kyle_lambda * 1000)
            scores.append(kyle_score)
            weights.append(0.3)
        elif metrics.kyle_lambda == 0:
            # If lambda is 0, give neutral score
            scores.append(50.0)
            weights.append(0.3)
        
        # Amihud ratio (lower is better)
        if metrics.amihud_ratio > 0:
            amihud_score = 100 * np.exp(-metrics.amihud_ratio * 10000)
            scores.append(amihud_score)
            weights.append(0.2)
        elif metrics.amihud_ratio == 0:
            scores.append(50.0)
            weights.append(0.2)
        
        # LCR (higher is better)
        lcr_score = min(metrics.liquidity_coverage_ratio * 100, 100)
        scores.append(lcr_score)
        weights.append(0.2)
        
        # Resilience score (already 0-100)
        scores.append(metrics.resilience_score)
        weights.append(0.2)
        
        # Price efficiency (0-1 scale to 0-100)
        efficiency_score = metrics.price_efficiency * 100
        scores.append(efficiency_score)
        weights.append(0.1)
        
        # Weighted average
        if scores and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
                return float(max(0, min(100, weighted_score)))
        
        return 50.0
    
    def update_history(self, price: float, volume: float):
        """Update price and volume history"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Trim history if too long
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        if len(self.volume_history) > self.max_history:
            self.volume_history.pop(0)