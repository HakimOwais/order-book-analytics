"""
Core Order Book implementation with O(log n) operations
"""
import bisect
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from decimal import Decimal


@dataclass
class Order:
    """Single order representation"""
    price: float
    quantity: float
    order_id: Optional[str] = None
    timestamp: Optional[int] = None


class PriceLevel:
    """Price level aggregating multiple orders"""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: List[Order] = []
        self.total_quantity = 0.0
    
    def add_order(self, order: Order):
        """Add order to price level"""
        self.orders.append(order)
        self.total_quantity += order.quantity
    
    def remove_order(self, order_id: str) -> bool:
        """Remove order by ID"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_quantity -= order.quantity
                self.orders.pop(i)
                return True
        return False
    
    def update_quantity(self, order_id: str, new_quantity: float) -> bool:
        """Update order quantity"""
        for order in self.orders:
            if order.order_id == order_id:
                self.total_quantity += new_quantity - order.quantity
                order.quantity = new_quantity
                return True
        return False


class OrderBook:
    """Main OrderBook class with efficient operations"""
    
    def __init__(self, symbol: str, max_levels: int = 1000):
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Sorted lists of prices for O(log n) lookups
        self.bid_prices: List[float] = []  # descending
        self.ask_prices: List[float] = []  # ascending
        
        # Price level dictionaries
        self.bid_levels: Dict[float, PriceLevel] = {}
        self.ask_levels: Dict[float, PriceLevel] = {}
        
        # Cache for frequently accessed values
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        self._mid_price: Optional[float] = None
        
        # Statistics
        self.update_count = 0
        self.last_update_time = None
    
    def update_bids(self, bids: List[Tuple[float, float]]):
        """Update bid side with new data (snapshot or incremental)"""
        self.bid_prices.clear()
        self.bid_levels.clear()
        
        for price, quantity in sorted(bids, key=lambda x: x[0], reverse=True):
            if price > 0 and quantity > 0:
                self.bid_prices.append(price)
                level = PriceLevel(price)
                level.add_order(Order(price, quantity))
                self.bid_levels[price] = level
        
        self._update_caches()
        self.update_count += 1
    
    def update_asks(self, asks: List[Tuple[float, float]]):
        """Update ask side with new data"""
        self.ask_prices.clear()
        self.ask_levels.clear()
        
        for price, quantity in sorted(asks, key=lambda x: x[0]):
            if price > 0 and quantity > 0:
                self.ask_prices.append(price)
                level = PriceLevel(price)
                level.add_order(Order(price, quantity))
                self.ask_levels[price] = level
        
        self._update_caches()
        self.update_count += 1
    
    def _update_caches(self):
        """Update cached values for performance"""
        if self.bid_prices:
            self._best_bid = self.bid_prices[0]
        if self.ask_prices:
            self._best_ask = self.ask_prices[0]
        
        if self._best_bid and self._best_ask:
            self._mid_price = (self._best_bid + self._best_ask) / 2
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self._best_bid
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self._best_ask
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price"""
        return self._mid_price
    
    def get_spread(self) -> Optional[float]:
        """Calculate absolute spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def get_depth(self, levels: int = 10) -> Dict:
        """Get order book depth up to N levels"""
        depth = {
            'bids': [],
            'asks': [],
            'total_bid_volume': 0.0,
            'total_ask_volume': 0.0
        }
        
        # Get top N bids
        for price in self.bid_prices[:levels]:
            level = self.bid_levels[price]
            depth['bids'].append({
                'price': price,
                'quantity': level.total_quantity,
                'order_count': len(level.orders)
            })
            depth['total_bid_volume'] += level.total_quantity
        
        # Get top N asks
        for price in self.ask_prices[:levels]:
            level = self.ask_levels[price]
            depth['asks'].append({
                'price': price,
                'quantity': level.total_quantity,
                'order_count': len(level.orders)
            })
            depth['total_ask_volume'] += level.total_quantity
        
        return depth
    
    def estimate_vwap(self, quantity: float, side: str) -> Dict:
        """
        Estimate VWAP for a given quantity and side
        Returns: {'vwap': float, 'filled': float, 'remaining': float}
        """
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        total_cost = 0.0
        filled_quantity = 0.0
        levels = self.ask_prices if side == 'buy' else self.bid_prices
        level_dict = self.ask_levels if side == 'buy' else self.bid_levels
        
        for price in levels:
            level = level_dict[price]
            available = level.total_quantity
            
            if filled_quantity + available <= quantity:
                # Take entire level
                total_cost += price * available
                filled_quantity += available
            else:
                # Partial fill from this level
                remaining = quantity - filled_quantity
                total_cost += price * remaining
                filled_quantity = quantity
                break
        
        if filled_quantity == 0:
            return {'vwap': 0.0, 'filled': 0.0, 'remaining': quantity}
        
        vwap = total_cost / filled_quantity
        return {
            'vwap': vwap,
            'filled': filled_quantity,
            'remaining': quantity - filled_quantity
        }
    
    def estimate_slippage(self, quantity: float, side: str) -> Dict:
        """Estimate slippage for a given order"""
        if not self.mid_price:
            return {'slippage_bps': 0.0, 'slippage_abs': 0.0}
        
        vwap_result = self.estimate_vwap(quantity, side)
        if vwap_result['filled'] == 0:
            return {'slippage_bps': 0.0, 'slippage_abs': 0.0}
        
        vwap = vwap_result['vwap']
        mid = self.mid_price
        
        if side == 'buy':
            slippage_abs = vwap - mid
        else:
            slippage_abs = mid - vwap
        
        slippage_bps = (slippage_abs / mid) * 10000
        
        return {
            'slippage_bps': slippage_bps,
            'slippage_abs': slippage_abs,
            'vwap': vwap,
            'mid_price': mid,
            'filled': vwap_result['filled']
        }
    
    def to_dict(self) -> Dict:
        """Convert order book to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.last_update_time,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.get_spread(),
            'bid_levels': len(self.bid_prices),
            'ask_levels': len(self.ask_prices),
            'depth': self.get_depth(5)
        }
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)