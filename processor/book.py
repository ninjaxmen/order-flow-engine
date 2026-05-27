from typing import List, Dict, Tuple
from utils.logger import logger

class OrderBook:
    """
    In-memory representation and reconstructor of the Limit Order Book (LOB).
    Maintains sorted price-level depth, best bid/ask bounds, and depth skewness metrics.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Bid levels: price -> size
        self.bids: Dict[float, float] = {}
        # Ask levels: price -> size
        self.asks: Dict[float, float] = {}
        self.last_update_ts = 0
        
        # Histograms for depth tracking (spoofing analysis)
        self.prev_total_bid_depth = 0.0
        self.prev_total_ask_depth = 0.0

    def update_depth(self, bids_list: List[List[float]], asks_list: List[List[float]]) -> None:
        """
        Updates LOB state with a new depth snapshot or L2 delta.
        Prices with size 0 are deleted (standard L2 mechanism).
        """
        # Save previous depth values for delta analysis
        self.prev_total_bid_depth = sum(self.bids.values())
        self.prev_total_ask_depth = sum(self.asks.values())

        # Update bids
        for price, size in bids_list:
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        # Update asks
        for price, size in asks_list:
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

    def get_best_bid(self) -> Tuple[float, float]:
        """Returns (best_bid_price, size) or (0.0, 0.0) if empty."""
        if not self.bids:
            return 0.0, 0.0
        best_price = max(self.bids.keys())
        return best_price, self.bids[best_price]

    def get_best_ask(self) -> Tuple[float, float]:
        """Returns (best_ask_price, size) or (0.0, 0.0) if empty."""
        if not self.asks:
            return 0.0, 0.0
        best_price = min(self.asks.keys())
        return best_price, self.asks[best_price]

    def get_sorted_bids(self, depth: int = 10) -> List[Tuple[float, float]]:
        """Returns bids sorted descending (highest bid first)."""
        sorted_keys = sorted(self.bids.keys(), reverse=True)
        return [(price, self.bids[price]) for price in sorted_keys[:depth]]

    def get_sorted_asks(self, depth: int = 10) -> List[Tuple[float, float]]:
        """Returns asks sorted ascending (lowest ask first)."""
        sorted_keys = sorted(self.asks.keys())
        return [(price, self.asks[price]) for price in sorted_keys[:depth]]

    def calculate_skew(self, depth_levels: int = 5) -> float:
        """
        Calculates the bid-ask depth volume skewness.
        Skew > 0 indicates bid-side dominates (bullish support).
        Skew < 0 indicates ask-side dominates (bearish resistance).
        Formula: (BidVolume - AskVolume) / (BidVolume + AskVolume)
        """
        sorted_bids = self.get_sorted_bids(depth_levels)
        sorted_asks = self.get_sorted_asks(depth_levels)
        
        bid_vol = sum(size for _, size in sorted_bids)
        ask_vol = sum(size for _, size in sorted_asks)
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / total_vol

    def detect_depth_spoofing(self, depth_levels: int = 10) -> Tuple[bool, str]:
        """
        Analyzes quick volume changes in the deep order book.
        Spoofing is characterized by huge quantities added and removed in seconds
        without actual execution (trades) occurring.
        """
        current_bid_depth = sum(size for _, size in self.get_sorted_bids(depth_levels))
        current_ask_depth = sum(size for _, size in self.get_sorted_asks(depth_levels))
        
        bid_diff = current_bid_depth - self.prev_total_bid_depth
        ask_diff = current_ask_depth - self.prev_total_ask_depth
        
        # Flag if depth shifts by > 50% in a single update frame without executions
        threshold_pct = 0.50
        
        if self.prev_total_bid_depth > 0 and abs(bid_diff) / self.prev_total_bid_depth > threshold_pct:
            if bid_diff < 0:
                return True, f"Bid Spoofing: Large liquidity withdrawal ({abs(bid_diff):.2f} contracts)"
            else:
                return True, f"Bid Bloating: Large liquidity injection ({bid_diff:.2f} contracts)"
                
        if self.prev_total_ask_depth > 0 and abs(ask_diff) / self.prev_total_ask_depth > threshold_pct:
            if ask_diff < 0:
                return True, f"Ask Spoofing: Large liquidity withdrawal ({abs(ask_diff):.2f} contracts)"
            else:
                return True, f"Ask Bloating: Large liquidity injection ({ask_diff:.2f} contracts)"
                
        return False, "Normal book replenishment"
