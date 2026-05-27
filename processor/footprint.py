import math
from typing import Dict, List, Optional
from config import settings
from utils.logger import logger

class FootprintBar:
    """
    Represents a single Footprint Bar for a discrete time interval.
    Tracks aggressive bid and ask volumes binned at customized price scales.
    """
    def __init__(self, start_timestamp_sec: int, bin_size: float = 0.5):
        self.start_time = start_timestamp_sec
        self.bin_size = bin_size
        # price -> {"bid_vol": float, "ask_vol": float}
        self.volume_at_price: Dict[float, Dict[str, float]] = {}
        
        self.open: Optional[float] = None
        self.high: float = -float('inf')
        self.low: float = float('inf')
        self.close: Optional[float] = None
        
        self.total_volume = 0.0
        self.total_delta = 0.0

    def bin_price(self, price: float) -> float:
        """
        Bins price levels dynamically.
        For example: price 100.27 with bin_size 0.5 is binned into 100.0.
        """
        return round(math.floor(price / self.bin_size) * self.bin_size, 4)

    def add_trade(self, price: float, volume: float, side: str) -> None:
        """
        Registers aggressive volume at the binned price level.
        Side: 'BUY' (hits ask, recorded as ask_vol) or 'SELL' (hits bid, recorded as bid_vol).
        """
        # Initialize OHLC bounds
        if self.open is None:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price

        # Group raw price into standard binned tick level
        price_key = self.bin_price(price)
        
        if price_key not in self.volume_at_price:
            self.volume_at_price[price_key] = {"bid_vol": 0.0, "ask_vol": 0.0}

        if side == "BUY":
            self.volume_at_price[price_key]["ask_vol"] += volume
            self.total_delta += volume
        elif side == "SELL":
            self.volume_at_price[price_key]["bid_vol"] += volume
            self.total_delta -= volume

        self.total_volume += volume


class FootprintStateEngine:
    """
    State manager tracking active and historical footprint bars for an asset.
    Manages rollover logic and dynamic grouping profiles.
    """
    def __init__(self, symbol: str, interval_sec: int = settings.FOOTPRINT_INTERVAL_SEC, bin_size: float = 0.5):
        self.symbol = symbol
        self.interval_sec = interval_sec
        self.bin_size = bin_size
        self.bars: List[FootprintBar] = []
        self.active_bar: Optional[FootprintBar] = None

    def add_tick(self, price: float, volume: float, side: str, timestamp_ns: int) -> None:
        """
        Injects a tick event into the footprint engine.
        Triggers bar rollover if the tick's timestamp falls in a new interval.
        """
        timestamp_sec = int(timestamp_ns / 1_000_000_000)
        bar_start_time = timestamp_sec - (timestamp_sec % self.interval_sec)

        # Check if we need to roll over to a new bar
        if self.active_bar is None:
            self.active_bar = FootprintBar(bar_start_time, self.bin_size)
            self.bars.append(self.active_bar)
            logger.info(f"Initialized first Footprint Bar for {self.symbol} at {bar_start_time} | Bin size: {self.bin_size}")
            
        elif bar_start_time > self.active_bar.start_time:
            logger.info(
                f"Rolling over Footprint Bar for {self.symbol}. "
                f"Completed bar at {self.active_bar.start_time} | Total Vol: {self.active_bar.total_volume:.2f}"
            )
            
            # Start new bar
            self.active_bar = FootprintBar(bar_start_time, self.bin_size)
            self.bars.append(self.active_bar)
            
            # Clean up memory
            if len(self.bars) > settings.MAX_Footprint_Hist_Bars:
                self.bars.pop(0)

        # Add transaction
        self.active_bar.add_trade(price, volume, side)

    def load_from_cache_data(self, cached_bars: List[dict]) -> None:
        """
        Reconstructs footprint historical logs from cached dictionary logs.
        """
        self.bars = []
        for raw in cached_bars:
            bar = FootprintBar(raw["start_time"], self.bin_size)
            bar.open = raw.get("open")
            bar.high = raw.get("high", -float('inf'))
            bar.low = raw.get("low", float('inf'))
            bar.close = raw.get("close")
            bar.total_volume = raw.get("total_volume", 0.0)
            bar.total_delta = raw.get("total_delta", 0.0)
            
            # Reconstruct volume map keys as float
            bar.volume_at_price = {float(k): v for k, v in raw.get("volume_at_price", {}).items()}
            self.bars.append(bar)
            
        if self.bars:
            self.active_bar = self.bars[-1]
            logger.info(f"Successfully restored {len(self.bars)} historical footprint bars from cache database.")

    def get_active_bar(self) -> Optional[FootprintBar]:
        return self.active_bar

    def get_history(self) -> List[FootprintBar]:
        return self.bars
