from typing import Optional, Dict, List
from processor.footprint import FootprintBar
from config import settings
from utils.logger import logger

class ExhaustionAnalyser:
    """
    Exhaustion occurs when price attempts to make a new high or low, but aggressive volume 
    at the extreme edge dries up completely. This signals that aggressive buyers (at highs) 
    or aggressive sellers (at lows) are out of fuel, leading to high-probability reversals.
    """
    def __init__(self, exhaustion_ratio: float = settings.EXHAUSTION_VOLUME_RATIO):
        self.exhaustion_ratio = exhaustion_ratio

    def analyze(self, history: List[FootprintBar]) -> Optional[Dict]:
        """
        Analyzes historical footprint bars to spot exhaustion at extremums.
        """
        if len(history) < 2:
            return None

        # Look at the most recently completed bar
        bar = history[-2]
        
        # Check that we have a valid high and low
        if bar.high == -float('inf') or bar.low == float('inf') or not bar.volume_at_price:
            return None

        # We look at the extreme high and extreme low of the bar
        high_price = round(bar.high, 4)
        low_price = round(bar.low, 4)

        # Average volume per price level in the bar to establish a baseline
        total_levels = len(bar.volume_at_price)
        if total_levels < 3:
            return None
            
        avg_level_volume = bar.total_volume / total_levels

        # 1. Buying Exhaustion at the High of the bar
        # Buy volume (Ask Vol) at the high price level should be very low compared to average bar volume
        high_vol_data = bar.volume_at_price.get(high_price, {"bid_vol": 0.0, "ask_vol": 0.0})
        ask_vol_at_high = high_vol_data["ask_vol"]
        
        if ask_vol_at_high > 0 and ask_vol_at_high < avg_level_volume * self.exhaustion_ratio:
            # Confirm we had some trading volume in the bar overall
            if bar.total_volume > 100.0:
                logger.info(f"Detected Buying Exhaustion at bar high {high_price}. Ask Vol: {ask_vol_at_high:.2f} (Avg: {avg_level_volume:.2f})")
                return {
                    "type": "EXHAUSTION_SELL",
                    "price": high_price,
                    "volume": ask_vol_at_high,
                    "evidence": f"Aggressive buyers dried up at bar high {high_price} (Ask Vol {ask_vol_at_high:.2f} vs Avg {avg_level_volume:.2f})",
                    "confidence": 0.75
                }

        # 2. Selling Exhaustion at the Low of the bar
        # Sell volume (Bid Vol) at the low price level should be very low compared to average bar volume
        low_vol_data = bar.volume_at_price.get(low_price, {"bid_vol": 0.0, "ask_vol": 0.0})
        bid_vol_at_low = low_vol_data["bid_vol"]

        if bid_vol_at_low > 0 and bid_vol_at_low < avg_level_volume * self.exhaustion_ratio:
            if bar.total_volume > 100.0:
                logger.info(f"Detected Selling Exhaustion at bar low {low_price}. Bid Vol: {bid_vol_at_low:.2f} (Avg: {avg_level_volume:.2f})")
                return {
                    "type": "EXHAUSTION_BUY",
                    "price": low_price,
                    "volume": bid_vol_at_low,
                    "evidence": f"Aggressive sellers dried up at bar low {low_price} (Bid Vol {bid_vol_at_low:.2f} vs Avg {avg_level_volume:.2f})",
                    "confidence": 0.75
                }

        return None
