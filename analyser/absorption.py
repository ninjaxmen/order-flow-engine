from typing import Optional, Dict
from processor.footprint import FootprintBar
from processor.book import OrderBook
from config import settings
from utils.logger import logger

class AbsorptionAnalyser:
    """
    Heuristics engine to detect Passive Absorption and Iceberg order blockages.
    Absorption occurs when aggressive market orders execute heavily at a price level 
    but passive limit orders completely absorb the flow, preventing the price from moving further.
    """
    def __init__(self, min_vol_threshold: float = settings.ABSORPTION_MIN_VOL_THRESHOLD):
        self.min_vol_threshold = min_vol_threshold

    def analyze(self, bar: FootprintBar, book: OrderBook) -> Optional[Dict]:
        """
        Analyzes footprint and LOB to spot absorption at support or resistance.
        Returns a signal dictionary if absorption is detected, else None.
        """
        best_bid, _ = book.get_best_bid()
        best_ask, _ = book.get_best_ask()
        
        if best_bid == 0.0 or best_ask == 0.0:
            return None

        # Look for price levels with massive aggressive activity in the current bar
        for price, vols in bar.volume_at_price.items():
            bid_vol = vols["bid_vol"]
            ask_vol = vols["ask_vol"]

            # 1. Buying Absorption: Huge aggressive buys (Ask Volume) but price cannot breach above
            if ask_vol >= self.min_vol_threshold:
                # If price is near best ask/bid, and price hasn't moved above this level
                if price >= best_ask - 0.5:
                    # Check if the ratio of buying is significantly higher than bid volume at this level
                    if ask_vol > bid_vol * 2.0:
                        logger.info(f"Detected potential Buying Absorption at {price}. Buyers hitting Ask: {ask_vol:.2f}, Sellers defending.")
                        return {
                            "type": "ABSORPTION_SELL",
                            "price": price,
                            "volume": ask_vol,
                            "evidence": f"Heavy aggressive buying ({ask_vol:.2f} contracts) absorbed at Ask resistance {price}",
                            "confidence": min(0.95, 0.5 + (ask_vol / self.min_vol_threshold) * 0.1)
                        }

            # 2. Selling Absorption: Huge aggressive sells (Bid Volume) but price cannot breach below
            if bid_vol >= self.min_vol_threshold:
                if price <= best_bid + 0.5:
                    if bid_vol > ask_vol * 2.0:
                        logger.info(f"Detected potential Selling Absorption at {price}. Sellers hitting Bid: {bid_vol:.2f}, Buyers defending.")
                        return {
                            "type": "ABSORPTION_BUY",
                            "price": price,
                            "volume": bid_vol,
                            "evidence": f"Heavy aggressive selling ({bid_vol:.2f} contracts) absorbed at Bid support {price}",
                            "confidence": min(0.95, 0.5 + (bid_vol / self.min_vol_threshold) * 0.1)
                        }

        return None
