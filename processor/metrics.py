from typing import Dict, List, Tuple, Optional
from processor.footprint import FootprintBar
from config import settings
from utils.logger import logger

class MetricsCalculator:
    """
    Computes mathematical indicators from Footprint State.
    Calculates Cumulative Volume Delta (CVD), Delta Divergences, and diagonal Volume Imbalances.
    """
    def __init__(self):
        self.cvd_value = 0.0
        
        # VPIN Tracking
        self.buy_vol_bucket = 0.0
        self.sell_vol_bucket = 0.0
        self.vpin = 0.0
        self.volume_history = []  # List of tuples: (abs_imbalance, total_vol)
        
        # Hawkes Process Tracking
        self.hawkes_intensity = 0.0
        self.last_update_ms = 0

    def update_advanced_metrics(self, active_bar: FootprintBar, current_time_ms: int):
        """
        Updates VPIN and Hawkes Intensity based on recent footprint bar activity.
        """
        import time
        if self.last_update_ms == 0:
            self.last_update_ms = current_time_ms
        
        dt_seconds = (current_time_ms - self.last_update_ms) / 1000.0
        self.last_update_ms = current_time_ms

        # 1. Hawkes Process (Self-Exciting Intensity)
        # Decay the previous intensity
        beta = 0.5  # Decay rate (half-life of ~1.4 seconds)
        alpha = 0.001 # Excitation factor per unit of volume
        
        self.hawkes_intensity *= __import__("math").exp(-beta * dt_seconds)
        
        # Add new excitation from recent volume (simulated as the absolute delta)
        recent_trade_intensity = abs(active_bar.total_delta)
        self.hawkes_intensity += alpha * recent_trade_intensity

        # 2. VPIN Calculation (Rolling window of 50 updates)
        buy_vol = sum(v["ask_vol"] for v in active_bar.volume_at_price.values())
        sell_vol = sum(v["bid_vol"] for v in active_bar.volume_at_price.values())
        
        abs_imbalance = abs(buy_vol - sell_vol)
        tot_vol = buy_vol + sell_vol
        
        self.volume_history.append((abs_imbalance, tot_vol))
        if len(self.volume_history) > 50:
            self.volume_history.pop(0)
            
        sum_imbalance = sum(x[0] for x in self.volume_history)
        sum_vol = sum(x[1] for x in self.volume_history)
        
        if sum_vol > 0:
            self.vpin = sum_imbalance / sum_vol
        else:
            self.vpin = 0.0

    def update_cvd(self, active_bar: FootprintBar) -> float:
        """
        Maintains cumulative volume delta over the trading session.
        """
        self.cvd_value += active_bar.total_delta
        return self.cvd_value

    def calculate_diagonal_imbalances(self, bar: FootprintBar) -> Dict[str, List[Dict]]:
        """
        Calculates diagonal footprint volume imbalances.
        Compares aggressive BUY (Ask Volume) at price P with aggressive SELL (Bid Volume) at price P - tick_size.
        
        Formula:
          - Buying Imbalance: Ask_Vol(P) / Bid_Vol(P - tick_size) >= VOLUME_IMBALANCE_RATIO
          - Selling Imbalance: Bid_Vol(P - tick_size) / Ask_Vol(P) >= VOLUME_IMBALANCE_RATIO
        """
        imbalances = {
            "buying": [],
            "selling": []
        }
        
        prices = sorted(bar.volume_at_price.keys())
        if len(prices) < 2:
            return imbalances

        threshold = settings.VOLUME_IMBALANCE_RATIO
        tick_size = bar.bin_size

        # Walk through sorted price levels and compare diagonally
        for i in range(1, len(prices)):
            price_low = prices[i - 1]
            price_high = prices[i]
            
            # Check if they are adjacent ticks (within reasonable tolerance)
            if price_high - price_low > tick_size * 2.5:
                continue

            bid_vol_low = bar.volume_at_price[price_low]["bid_vol"]
            ask_vol_high = bar.volume_at_price[price_high]["ask_vol"]

            # 1. Buying Imbalance: Aggressive buyers hit the Ask at price_high hard compared to sellers at price_low
            if bid_vol_low > 0:
                ratio = ask_vol_high / bid_vol_low
                if ratio >= threshold and ask_vol_high > 10.0:
                    imbalances["buying"].append({
                        "price": price_high,
                        "bid_vol": bid_vol_low,
                        "ask_vol": ask_vol_high,
                        "ratio": ratio
                    })
            elif ask_vol_high >= threshold * 5.0:
                imbalances["buying"].append({
                    "price": price_high,
                    "bid_vol": 0.0,
                    "ask_vol": ask_vol_high,
                    "ratio": float('inf')
                })

            # 2. Selling Imbalance: Aggressive sellers hit the Bid at price_low hard compared to buyers at price_high
            if ask_vol_high > 0:
                ratio = bid_vol_low / ask_vol_high
                if ratio >= threshold and bid_vol_low > 10.0:
                    imbalances["selling"].append({
                        "price": price_low,
                        "bid_vol": bid_vol_low,
                        "ask_vol": ask_vol_high,
                        "ratio": ratio
                    })
            elif bid_vol_low >= threshold * 5.0:
                imbalances["selling"].append({
                    "price": price_low,
                    "bid_vol": bid_vol_low,
                    "ask_vol": 0.0,
                    "ratio": float('inf')
                })

        return imbalances

    def check_delta_divergence(self, history: List[FootprintBar]) -> Optional[str]:
        """
        Detects divergence between price movement and cumulative delta.
        - Bearish Divergence: Price makes a new high but CVD drops or fails to make a new high.
        - Bullish Divergence: Price makes a new low but CVD rises or fails to make a new low.
        """
        if len(history) < 3:
            return None

        bar1 = history[-3]
        bar2 = history[-2]

        if bar1.close is None or bar2.close is None:
            return None

        price_diff = bar2.close - bar1.close
        delta_diff = bar2.total_delta - bar1.total_delta

        if price_diff > 0 and delta_diff < 0:
            return "BEARISH_DIVERGENCE: Price closed higher but Aggressive Delta dropped (passive absorption/sellers stepping in)"
        
        if price_diff < 0 and delta_diff > 0:
            return "BULLISH_DIVERGENCE: Price closed lower but Aggressive Delta increased (passive absorption/buyers stepping in)"

        return None
