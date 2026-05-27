from typing import Dict, List, Tuple
from processor.footprint import FootprintBar
from utils.logger import logger

class VolumeProfileAnalyser:
    """
    Computes a composite Volume Profile from a window of footprint bars.
    Identifies Point of Control (POC), High Volume Nodes (HVN), and Low Volume Nodes (LVN).
    """
    def __init__(self):
        pass

    def compute_profile(self, history: List[FootprintBar]) -> Dict:
        """
        Aggregates volume at each price level across the provided bars,
        then finds POC, HVNs, and LVNs.
        """
        profile: Dict[float, float] = {}

        # 1. Aggregate volume across all bars
        for bar in history:
            for price, vols in bar.volume_at_price.items():
                lvl_vol = vols["bid_vol"] + vols["ask_vol"]
                profile[price] = profile.get(price, 0.0) + lvl_vol

        if not profile:
            return {
                "poc": None,
                "hvns": [],
                "lvns": []
            }

        sorted_prices = sorted(profile.keys())
        volumes = [profile[p] for p in sorted_prices]

        # 2. Identify Point of Control (POC) - highest absolute volume
        poc_idx = int(profile_max_index := sorted_prices[volumes.index(max(volumes))])
        poc = profile_max_index

        # 3. Identify Peaks (HVN) and Valleys (LVN) using local extrema checks
        hvns = []
        lvns = []

        # We look at a window of 3 adjacent levels to find peaks and valleys
        for i in range(1, len(sorted_prices) - 1):
            prev_vol = volumes[i - 1]
            curr_vol = volumes[i]
            next_vol = volumes[i + 1]

            curr_price = sorted_prices[i]

            # Peak (Local Maxima) -> HVN
            if curr_vol > prev_vol and curr_vol > next_vol:
                hvns.append((curr_price, curr_vol))

            # Valley (Local Minima) -> LVN
            elif curr_vol < prev_vol and curr_vol < next_vol:
                lvns.append((curr_price, curr_vol))

        # Sort HVNs by volume (descending) and LVNs by volume (ascending)
        hvns = sorted(hvns, key=lambda x: x[1], reverse=True)
        lvns = sorted(lvns, key=lambda x: x[1])

        return {
            "poc": poc,
            "hvns": [p for p, _ in hvns[:3]], # top 3 high volume nodes
            "lvns": [p for p, _ in lvns[:3]]  # top 3 low volume nodes (support/resistance)
        }
