import os
import json
from typing import List, Dict, Optional
from utils.logger import logger

class CachePersistenceManager:
    """
    Handles local memory cache backup and recovery.
    Periodically dumps in-memory footprint state into local storage to survive process restarts
    and allow warm reconnections without losing historical indicators.
    """
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_footprint_history(self, symbol: str, bars_history: list) -> bool:
        """
        Serializes and dumps historical footprint bars into a local JSON cache file.
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol.lower()}_footprint_cache.json")
            serialized_bars = []
            
            for bar in bars_history:
                serialized_bars.append({
                    "start_time": bar.start_time,
                    "volume_at_price": {str(k): v for k, v in bar.volume_at_price.items()},
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "total_volume": bar.total_volume,
                    "total_delta": bar.total_delta
                })
                
            with open(cache_file, mode='w', encoding='utf-8') as f:
                json.dump(serialized_bars, f, indent=2)
                
            logger.debug(f"Saved {len(serialized_bars)} footprint bars for {symbol} to cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to save footprint history cache for {symbol}: {str(e)}")
            return False

    def load_footprint_history(self, symbol: str) -> List[dict]:
        """
        Loads and recovers historical footprint bars from the local JSON cache file.
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol.lower()}_footprint_cache.json")
        if not os.path.exists(cache_file):
            return []

        try:
            with open(cache_file, mode='r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"Loaded {len(data)} cached footprint bars for {symbol} from storage.")
            return data
        except Exception as e:
            logger.error(f"Failed to load footprint history cache for {symbol}: {str(e)}")
            return []
