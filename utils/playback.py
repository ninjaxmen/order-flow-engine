import csv
import time
from typing import List, Dict
from utils.logger import logger

class TickPlaybackHarness:
    """
    Harness to stream pre-recorded or deterministic tick files.
    Allows exact replication of market profiles and footprint calculations for validation.
    """
    def __init__(self):
        pass

    @staticmethod
    def generate_deterministic_ticks() -> List[Dict]:
        """
        Generates a deterministic sequence of ticks with known profiles.
        Used to test footprints, delta metrics, and imbalances.
        """
        ticks = []
        base_time_ns = int(time.time() * 1_000_000_000)
        
        # 1. 100 Buy Ticks at price 100.0, volume 10.0
        # Expected Ask Vol at 100.0: 1000.0
        for i in range(100):
            ticks.append({
                "exchange": "DHAN",
                "symbol": "DHAN_1333",
                "type": "TICK",
                "price": 100.0,
                "volume": 10.0,
                "side": "BUY",
                "timestamp_ns": base_time_ns + i * 1_000_000 # increment by 1ms
            })

        # 2. 50 Sell Ticks at price 100.0, volume 5.0
        # Expected Bid Vol at 100.0: 250.0
        # Total Vol at 100.0: 1250.0, Net Delta: 750.0
        for i in range(50):
            ticks.append({
                "exchange": "DHAN",
                "symbol": "DHAN_1333",
                "type": "TICK",
                "price": 100.0,
                "volume": 5.0,
                "side": "SELL",
                "timestamp_ns": base_time_ns + (100 + i) * 1_000_000
            })

        # 3. 50 Buy Ticks at price 100.5, volume 15.0
        # Expected Ask Vol at 100.5: 750.0
        # This will set up a diagonal buy imbalance: Ask Vol(100.5) / Bid Vol(100.0) = 750.0 / 250.0 = 3.0x
        for i in range(50):
            ticks.append({
                "exchange": "DHAN",
                "symbol": "DHAN_1333",
                "type": "TICK",
                "price": 100.5,
                "volume": 15.0,
                "side": "BUY",
                "timestamp_ns": base_time_ns + (150 + i) * 1_000_000
            })

        return ticks

    @staticmethod
    def load_ticks_from_csv(filepath: str) -> List[Dict]:
        """
        Loads ticks from a CSV file.
        CSV format: timestamp_ns, symbol, price, volume, side, exchange
        """
        ticks = []
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticks.append({
                        "exchange": row["exchange"],
                        "symbol": row["symbol"],
                        "type": "TICK",
                        "price": float(row["price"]),
                        "volume": float(row["volume"]),
                        "side": row["side"],
                        "timestamp_ns": int(row["timestamp_ns"])
                    })
            logger.info(f"Successfully loaded {len(ticks)} ticks from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load ticks from CSV: {str(e)}")
        return ticks
