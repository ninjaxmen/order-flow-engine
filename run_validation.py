import sys
from utils.logger import logger
from utils.playback import TickPlaybackHarness
from processor.footprint import FootprintStateEngine
from processor.metrics import MetricsCalculator
from processor.book import OrderBook

def run_validation_suite():
    logger.info("==================================================")
    logger.info("RUNNING DETERMINISTIC FOOTPRINT VALIDATION SUITE")
    logger.info("==================================================")
    
    # Initialize Engine
    symbol = "DHAN_1333"
    footprint_engine = FootprintStateEngine(symbol, interval_sec=60, bin_size=0.5)
    metrics_calc = MetricsCalculator()
    
    # 1. Fetch deterministic tick ticks
    ticks = TickPlaybackHarness.generate_deterministic_ticks()
    logger.info(f"Loaded {len(ticks)} deterministic ticks for simulation.")

    # 2. Feed ticks into the footprint engine
    for tick in ticks:
        footprint_engine.add_tick(
            price=tick["price"],
            volume=tick["volume"],
            side=tick["side"],
            timestamp_ns=tick["timestamp_ns"]
        )

    # 3. Access the resulting footprint bar
    bars = footprint_engine.get_history()
    if not bars:
        logger.error("VALIDATION FAILED: No footprint bars generated.")
        sys.exit(1)
        
    bar = bars[0]
    
    # 4. Assert Invariants
    logger.info("Evaluating Mathematical Invariants...")

    # Invariant 1: Total Volume at price level 100.0
    # Expected Ask = 100 * 10 = 1000.0
    # Expected Bid = 50 * 5 = 250.0
    # Total = 1250.0
    vol_100 = bar.volume_at_price.get(100.0)
    if not vol_100:
        logger.error("VALIDATION FAILED: No footprint level recorded for price 100.0")
        sys.exit(1)
        
    assert vol_100["ask_vol"] == 1000.0, f"Expected Ask Vol 1000.0 at 100.0, got {vol_100['ask_vol']}"
    assert vol_100["bid_vol"] == 250.0, f"Expected Bid Vol 250.0 at 100.0, got {vol_100['bid_vol']}"
    logger.info("[OK] Invariant 1 Passed: Exact Ask/Bid volumes verified at price 100.0")

    # Invariant 2: Total Volume at price level 100.5
    # Expected Ask = 50 * 15 = 750.0
    # Expected Bid = 0.0
    vol_100_5 = bar.volume_at_price.get(100.5)
    if not vol_100_5:
        logger.error("VALIDATION FAILED: No footprint level recorded for price 100.5")
        sys.exit(1)
        
    assert vol_100_5["ask_vol"] == 750.0, f"Expected Ask Vol 750.0 at 100.5, got {vol_100_5['ask_vol']}"
    assert vol_100_5["bid_vol"] == 0.0, f"Expected Bid Vol 0.0 at 100.5, got {vol_100_5['bid_vol']}"
    logger.info("[OK] Invariant 2 Passed: Exact Ask/Bid volumes verified at price 100.5")

    # Invariant 3: Total Bar Volume
    # Expected Total = 1000 + 250 + 750 = 2000.0
    assert bar.total_volume == 2000.0, f"Expected Total Bar Volume 2000.0, got {bar.total_volume}"
    logger.info("[OK] Invariant 3 Passed: Total Bar Volume matches sum of aggressive trades")

    # Invariant 4: Net Delta
    # Expected Delta = 1000 - 250 + 750 = 1500.0
    assert bar.total_delta == 1500.0, f"Expected Total Bar Delta 1500.0, got {bar.total_delta}"
    logger.info("[OK] Invariant 4 Passed: Net bar Delta matches net aggressor volume differences")

    # Invariant 5: Diagonal Imbalances
    # High price = 100.5, Ask Vol = 750.0
    # Low price = 100.0, Bid Vol = 250.0
    # Ratio = 750.0 / 250.0 = 3.0x (Matches threshold exactly)
    imbalances = metrics_calc.calculate_diagonal_imbalances(bar)
    buying_imbalances = imbalances["buying"]
    
    assert len(buying_imbalances) >= 1, "Expected at least 1 diagonal Buying Imbalance, got 0"
    matched_imb = next((imb for imb in buying_imbalances if imb["price"] == 100.5), None)
    assert matched_imb is not None, "Buying imbalance not flagged at price 100.5"
    assert matched_imb["ratio"] == 3.0, f"Expected Imbalance Ratio to be 3.0, got {matched_imb['ratio']}"
    logger.info("[OK] Invariant 5 Passed: Diagonal Buying Imbalance (3.0x) correctly calculated and flagged")

    logger.info("==================================================")
    logger.info("ALL FOOTPRINT ENGINE MATHEMATICAL INVARIANTS PASSED!")
    logger.info("==================================================")

if __name__ == "__main__":
    run_validation_suite()
