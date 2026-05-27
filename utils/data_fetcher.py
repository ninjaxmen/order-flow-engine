import urllib.request
import urllib.parse
import json
import time
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_fetcher")

def fetch_binance_agg_trades(symbol: str, target_trades: int = 100_000, output_dir: str = "cache"):
    """
    Downloads historical aggregate trades from Binance Futures.
    Used to reconstruct Order Flow (CVD, Footprints) for offline RL training.
    """
    os.makedirs(output_dir, exist_ok=True)
    binance_symbol = symbol.replace("USD", "USDT")
    output_file = os.path.join(output_dir, f"{symbol}_historical_trades.csv")
    
    logger.info(f"Fetching {target_trades} historical trades for {binance_symbol} from Binance Futures...")
    
    # First, get the most recent trade ID to work backwards
    url = f"https://fapi.binance.com/fapi/v1/aggTrades"
    try:
        req = urllib.request.Request(f"{url}?symbol={binance_symbol}&limit=1")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
        latest_trade = data[0]
        end_id = latest_trade["a"]
    except Exception as e:
        logger.error(f"Failed to fetch initial trade ID: {e}")
        return

    all_trades = []
    current_end_id = end_id
    limit = 1000
    
    # Estimate starting ID
    start_id = max(0, current_end_id - target_trades)
    current_from_id = start_id
    
    while len(all_trades) < target_trades:
        try:
            params = urllib.parse.urlencode({
                "symbol": binance_symbol,
                "fromId": current_from_id,
                "limit": limit
            })
            req = urllib.request.Request(f"{url}?{params}")
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if not data:
                break
                
            all_trades.extend(data)
            current_from_id = data[-1]["a"] + 1
            
            if len(all_trades) % 10_000 == 0:
                logger.info(f"Downloaded {len(all_trades)} / {target_trades} trades...")
                
            time.sleep(0.1)  # Respect rate limits
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            time.sleep(2)
    
    # Process into csv
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker", "direction", "signed_volume"])
        for trade in all_trades:
            price = float(trade["p"])
            quantity = float(trade["q"])
            is_buyer_maker = trade["m"]
            direction = "SELL" if is_buyer_maker else "BUY"
            signed_volume = -quantity if is_buyer_maker else quantity
            writer.writerow([
                trade["a"], price, quantity, trade["f"], trade["l"], trade["T"], is_buyer_maker, direction, signed_volume
            ])
    
    logger.info(f"Successfully saved {len(all_trades)} trades to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSD")
    parser.add_argument("--count", type=int, default=100_000)
    args = parser.parse_args()
    
    fetch_binance_agg_trades(args.symbol, args.count)
