import sys
import os
import time
import json
import socket
import asyncio
from typing import List, Dict, Optional

# Dynamically append parent directory to sys.path for robust absolute package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

class PaperScalper:
    """
    Decoupled autonomous paper scalper.
    Listens to the live stream of order flow metrics from http://127.0.0.1:8080/stream,
    initiates scalps based on imbalances/absorptions, and tracks PnL.
    """
    def __init__(self, target_tp_points: float = 30.0, target_sl_points: float = 15.0):
        self.tp_points = target_tp_points
        self.sl_points = target_sl_points
        
        # Position state: {"direction": "LONG"/"SHORT", "entry_price": float, "tp": float, "sl": float, "timestamp": float}
        self.active_position: Optional[dict] = None
        self.trades_history: List[dict] = []
        
        self.cache_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "cache", 
            "paper_scalper_trades.json"
        )
        
        # Performance metrics
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

    def load_history(self):
        """Loads previous trade history if cache exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.trades_history = json.load(f)
                closed_trades = [t for t in self.trades_history if t["status"] == "CLOSED"]
                self.wins = sum(1 for t in closed_trades if t["pnl"] > 0)
                self.losses = sum(1 for t in closed_trades if t["pnl"] <= 0)
                self.total_pnl = sum(t["pnl"] for t in closed_trades)
                logger.info(f"Loaded {len(self.trades_history)} paper trades from cache. Historical PnL: {self.total_pnl:+.2f}")
            except Exception as e:
                logger.error(f"Failed to load paper scalper history: {str(e)}")

    def save_history(self):
        """Saves trade history to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.trades_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper scalper history: {str(e)}")

    def handle_signal_alert(self, alert: dict, current_price: float):
        """
        Evaluates signals from the order flow engine to open scalp positions.
        """
        if self.active_position is not None:
            return  # Limit to 1 active position at a time for strict risk control

        signal_type = alert.get("signal_type", "")
        evidence = alert.get("evidence", "")
        asset = alert.get("asset", "")

        direction = None
        trigger = ""

        # 1. Evaluate Long Entries
        if "BUY" in signal_type or "buying" in evidence.lower():
            direction = "LONG"
            trigger = f"{signal_type} ({evidence})"
            
        # 2. Evaluate Short Entries
        elif "SELL" in signal_type or "selling" in evidence.lower():
            direction = "SHORT"
            trigger = f"{signal_type} ({evidence})"

        if direction and current_price > 0:
            tp = current_price + self.tp_points if direction == "LONG" else current_price - self.tp_points
            sl = current_price - self.sl_points if direction == "LONG" else current_price + self.sl_points

            self.active_position = {
                "asset": asset,
                "direction": direction,
                "entry_price": current_price,
                "tp": tp,
                "sl": sl,
                "entry_time": time.time(),
                "trigger": trigger,
                "status": "ACTIVE",
                "pnl": 0.0,
                "exit_price": None
            }
            
            self.trades_history.append(self.active_position)
            self.save_history()
            
            logger.info(
                f"\n[ENTRY] POSITION ENTERED: {direction} at {current_price:.2f} | "
                f"TP: {tp:.2f} | SL: {sl:.2f} | Reason: {trigger}\n"
            )

    def check_position_limits(self, last_price: float):
        """
        Monitors active positions against Take-Profit and Stop-Loss levels on every price tick.
        """
        if not self.active_position:
            return

        pos = self.active_position
        direction = pos["direction"]
        entry = pos["entry_price"]

        pnl = (last_price - entry) if direction == "LONG" else (entry - last_price)
        pos["pnl"] = pnl

        is_tp = (last_price >= pos["tp"]) if direction == "LONG" else (last_price <= pos["tp"])
        is_sl = (last_price <= pos["sl"]) if direction == "LONG" else (last_price >= pos["sl"])

        if is_tp or is_sl:
            pos["status"] = "CLOSED"
            pos["exit_price"] = last_price
            pos["exit_time"] = time.time()
            
            if pnl > 0:
                self.wins += 1
                outcome = "WIN (Take Profit Hit)"
            else:
                self.losses += 1
                outcome = "LOSS (Stop Loss Hit)"

            self.total_pnl += pnl
            self.active_position = None
            self.save_history()
            
            logger.info(
                f"\n[EXIT] POSITION CLOSED: {direction} | Entry: {entry:.2f} -> Exit: {last_price:.2f} | "
                f"PnL: {pnl:+.2f} | Outcome: {outcome} | Total Session PnL: {self.total_pnl:+.2f}\n"
            )

    async def run(self):
        """
        Establishes raw connection to local SSE server and streams updates.
        """
        self.load_history()
        logger.info("Paper Scalper connecting to localhost:8080 stream...")
        
        while True:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", 8080)
                
                # Send standard HTTP request
                req = "GET /stream HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: keep-alive\r\n\r\n"
                writer.write(req.encode('utf-8'))
                await writer.drain()
                
                last_price = 0.0

                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        try:
                            data = json.loads(json_str)
                            
                            # Update latest price tracker
                            if data.get("type") == "DEPTH":
                                bids = data.get("bids", [])
                                asks = data.get("asks", [])
                                if bids and asks:
                                    best_bid = float(bids[0][0])
                                    best_ask = float(asks[0][0])
                                    last_price = (best_bid + best_ask) / 2.0
                                    self.check_position_limits(last_price)
                                    
                            # Process Signals/Alerts
                            elif "alert" in data:
                                self.handle_signal_alert(data["alert"], last_price)
                                
                        except Exception as e:
                            pass
                            
                writer.close()
                await writer.wait_closed()
                
            except ConnectionRefusedError:
                logger.warning("Visual Dashboard not running. Retrying in 3 seconds...")
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"Paper scalper stream error: {str(e)}")
                await asyncio.sleep(2.0)

if __name__ == "__main__":
    scalper = PaperScalper()
    asyncio.run(scalper.run())
