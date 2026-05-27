import asyncio
import csv
import time
import os
import math
from utils.logger import logger
from main import OrderFlowEngine

class OfflineTrainer:
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol
        self.orchestrator = OrderFlowEngine()
        
        # Disable networking and live visual components
        self.orchestrator.dashboard_server.broadcast_event = self._mock_broadcast
        self.orchestrator.dashboard_server.start = self._mock_start
        self.orchestrator.dashboard_server.stop = self._mock_stop
        
        # Override alert trigger to prevent console spam
        self.orchestrator.trigger_alert = self._mock_alert
        
        # Set up for fast RL training
        self.data_path = f"cache/{self.symbol}_historical_trades.csv"

    async def _mock_broadcast(self, event):
        pass
        
    async def _mock_start(self):
        pass
        
    async def _mock_stop(self):
        pass

    def _mock_alert(self, event):
        pass
        
    async def run(self):
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}. Please run data_fetcher.py first.")
            return
            
        logger.info(f"Loading {self.data_path} into memory...")
        
        # Read using standard csv
        trades = []
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
        
        logger.info("Initializing offline orchestrator components...")
        self.orchestrator.init_symbol_buffers(self.symbol)
        
        total_rows = len(trades)
        logger.info(f"Starting High-Speed Offline Simulation: {total_rows} ticks...")
        
        # Bypass queue limits for offline training
        self.orchestrator.event_queue = asyncio.Queue()
        
        # Patch time.time globally to simulate historical time
        global sim_time
        sim_time = time.time()
        time.time = lambda: sim_time
        
        for idx, row in enumerate(trades):
            # Update simulated time
            sim_time = int(row["timestamp"]) / 1000.0
            
            # Mock DEPTH event
            event_depth = {
                "symbol": self.symbol,
                "type": "DEPTH",
                "bids": [[float(row["price"]) - 0.5, float(row["quantity"])]],
                "asks": [[float(row["price"]) + 0.5, float(row["quantity"])]]
            }
            
            # Construct mock TICK event
            event_tick = {
                "symbol": self.symbol,
                "type": "TICK",
                "price": float(row["price"]),
                "volume": float(row["quantity"]),
                "side": row["direction"],
                "timestamp_ns": int(row["timestamp"]) * 1_000_000
            }
            
            self.orchestrator.event_queue.put_nowait(event_depth)
            self.orchestrator.event_queue.put_nowait(event_tick)
            
        logger.info(f"Queued {self.orchestrator.event_queue.qsize()} events.")
        
        # Start the RL Subprocess since we bypassed orchestrator.start()
        from rl_process import RLProcess
        self.orchestrator.rl_process = RLProcess(self.orchestrator.rl_request_queue, self.orchestrator.rl_response_queue)
        self.orchestrator.rl_process.start()
        
        self.orchestrator.is_running = True
        loop_task = asyncio.create_task(self.orchestrator.processing_loop())
        
        start_time = time.time()
        while self.orchestrator.event_queue.qsize() > 0:
            remaining = self.orchestrator.event_queue.qsize()
            processed = total_rows * 2 - remaining  # 2 events per row
            elapsed = time.time() - start_time
            tps = processed / max(1.0, elapsed)
            logger.info(f"Progress: {processed}/{(total_rows*2)} events | {tps:.2f} events/sec | Session PnL: {self.orchestrator.session_pnl:.4f} | Win Rate: {(self.orchestrator.wins / max(1, self.orchestrator.wins + self.orchestrator.losses) * 100):.1f}%")
            
            await asyncio.sleep(2.0)
            
        self.orchestrator.is_running = False
        # wait a bit for task to close gracefully
        await asyncio.sleep(2.0)
        
        # Shutdown RL Process gracefully to trigger save_checkpoint
        self.orchestrator.rl_request_queue.put({"type": "SHUTDOWN"})
        self.orchestrator.rl_process.join(timeout=5.0)
        
        elapsed = time.time() - start_time
        logger.info(f"Offline Simulation Complete! Time: {elapsed:.2f}s")
        logger.info(f"Final Session PnL: {self.orchestrator.session_pnl:.4f}")
        logger.info(f"Total Wins: {self.orchestrator.wins} | Losses: {self.orchestrator.losses}")

if __name__ == "__main__":
    trainer = OfflineTrainer("BTCUSD")
    asyncio.run(trainer.run())
