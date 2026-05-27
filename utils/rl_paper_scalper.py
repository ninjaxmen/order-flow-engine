import sys
import os
import time
import json
import asyncio
from typing import List, Dict, Tuple, Optional

# Dynamically append parent directory to sys.path for robust absolute package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from agent.brain import DeepOrderFlowAgent

class RLPaperScalper:
    """
    Intelligent online-learning paper scalper driven by the Deep Actor-Critic RL Brain.
    Compiles a real-time 34-dimensional feature vector of LOB, footprint, and alert states,
    queries the neural network policy for trading actions, and updates policy weights 
    using backpropagated PnL feedback.
    """
    def __init__(self, target_tp_points: float = 30.0, target_sl_points: float = 15.0):
        self.tp_points = target_tp_points
        self.sl_points = target_sl_points
        
        # Initialize Deep RL Agent (34 inputs, 4 outputs: HOLD, BUY, SELL, CLOSE)
        self.agent = DeepOrderFlowAgent(state_dim=34, action_dim=4, learning_rate=0.001)
        
        # Position state
        self.active_position: Optional[dict] = None
        self.trades_history: List[dict] = []
        
        self.cache_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "cache", 
            "paper_scalper_trades.json"
        )
        
        # Session metrics
        self.session_pnl = 0.0
        self.wins = 0
        self.losses = 0
        
        # Real-time state metrics cache (for feature compiling)
        self.last_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        
        # Depth cache
        self.bids_depth: List[Tuple[float, float]] = []
        self.asks_depth: List[Tuple[float, float]] = []
        
        # Footprint cache
        self.active_bar_volume = 0.0
        self.active_bar_delta = 0.0
        self.cvd = 0.0
        self.cvd_history: List[Tuple[float, float]] = [] # (timestamp, cvd)
        
        # Signals cache with active decay timestamps
        self.signal_events: Dict[str, float] = {
            "absorption_buy": 0.0,
            "absorption_sell": 0.0,
            "exhaustion_buy": 0.0,
            "exhaustion_sell": 0.0,
            "buying_imbalance": 0.0,
            "selling_imbalance": 0.0,
            "spoofing_bid": 0.0,
            "spoofing_ask": 0.0
        }
        
        # Profile nodes cache
        self.poc_price = 0.0
        self.hvn_price = 0.0
        self.lvn_price = 0.0
        
        # Price history for momentum features
        self.price_history: List[Tuple[float, float]] = [] # (timestamp, price)
        
        # Algorithmic risk control: entry cooldown
        self.last_entry_time = 0.0

    def load_history(self):
        """Loads previous trade history and resumes outstanding active trades to prevent duplicate entries."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.trades_history = json.load(f)
                
                closed_trades = [t for t in self.trades_history if t["status"] == "CLOSED"]
                self.wins = sum(1 for t in closed_trades if t["pnl"] > 0)
                self.losses = sum(1 for t in closed_trades if t["pnl"] <= 0)
                self.session_pnl = sum(t["pnl"] for t in closed_trades)
                
                # Critical bug fix: Resume any outstanding active position from cache
                active_trades = [t for t in self.trades_history if t["status"] == "ACTIVE"]
                if active_trades:
                    self.active_position = active_trades[-1]
                    logger.info(f"Resumed active LONG/SHORT position from cache: {self.active_position['direction']} entered at {self.active_position['entry_price']:.2f}")
                
                logger.info(f"Session metrics: Wins: {self.wins} | Losses: {self.losses} | Session PnL: {self.session_pnl:+.2f}")
            except Exception as e:
                logger.error(f"Failed to load paper history: {str(e)}")

    def save_history(self):
        """Saves trade history to cache database."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.trades_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper history: {str(e)}")

    def compile_state_vector(self) -> List[float]:
        """
        Compiles the current order flow, depth, footprint, signals, and position variables
        into a synchronized 34-dimensional feature vector.
        """
        state = [0.0] * 34
        now = time.time()
        
        # 1-5. Bid depth ratio at Level 1, 2, 3, 5, 10
        for i, idx in enumerate([0, 1, 2, 4, 9]):
            if len(self.bids_depth) > idx and len(self.asks_depth) > idx:
                bid_vol = sum(self.bids_depth[j][1] for j in range(idx + 1))
                ask_vol = sum(self.asks_depth[j][1] for j in range(idx + 1))
                state[i] = bid_vol / (bid_vol + ask_vol + 1e-6)
        
        # 6-10. Ask depth ratio at Level 1, 2, 3, 5, 10
        for i, idx in enumerate([0, 1, 2, 4, 9]):
            if len(self.bids_depth) > idx and len(self.asks_depth) > idx:
                bid_vol = sum(self.bids_depth[j][1] for j in range(idx + 1))
                ask_vol = sum(self.asks_depth[j][1] for j in range(idx + 1))
                state[5 + i] = ask_vol / (bid_vol + ask_vol + 1e-6)
                
        # 11. Normalized spread
        spread = max(0.0, self.best_ask - self.best_bid)
        state[10] = min(1.0, spread / 10.0)
        
        # 12-13. Active footprint bar metrics
        state[11] = min(5.0, self.active_bar_volume / 5000.0)
        state[12] = max(-5.0, min(5.0, self.active_bar_delta / 5000.0))
        
        # 14. CVD normalized
        state[13] = max(-10.0, min(10.0, self.cvd / 100000.0))
        
        # 15. CVD trend (change over last 10 seconds)
        cvd_change = 0.0
        past_cvds = [c for t, c in self.cvd_history if now - t > 10.0]
        if past_cvds:
            cvd_change = self.cvd - past_cvds[-1]
        state[14] = max(-5.0, min(5.0, cvd_change / 20000.0))
        
        # 16-23. Binary signal indicators with 30s exponential decay
        decay_period = 30.0
        for idx, key in enumerate([
            "absorption_buy", "absorption_sell", 
            "exhaustion_buy", "exhaustion_sell",
            "buying_imbalance", "selling_imbalance", 
            "spoofing_bid", "spoofing_ask"
        ]):
            trigger_time = self.signal_events[key]
            elapsed = now - trigger_time
            if elapsed < decay_period:
                state[15 + idx] = 1.0 - (elapsed / decay_period)
                
        # 24-26. Distance to profile nodes
        state[23] = min(5.0, abs(self.last_price - self.poc_price) / 50.0) if self.poc_price > 0 else 0.0
        state[24] = min(5.0, abs(self.last_price - self.hvn_price) / 50.0) if self.hvn_price > 0 else 0.0
        state[25] = min(5.0, abs(self.last_price - self.lvn_price) / 50.0) if self.lvn_price > 0 else 0.0
        
        # 27-30. Recent price changes (momentum) over last 1, 3, 5, 10 seconds
        for i, elapsed_sec in enumerate([1.0, 3.0, 5.0, 10.0]):
            past_prices = [p for t, p in self.price_history if now - t > elapsed_sec]
            momentum = 0.0
            if past_prices and self.last_price > 0:
                momentum = (self.last_price - past_prices[-1]) / 10.0
            state[26 + i] = max(-5.0, min(5.0, momentum))
            
        # 31. Active position direction (1 for LONG, -1 for SHORT, 0 for NONE)
        if self.active_position:
            state[30] = 1.0 if self.active_position["direction"] == "LONG" else -1.0
            # 32. Unrealized PnL
            state[31] = max(-3.0, min(3.0, self.active_position["pnl"] / 30.0))
            # 33. Normalized duration active
            duration = now - self.active_position["entry_time"]
            state[32] = min(2.0, duration / 300.0)
        else:
            state[30] = 0.0
            state[31] = 0.0
            state[32] = 0.0
            
        # 34. Session PnL
        state[33] = max(-10.0, min(10.0, self.session_pnl / 100.0))
        
        return state

    def handle_signal_alert(self, alert: dict):
        """Records incoming heuristics events and updates their signal timestamps."""
        now = time.time()
        sig_type = alert.get("signal_type", "")
        evidence = alert.get("evidence", "").lower()
        
        if sig_type == "LIQUIDITY_SPOOFING":
            if "bid" in evidence:
                self.signal_events["spoofing_bid"] = now
            elif "ask" in evidence:
                self.signal_events["spoofing_ask"] = now
        elif sig_type == "ABSORPTION_BUY":
            self.signal_events["absorption_buy"] = now
        elif sig_type == "ABSORPTION_SELL":
            self.signal_events["absorption_sell"] = now
        elif sig_type == "EXHAUSTION_BUY":
            self.signal_events["exhaustion_buy"] = now
        elif sig_type == "EXHAUSTION_SELL":
            self.signal_events["exhaustion_sell"] = now
        elif sig_type == "AGGRESSIVE_BUYING_IMBALANCE":
            self.signal_events["buying_imbalance"] = now
        elif sig_type == "AGGRESSIVE_SELLING_IMBALANCE":
            self.signal_events["selling_imbalance"] = now

    def check_position_limits(self, last_price: float):
        """Monitors active trades against stop loss / take profit targets and triggers updates."""
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
            
            outcome = "WIN (Take Profit Hit)" if pnl > 0 else "LOSS (Stop Loss Hit)"
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
                
            self.session_pnl += pnl
            self.active_position = None
            self.save_history()
            
            logger.info(
                f"\n[EXIT] position closed: {direction} | Entry: {entry:.2f} -> Exit: {last_price:.2f} | "
                f"PnL: {pnl:+.2f} | Outcome: {outcome} | Total Session PnL: {self.session_pnl:+.2f}\n"
            )
            
            # Send Temporal Difference (TD) feedback to agent brain to update NN parameters
            next_state = self.compile_state_vector()
            self.agent.learn_from_feedback(pnl, next_state, done=True)
            
            # Print diagnostic stats of the RL policy
            metrics = self.agent.export_policy_metrics()
            logger.info(f"[RL BRAIN DIAGNOSTICS] Brain Harness: {metrics['harness']} | Replay buffer: {metrics['buffer_size']}")

    def evaluate_rl_policy(self):
        """Queries the PyTorch Brain on the compiled state vector and executes optimal trade action."""
        if self.last_price <= 0:
            return
            
        state = self.compile_state_vector()
        action_idx, action_label, confidence = self.agent.select_action(state, self.last_price)
        
        # 1. HOLD or CLOSE
        if self.active_position:
            if action_label == "CLOSE" or action_idx == 3:
                pos = self.active_position
                direction = pos["direction"]
                pnl = pos["pnl"]
                
                pos["status"] = "CLOSED"
                pos["exit_price"] = self.last_price
                pos["exit_time"] = time.time()
                
                if pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                    
                self.session_pnl += pnl
                self.active_position = None
                self.save_history()
                
                logger.info(
                    f"\n[RL POLICY EXIT] CLOSE order triggered by neural network: {direction} | "
                    f"PnL: {pnl:+.2f} | Total Session PnL: {self.session_pnl:+.2f}\n"
                )
                
                next_state = self.compile_state_vector()
                self.agent.learn_from_feedback(pnl, next_state, done=True)
            return

        # Check algorithmic entry cooldown to prevent high-frequency over-trading noise
        now = time.time()
        if now - self.last_entry_time < 30.0:
            return

        # 2. BUY (LONG) ENTRY
        if (action_label == "BUY" or action_idx == 1) and self.active_position is None:
            tp = self.last_price + self.tp_points
            sl = self.last_price - self.sl_points
            
            self.active_position = {
                "asset": "BTCUSD",
                "direction": "LONG",
                "entry_price": self.last_price,
                "tp": tp,
                "sl": sl,
                "entry_time": now,
                "trigger": f"Deep RL policy BUY (confidence {confidence:.2f})",
                "status": "ACTIVE",
                "pnl": 0.0,
                "exit_price": None
            }
            
            self.last_entry_time = now
            self.trades_history.append(self.active_position)
            self.save_history()
            
            logger.info(
                f"\n[RL POLICY ENTRY] POSITION ENTERED (LONG) | Entry: {self.last_price:.2f} | "
                f"TP: {tp:.2f} | SL: {sl:.2f} | Confidence: {confidence:.2f}\n"
            )
            
        # 3. SELL (SHORT) ENTRY
        elif (action_label == "SELL" or action_idx == 2) and self.active_position is None:
            tp = self.last_price - self.tp_points
            sl = self.last_price + self.sl_points
            
            self.active_position = {
                "asset": "BTCUSD",
                "direction": "SHORT",
                "entry_price": self.last_price,
                "tp": tp,
                "sl": sl,
                "entry_time": now,
                "trigger": f"Deep RL policy SELL (confidence {confidence:.2f})",
                "status": "ACTIVE",
                "pnl": 0.0,
                "exit_price": None
            }
            
            self.last_entry_time = now
            self.trades_history.append(self.active_position)
            self.save_history()
            
            logger.info(
                f"\n[RL POLICY ENTRY] POSITION ENTERED (SHORT) | Entry: {self.last_price:.2f} | "
                f"TP: {tp:.2f} | SL: {sl:.2f} | Confidence: {confidence:.2f}\n"
            )

    async def run(self):
        """Establishes raw connection to local SSE server and streams live order flow metrics."""
        self.load_history()
        logger.info(f"Deep RL Paper Scalper successfully initialized using: {self.agent.harness_type}")
        logger.info("Connecting to live dashboard engine at http://127.0.0.1:8080/stream...")
        
        while True:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", 8080)
                
                # Send standard HTTP request
                req = "GET /stream HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: keep-alive\r\n\r\n"
                writer.write(req.encode('utf-8'))
                await writer.drain()
                
                logger.info("Connected. Streaming live metric matrices...")

                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        try:
                            data = json.loads(json_str)
                            now = time.time()
                            
                            # 1. Update order book cache
                            if data.get("type") == "DEPTH":
                                self.bids_depth = data.get("bids", [])
                                self.asks_depth = data.get("asks", [])
                                if self.bids_depth and self.asks_depth:
                                    self.best_bid = float(self.bids_depth[0][0])
                                    self.best_ask = float(self.asks_depth[0][0])
                                    self.last_price = (self.best_bid + self.best_ask) / 2.0
                                    
                                    # Cache price momentum history (keep last 30s)
                                    self.price_history.append((now, self.last_price))
                                    self.price_history = [(t, p) for t, p in self.price_history if now - t <= 30.0]
                                    
                                    # Monitor bracket risk limits
                                    self.check_position_limits(self.last_price)
                                    
                                    # Query RL policy for CLOSE actions on LOB ticks if position is active
                                    if self.active_position:
                                        self.evaluate_rl_policy()
                                    
                            # 2. Update footprint grid cache
                            elif data.get("type") == "FOOTPRINT":
                                self.active_bar_delta = float(data.get("total_delta", 0.0))
                                vol_grid = data.get("volume_at_price", {})
                                self.active_bar_volume = sum(
                                    float(v.get("bid_vol", 0.0)) + float(v.get("ask_vol", 0.0)) 
                                    for v in vol_grid.values()
                                )
                                
                            # 3. Process signal alerts and evaluate policy for dynamic ENTRIES
                            elif "alert" in data:
                                self.handle_signal_alert(data["alert"])
                                if not self.active_position:
                                    self.evaluate_rl_policy()
                                
                        except Exception:
                            pass
                            
                writer.close()
                await writer.wait_closed()
                
            except ConnectionRefusedError:
                logger.warning("Visual Dashboard engine not online. Retrying connection in 3 seconds...")
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"RL Paper Scalper stream error: {str(e)}")
                await asyncio.sleep(2.0)

if __name__ == "__main__":
    scalper = RLPaperScalper()
    try:
        asyncio.run(scalper.run())
    except KeyboardInterrupt:
        logger.info("RL Paper Scalper terminated by user.")
