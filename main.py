import asyncio
import signal
import json
import time
import os
import sys
import random
from typing import Dict
from config import settings
from utils.logger import logger
from utils.server import OrderFlowDashboardServer
from connector.dhan import DhanConnector
from connector.delta_exchange import DeltaExchangeConnector
from processor.book import OrderBook
from processor.footprint import FootprintStateEngine
from processor.metrics import MetricsCalculator
from processor.cache import CachePersistenceManager
from analyser.absorption import AbsorptionAnalyser
from analyser.exhaustion import ExhaustionAnalyser
from analyser.profile import VolumeProfileAnalyser

# Dynamically append parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
from rl_process import RLProcess

from delta_rest_client import DeltaRestClient

class OrderFlowEngine:
    """
    Main orchestrator for the quantitative order flow and footprint analysis engine.
    Integrated with a lightweight Server-Sent Events (SSE) web dashboard server for live UI streaming.
    Contains the PyTorch Deep Reinforcement Learning online paper-trading agent.
    """
    def __init__(self):
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.TICK_BUFFER_SIZE)
        
        # Initialize Delta REST Client for direct trade placement on Delta Demo
        self.rest_client = DeltaRestClient(
            base_url=settings.DELTA_API_ENDPOINT,
            api_key=settings.DELTA_API_KEY,
            api_secret=settings.DELTA_API_SECRET
        )
        
        # Initialize Connectors (Focusing purely on Delta Exchange for now)
        self.connectors = [
            DeltaExchangeConnector(self.event_queue, simulation_mode=settings.SIMULATION_MODE)
        ]
        
        # In-Memory State Managers (Keyed by Symbol)
        self.books: Dict[str, OrderBook] = {}
        self.footprints: Dict[str, FootprintStateEngine] = {}
        self.metrics_calcs: Dict[str, MetricsCalculator] = {}
        
        # State tracker to record historical bar counts for cache writes
        self.bar_counts: Dict[str, int] = {}
        
        # Cache Manager for state persistence
        self.cache_manager = CachePersistenceManager()
        
        # Dashboard HTTP/SSE Server (Bind to 0.0.0.0 for external container access)
        self.dashboard_server = OrderFlowDashboardServer(host="0.0.0.0", on_kill_switch=self.trigger_kill_switch)
        
        # Heuristics Analyzers
        self.absorption_analyser = AbsorptionAnalyser()
        self.exhaustion_analyser = ExhaustionAnalyser()
        self.profile_analyser = VolumeProfileAnalyser()
        
        # Deep RL Multiprocessing Isolation
        self.rl_request_queue = mp.Queue()
        self.rl_response_queue = mp.Queue()
        self.rl_process = None
        self.agent_locked = False
        
        # RL Position & Session Stats (Multi-Asset)
        self.active_positions: Dict[str, dict] = {}
        self.trades_history = []
        self.session_pnl = 0.0
        self.wins = 0
        self.losses = 0
        
        # In-memory history for feature compilation (Multi-Asset)
        self.last_prices: Dict[str, float] = {}
        self.price_histories: Dict[str, list] = {}
        self.cvd_histories: Dict[str, list] = {}
        self.last_entry_times: Dict[str, float] = {}
        self.cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "paper_scalper_trades.json")
        
        # Signal triggers for time-decayed state features (Multi-Asset)
        self.signal_events: Dict[str, Dict[str, float]] = {}
        
        self.is_running = False
        self.process_task = None
        self.training_task = None
        
        # Risk Management State
        self.starting_capital = 100000.0
        self.agent_locked = False
        self.agent_paused = False
        self.latency_violations = 0

    def init_symbol_buffers(self, symbol: str) -> None:
        """Initializes all in-memory tracking structures for a symbol if not present."""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = 0.0
            self.price_histories[symbol] = []
            self.cvd_histories[symbol] = []
            self.last_entry_times[symbol] = 0.0
            self.active_positions[symbol] = None
            self.signal_events[symbol] = {
                "absorption_buy": 0.0,
                "absorption_sell": 0.0,
                "exhaustion_buy": 0.0,
                "exhaustion_sell": 0.0,
                "buying_imbalance": 0.0,
                "selling_imbalance": 0.0,
                "spoofing_bid": 0.0,
                "spoofing_ask": 0.0
            }

    async def start(self) -> None:
        """Starts all connectors, visual dashboard server, and the main processing loop."""
        self.is_running = True
        
        # Load Deep RL Agent paper trading stats
        self.load_history()
        
        # Start Isolated Multiprocessing RL Brain
        self.rl_process = RLProcess(self.rl_request_queue, self.rl_response_queue)
        self.rl_process.start()
        
        # Start Dashboard Server
        await self.dashboard_server.start()
        
        # Start Connectors
        for conn in self.connectors:
            await conn.run()
            
        # Start Processing Loops
        self.action_task = asyncio.create_task(self.action_listener())
        self.process_task = asyncio.create_task(self.processing_loop())
        logger.info("Order Flow Orchestrator running.")

    async def action_listener(self) -> None:
        """Continuously polls the isolated RL Response queue for ultra-low latency execution."""
        import queue
        while self.is_running:
            try:
                while True:
                    resp = self.rl_response_queue.get_nowait()
                    if resp["type"] == "ACTION":
                        self.execute_action(resp["symbol"], resp["action_idx"], resp["action_label"], resp["confidence"], resp["probs"])
            except queue.Empty:
                pass
            await asyncio.sleep(0.0001)

    async def processing_loop(self) -> None:
        """Consumes events from central queue and dispatches to states and heuristics."""
        logger.info("Main processing loop started.")
        while self.is_running:
            try:
                # Retrieve next event
                event = await self.event_queue.get()
                symbol = event["symbol"]
                
                # Latency Watchdog
                now_ns = int(time.time() * 1_000_000_000)
                event_ts = event.get("timestamp_ns", now_ns)
                latency_ms = (now_ns - event_ts) / 1_000_000.0
                
                if 100.0 < latency_ms < 60000.0:  # Ignore extreme clock skew
                    self.latency_violations += 1
                    if self.latency_violations >= 3:
                        if not self.agent_paused:
                            logger.warning(f"LATENCY WATCHDOG: WebSocket latency {latency_ms:.1f}ms > 100ms for 3 ticks. Soft-Pausing agent.")
                            self.agent_paused = True
                elif latency_ms <= 100.0:
                    if self.agent_paused:
                        logger.info(f"LATENCY WATCHDOG: Latency recovered ({latency_ms:.1f}ms). Resuming agent.")
                        self.agent_paused = False
                    self.latency_violations = 0
                
                # Proactively initialize state machines and buffers for new symbols
                if symbol not in self.books:
                    bin_size = 10.0 if "BTC" in symbol or "DELTA" in symbol else (2.0 if "ETH" in symbol else (0.5 if "SOL" in symbol or "AVAX" in symbol else 0.005))
                    
                    self.books[symbol] = OrderBook(symbol)
                    self.footprints[symbol] = FootprintStateEngine(symbol, bin_size=bin_size)
                    self.metrics_calcs[symbol] = MetricsCalculator()
                    self.bar_counts[symbol] = 0
                    self.init_symbol_buffers(symbol)
                    
                    # --- WARM BOOT RECOVERY ROUTINE ---
                    cached_bars = self.cache_manager.load_footprint_history(symbol)
                    if cached_bars:
                        self.footprints[symbol].load_from_cache_data(cached_bars)
                        self.bar_counts[symbol] = len(cached_bars)
                        # Rebuild Cumulative Volume Delta value from historical data
                        for bar in self.footprints[symbol].get_history():
                            self.metrics_calcs[symbol].update_cvd(bar)
                        logger.info(f"Engine recovered {len(cached_bars)} bars for {symbol}. Re-calculated baseline CVD.")

                book = self.books[symbol]
                footprint = self.footprints[symbol]
                metrics = self.metrics_calcs[symbol]

                # --- 1. LOB DEPTH EVENT PROCESSING ---
                if event["type"] == "DEPTH":
                    logger.info(f"[DEBUG] Processing DEPTH event for {symbol}")
                    book.update_depth(event["bids"], event["asks"])
                    

                    
                    # Update real-time price history and caches
                    best_bid, _ = book.get_best_bid()
                    best_ask, _ = book.get_best_ask()
                    if best_bid > 0 and best_ask > 0:
                        now = time.time()
                        self.last_prices[symbol] = (best_bid + best_ask) / 2.0
                        self.price_histories[symbol].append((now, self.last_prices[symbol]))
                        self.price_histories[symbol] = [(t, p) for t, p in self.price_histories[symbol] if now - t <= 30.0]
                        self.cvd_histories[symbol].append((now, metrics.cvd_value))
                        self.cvd_histories[symbol] = [(t, c) for t, c in self.cvd_histories[symbol] if now - t <= 30.0]
                        
                        # Monitor active trade risk and neural net close decisions
                        self.check_position_limits(symbol)
                        if self.active_positions.get(symbol):
                            self.evaluate_rl_policy(symbol)
                            
                        # Periodically broadcast agent metrics to visual dashboard (at most once every 500ms)
                        if not hasattr(self, "_last_broadcast_time") or now - self._last_broadcast_time > 0.5:
                            self._last_broadcast_time = now
                            asyncio.create_task(self.broadcast_agent_update())
                            
                    # Broadcast LOB depth to visual client
                    await self.dashboard_server.broadcast_event({
                        "symbol": symbol,
                        "type": "DEPTH",
                        "bids": book.get_sorted_bids(10),
                        "asks": book.get_sorted_asks(10)
                    })
                    
                    # Track Spoofing
                    is_spoof, msg = book.detect_depth_spoofing()
                    if is_spoof:
                        self.trigger_alert({
                            "asset": symbol,
                            "signal_type": "LIQUIDITY_SPOOFING",
                            "confidence": 0.65,
                            "evidence": msg,
                            "entry_zone": "Monitor surrounding depth for consolidation",
                            "exit_zone": "N/A"
                        })

                # --- 2. TICK EXECUTION EVENT PROCESSING ---
                elif event["type"] == "TICK":
                    price = event["price"]
                    volume = event["volume"]
                    side = event["side"]
                    ts_ns = event["timestamp_ns"]

                    # Update Footprint volume counts
                    footprint.add_tick(price, volume, side, ts_ns)
                    active_bar = footprint.get_active_bar()
                    
                    if active_bar:
                        metrics.update_cvd(active_bar)
                        metrics.update_advanced_metrics(active_bar, int(ts_ns / 1_000_000))
                        
                        # Broadcast Footprint grid updates to the visual dashboard client
                        await self.dashboard_server.broadcast_event({
                            "symbol": symbol,
                            "type": "FOOTPRINT",
                            "volume_at_price": active_bar.volume_at_price,
                            "total_delta": active_bar.total_delta
                        })
                        
                        # Check for Absorption Heuristic (Iceberg defend levels)
                        abs_signal = self.absorption_analyser.analyze(active_bar, book)
                        if abs_signal:
                            best_bid, _ = book.get_best_bid()
                            best_ask, _ = book.get_best_ask()
                            
                            self.trigger_alert({
                                "asset": symbol,
                                "signal_type": abs_signal["type"],
                                "confidence": abs_signal["confidence"],
                                "evidence": abs_signal["evidence"],
                                "entry_zone": f"{price:.2f} (Defended Zone)",
                                "exit_zone": f"Stop loss outside {price:.2f} | Target opposite depth ({best_bid:.2f}/{best_ask:.2f})"
                            })

                        # Check for Diagonal Volume Imbalances
                        imbalances = metrics.calculate_diagonal_imbalances(active_bar)
                        
                        # Alert on severe buying imbalances
                        for imb in imbalances["buying"]:
                            if imb["ratio"] > settings.VOLUME_IMBALANCE_RATIO * 1.5:
                                self.trigger_alert({
                                    "asset": symbol,
                                    "signal_type": "AGGRESSIVE_BUYING_IMBALANCE",
                                    "confidence": 0.80,
                                    "evidence": f"Diagonal buying imbalance of {imb['ratio']:.2f}x (Ask Vol {imb['ask_vol']:.2f} vs Bid Vol {imb['bid_vol']:.2f}) at price {imb['price']:.2f}",
                                    "entry_zone": f"{imb['price']:.2f} - {imb['price'] + footprint.bin_size:.2f}",
                                    "exit_zone": f"Stop loss below {imb['price'] - footprint.bin_size:.2f}"
                                })

                        # Alert on severe selling imbalances
                        for imb in imbalances["selling"]:
                            if imb["ratio"] > settings.VOLUME_IMBALANCE_RATIO * 1.5:
                                self.trigger_alert({
                                    "asset": symbol,
                                    "signal_type": "AGGRESSIVE_SELLING_IMBALANCE",
                                    "confidence": 0.80,
                                    "evidence": f"Diagonal selling imbalance of {imb['ratio']:.2f}x (Bid Vol {imb['bid_vol']:.2f} vs Ask Vol {imb['ask_vol']:.2f}) at price {imb['price']:.2f}",
                                    "entry_zone": f"{imb['price']:.2f} - {imb['price'] - footprint.bin_size:.2f}",
                                    "exit_zone": f"Stop loss above {imb['price'] + footprint.bin_size:.2f}"
                                })

                    # Check for RL entries on this fresh tick (alerts might have updated signal triggers)
                    if not self.active_positions.get(symbol):
                        self.evaluate_rl_policy(symbol)

                # --- 3. AUTO-CACHE TRIGGER & COMPLETION METRICS ---
                history = footprint.get_history()
                
                if len(history) > self.bar_counts[symbol]:
                    self.bar_counts[symbol] = len(history)
                    logger.info(f"Bar rollover detected for {symbol}. Saving footprint state to cache database.")
                    self.cache_manager.save_footprint_history(symbol, history)

                if len(history) >= 3:
                    # Run Price-Delta Divergence Check
                    divergence_msg = metrics.check_delta_divergence(history)
                    if divergence_msg:
                        self.trigger_alert({
                            "asset": symbol,
                            "signal_type": "PRICE_DELTA_DIVERGENCE",
                            "confidence": 0.70,
                            "evidence": divergence_msg,
                            "entry_zone": "Await next bar confirmation before entering",
                            "exit_zone": "N/A"
                        })
                    
                    # Run Trend Exhaustion check
                    exh_signal = self.exhaustion_analyser.analyze(history)
                    if exh_signal:
                        self.trigger_alert({
                            "asset": symbol,
                            "signal_type": exh_signal["type"],
                            "confidence": exh_signal["confidence"],
                            "evidence": exh_signal["evidence"],
                            "entry_zone": f"{exh_signal['price']:.2f}",
                            "exit_zone": "Tight stop outside exhaustion boundary"
                        })

                self.event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(0.1)

    def trigger_alert(self, signal_payload: dict) -> None:
        """
        Formats and prints trading alerts in structured JSON.
        Broadcasts the alert instantly to all active dashboard frontend client interfaces.
        """
        signal_payload["timestamp_ms"] = int(time.time() * 1000)
        alert_json = json.dumps(signal_payload, indent=2)
        logger.info(f"\n[ALERT] ACTIONABLE TRADING ALERT DETECTED:\n{alert_json}\n")
        
        # Track signal event locally for Deep RL State compiled inputs (symbol-specific)
        now = time.time()
        sig_type = signal_payload.get("signal_type", "")
        evidence = signal_payload.get("evidence", "").lower()
        sym = signal_payload.get("asset", "BTCUSD")
        
        if sym in self.signal_events:
            events = self.signal_events[sym]
            if sig_type == "LIQUIDITY_SPOOFING":
                if "bid" in evidence:
                    events["spoofing_bid"] = now
                elif "ask" in evidence:
                    events["spoofing_ask"] = now
            elif sig_type == "ABSORPTION_BUY":
                events["absorption_buy"] = now
            elif sig_type == "ABSORPTION_SELL":
                events["absorption_sell"] = now
            elif sig_type == "EXHAUSTION_BUY":
                events["exhaustion_buy"] = now
            elif sig_type == "EXHAUSTION_SELL":
                events["exhaustion_sell"] = now
            elif sig_type == "AGGRESSIVE_BUYING_IMBALANCE":
                events["buying_imbalance"] = now
            elif sig_type == "AGGRESSIVE_SELLING_IMBALANCE":
                events["selling_imbalance"] = now
            
        # Broadcast signal to visual dashboard
        asyncio.create_task(self.dashboard_server.broadcast_event({
            "alert": signal_payload
        }))

    def load_history(self):
        """Loads previous trade history and resumes outstanding active positions from cache."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.trades_history = json.load(f)
                
                closed_trades = [t for t in self.trades_history if t["status"] == "CLOSED"]
                self.wins = sum(1 for t in closed_trades if t["pnl"] > 0)
                self.losses = sum(1 for t in closed_trades if t["pnl"] <= 0)
                self.session_pnl = sum(t["pnl"] for t in closed_trades)
                
                active_trades = [t for t in self.trades_history if t["status"] == "ACTIVE"]
                for pt in active_trades:
                    sym = pt.get("asset", "BTCUSD")
                    self.active_positions[sym] = pt
                    logger.info(f"[RL BRAIN] Resumed active {pt['direction']} paper trade on {sym} from cache.")
                    
                logger.info(f"[RL BRAIN] Warm history loaded. Session PnL: {self.session_pnl:+.2f} | Wins: {self.wins} | Losses: {self.losses}")
            except Exception as e:
                logger.error(f"Failed to load paper trade history: {str(e)}")

    def save_history(self):
        """Saves current paper trade history to database cache."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.trades_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper trade history: {str(e)}")

    def compile_state_vector(self, symbol: str) -> list:
        """Compiles standard real-time 36-dimensional feature vector of current market and position state."""
        state = [0.0] * 36
        now = time.time()
        
        book = self.books.get(symbol)
        footprint = self.footprints.get(symbol)
        metrics = self.metrics_calcs.get(symbol)
        
        if not book or not footprint or not metrics:
            return state
            
        bids = book.get_sorted_bids(10)
        asks = book.get_sorted_asks(10)
        last_price = self.last_prices.get(symbol, 0.0)
        
        # 1-5. Bid depth ratio at Level 1, 2, 3, 5, 10
        for i, idx in enumerate([0, 1, 2, 4, 9]):
            if len(bids) > idx and len(asks) > idx:
                bid_vol = sum(bids[j][1] for j in range(idx + 1))
                ask_vol = sum(asks[j][1] for j in range(idx + 1))
                state[i] = bid_vol / (bid_vol + ask_vol + 1e-6)
                
        # 6-10. Ask depth ratio at Level 1, 2, 3, 5, 10
        for i, idx in enumerate([0, 1, 2, 4, 9]):
            if len(bids) > idx and len(asks) > idx:
                bid_vol = sum(bids[j][1] for j in range(idx + 1))
                ask_vol = sum(asks[j][1] for j in range(idx + 1))
                state[5 + i] = ask_vol / (bid_vol + ask_vol + 1e-6)
                
        # 11. Spread
        best_bid, _ = book.get_best_bid()
        best_ask, _ = book.get_best_ask()
        spread = max(0.0, best_ask - best_bid)
        state[10] = min(1.0, spread / 10.0)
        
        # 12-13. Active footprint bar metrics
        active_bar = footprint.get_active_bar()
        if active_bar:
            active_bar_vol = sum(v["bid_vol"] + v["ask_vol"] for v in active_bar.volume_at_price.values())
            active_bar_delta = active_bar.total_delta
            state[11] = min(5.0, active_bar_vol / 5000.0)
            state[12] = max(-5.0, min(5.0, active_bar_delta / 5000.0))
            
        # 14. CVD
        state[13] = max(-10.0, min(10.0, metrics.cvd_value / 100000.0))
        
        # 15. CVD trend (change over last 10 seconds)
        cvd_change = 0.0
        past_cvds = [c for t, c in self.cvd_histories.get(symbol, []) if now - t > 10.0]
        if past_cvds:
            cvd_change = metrics.cvd_value - past_cvds[-1]
        state[14] = max(-5.0, min(5.0, cvd_change / 20000.0))
        
        # 16-23. Signals with decay
        decay_period = 30.0
        events = self.signal_events.get(symbol, {})
        for idx, key in enumerate([
            "absorption_buy", "absorption_sell", 
            "exhaustion_buy", "exhaustion_sell",
            "buying_imbalance", "selling_imbalance", 
            "spoofing_bid", "spoofing_ask"
        ]):
            trigger_time = events.get(key, 0.0)
            elapsed = now - trigger_time
            if elapsed < decay_period:
                state[15 + idx] = 1.0 - (elapsed / decay_period)
                
        # 24-26. Distance to profile nodes
        history = footprint.get_history()
        poc, hvn, lvn = 0.0, 0.0, 0.0
        if history:
            profile = self.profile_analyser.compute_profile(history)
            poc = profile.get("poc") or 0.0
            hvns = profile.get("hvns") or []
            hvn = hvns[0] if hvns else 0.0
            lvns = profile.get("lvns") or []
            lvn = lvns[0] if lvns else 0.0
            
        state[23] = min(5.0, abs(last_price - poc) / 50.0) if poc > 0 else 0.0
        state[24] = min(5.0, abs(last_price - hvn) / 50.0) if hvn > 0 else 0.0
        state[25] = min(5.0, abs(last_price - lvn) / 50.0) if lvn > 0 else 0.0
        
        # 27-30. Price momentum
        for i, elapsed_sec in enumerate([1.0, 3.0, 5.0, 10.0]):
            past_prices = [p for t, p in self.price_histories.get(symbol, []) if now - t > elapsed_sec]
            momentum = 0.0
            if past_prices and last_price > 0:
                momentum = (last_price - past_prices[-1]) / 10.0
            state[26 + i] = max(-5.0, min(5.0, momentum))
            
        # 31. Active position
        pos = self.active_positions.get(symbol)
        if pos:
            state[30] = 1.0 if pos["direction"] == "LONG" else -1.0
            state[31] = max(-3.0, min(3.0, pos["pnl"] / 30.0))
            duration = now - pos["entry_time"]
            state[32] = min(2.0, duration / 300.0)
        else:
            state[30] = 0.0
            state[31] = 0.0
            state[32] = 0.0
            
        # 34. Session PnL
        state[33] = max(-10.0, min(10.0, self.session_pnl / 100.0))
        
        # 35-36. Institutional Advanced Metrics (VPIN & Hawkes Process)
        state[34] = max(0.0, min(1.0, metrics.vpin))
        state[35] = max(0.0, min(10.0, metrics.hawkes_intensity / 100.0))
        
        # --- INPUT VECTOR STATE SMOOTHING (NOISE REDUCTION) ---
        if not hasattr(self, "state_smoothing_buffers"):
            self.state_smoothing_buffers = {}
            
        if symbol not in self.state_smoothing_buffers:
            from collections import deque
            # 25 samples = ~5 seconds assuming 5 TPS evaluation rate
            self.state_smoothing_buffers[symbol] = deque(maxlen=25)
            
        self.state_smoothing_buffers[symbol].append(state)
        
        buffer_len = len(self.state_smoothing_buffers[symbol])
        smoothed_state = list(state)
        
        # Smooth Order Book Imbalance (0-10), CVD (13-14), and Footprint Imbalances (15-22)
        for i in range(11):
            smoothed_state[i] = sum(s[i] for s in self.state_smoothing_buffers[symbol]) / buffer_len
        for i in range(13, 15):
            smoothed_state[i] = sum(s[i] for s in self.state_smoothing_buffers[symbol]) / buffer_len
        for i in range(15, 23):
            smoothed_state[i] = sum(s[i] for s in self.state_smoothing_buffers[symbol]) / buffer_len
        
        return smoothed_state

    def check_position_limits(self, symbol: str) -> None:
        """Monitors active trade ticks against dynamic Take-Profit and Stop-Loss limits."""
        pos = self.active_positions.get(symbol)
        if not pos:
            return
            
        last_price = self.last_prices.get(symbol, 0.0)
        if last_price <= 0:
            return
            
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
            
            # Place direct reduce_only close trade on Delta Exchange Demo
            try:
                prod_id_map = {"BTCUSD": 84, "ETHUSD": 1699, "SOLUSD": 92572, "XRPUSD": 93723}
                p_id = prod_id_map.get(symbol)
                if p_id:
                    exit_side = "sell" if direction == "LONG" else "buy"
                    logger.info(f"[EXCHANGE BRACKET CLOSE] Sending reduce_only {exit_side.upper()} order for {symbol} (Product ID: {p_id}) to Delta Demo...")
                    order_res = self.rest_client.place_order(product_id=p_id, size=1, side=exit_side, order_type="market_order", reduce_only="true")
                    logger.info(f"[EXCHANGE BRACKET CLOSE SUCCESS] Order filled! Details: {order_res}")
            except Exception as ex:
                logger.error(f"[EXCHANGE BRACKET CLOSE ERROR] Failed to close position on Delta Demo: {str(ex)}")

            outcome = "WIN (Take Profit Hit)" if pnl > 0 else "LOSS (Stop Loss Hit)"
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
                
            self.session_pnl += pnl
            self.active_positions[symbol] = None
            self.save_history()
            
            logger.info(
                f"\n[RL BRACKET EXIT] Position closed via Bracket Limits on {symbol}: {direction} | "
                f"PnL: {pnl:+.4f} | Outcome: {outcome} | Total Session PnL: {self.session_pnl:+.4f}\n"
            )
            
            # Send TD backpropagation feedback to agent neural net
            next_state = self.compile_state_vector(symbol)
            try:
                self.rl_request_queue.put_nowait({
                    "type": "REWARD",
                    "symbol": symbol,
                    "reward": pnl
                })
            except:
                pass
            
            # Broadcast update
            asyncio.create_task(self.broadcast_agent_update())

    def trigger_kill_switch(self):
        """Manually locks the agent and closes all positions. Hooked to the Dashboard."""
        if not self.agent_locked:
            logger.critical(f"MANUAL KILL-SWITCH ENGAGED from Dashboard. Locking agent.")
            self.agent_locked = True
            for sym, pos in self.active_positions.items():
                if pos:
                    pnl = pos["pnl"]
                    pos["status"] = "CLOSED"
                    pos["exit_price"] = self.last_prices.get(sym, pos["entry_price"])
                    pos["exit_time"] = time.time()
                    
                    # Close position directly on Delta Exchange Demo
                    try:
                        prod_id_map = {"BTCUSD": 84, "ETHUSD": 1699, "SOLUSD": 92572, "XRPUSD": 93723}
                        p_id = prod_id_map.get(sym)
                        if p_id:
                            exit_side = "sell" if pos["direction"] == "LONG" else "buy"
                            logger.critical(f"[EXCHANGE EMERGENCY CLOSE] Closing {sym} with reduce_only {exit_side.upper()} order...")
                            self.rest_client.place_order(product_id=p_id, size=1, side=exit_side, order_type="market_order", reduce_only="true")
                    except Exception as ex:
                        logger.error(f"[EXCHANGE EMERGENCY CLOSE ERROR] Failed to emergency close on Delta Demo: {str(ex)}")

                    self.session_pnl += pnl
                    self.active_positions[sym] = None
                    logger.critical(f"EMERGENCY CLOSE: {sym} {pos['direction']} at {pos['exit_price']} | PnL: {pnl}")
            self.save_history()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast_agent_update())
            except RuntimeError:
                pass

    def evaluate_rl_policy(self, symbol: str) -> None:
        """Queries the active neural network policy stochastically to select and execute trades."""
        if self.agent_locked or self.agent_paused:
            return
            
        # Equity Hard-Stop: If session losses exceed 2% of total capital
        if self.session_pnl <= - (self.starting_capital * 0.02):
            if not self.agent_locked:
                logger.critical(f"KILL-SWITCH ENGAGED: Max daily drawdown exceeded 2% of ${self.starting_capital}. Locking agent.")
                self.agent_locked = True
                # Send Shutdown to RL Process
                if getattr(self, "rl_process", None) is not None:
                    self.rl_request_queue.put({"type": "SHUTDOWN"})
                # Market close all open positions
                for sym, pos in self.active_positions.items():
                    if pos:
                        pnl = pos["pnl"]
                        pos["status"] = "CLOSED"
                        pos["exit_price"] = self.last_prices.get(sym, pos["entry_price"])
                        pos["exit_time"] = time.time()
                        self.session_pnl += pnl
                        self.active_positions[sym] = None
                        logger.critical(f"EMERGENCY CLOSE: {sym} {pos['direction']} at {pos['exit_price']} | PnL: {pnl}")
                self.save_history()
                asyncio.create_task(self.broadcast_agent_update())
            return
            
        last_price = self.last_prices.get(symbol, 0.0)
        if last_price <= 0:
            return
            
        now = time.time()
        active_pos = self.active_positions.get(symbol)
        
        state = self.compile_state_vector(symbol)
        
        institutional_signal_active = any(state[i] > 0.0 for i in range(15, 23))
        
        # Request inference from RL Process
        self.rl_request_queue.put_nowait({
            "type": "EVALUATE",
            "symbol": symbol,
            "state": state,
            "last_price": last_price,
            "inst_sig": institutional_signal_active
        })
        
        # Note: Logic assumes asynchronous processing of inference results via self.rl_response_queue
    def execute_action(self, symbol: str, action_idx: int, action_label: str, confidence: float, probs: dict) -> None:
        """Callback from the RL Process. Executes the actual order."""
        now = time.time()
        active_pos = self.active_positions.get(symbol)
        
        # Signal Hysteresis & State Debouncing (REMOVED for absolute active trading frequency)
        # last_entry = self.last_entry_times.get(symbol, 0.0)
        # if not active_pos and (now - last_entry < 180.0):
        #     if action_label in ["BUY", "SELL"] or action_idx in [1, 2]:
        #         logger.info(f"[HYSTERESIS LOCK ACTIVE] Signal Debouncing blocked {action_label} on {symbol}. Agent on cooldown.")
        #         action_idx = 0
        #         action_label = "HOLD"
                
        # Save raw softmax policy probabilities in-memory for live neural network monitoring
        self.policy_probabilities = getattr(self, "policy_probabilities", {})
        self.policy_probabilities[symbol] = probs
        
        last_price = self.last_prices.get(symbol, 0.0)
        
        # 1. CLOSE POSITION
        if active_pos:
            if action_label == "CLOSE" or action_idx == 3:
                # Prevent stochastic premature closes; require at least 30 seconds hold time (REMOVED for absolute active trading frequency)
                # if now - active_pos["entry_time"] < 30.0:
                #     return
                direction = active_pos["direction"]
                pnl = active_pos["pnl"]
                
                active_pos["status"] = "CLOSED"
                active_pos["exit_price"] = last_price
                active_pos["exit_time"] = now
                
                # Place direct close order on Delta Exchange Demo
                try:
                    prod_id_map = {"BTCUSD": 84, "ETHUSD": 1699, "SOLUSD": 92572, "XRPUSD": 93723}
                    p_id = prod_id_map.get(symbol)
                    if p_id:
                        exit_side = "sell" if direction == "LONG" else "buy"
                        logger.info(f"[EXCHANGE CLOSE ORDER] Sending close reduce_only {exit_side.upper()} order for {symbol} to Delta Demo...")
                        order_res = self.rest_client.place_order(product_id=p_id, size=1, side=exit_side, order_type="market_order", reduce_only="true")
                        logger.info(f"[EXCHANGE CLOSE SUCCESS] Filled exit order: {order_res}")
                except Exception as ex:
                    logger.error(f"[EXCHANGE CLOSE ERROR] Failed to place close order on Delta Demo: {str(ex)}")

                if pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                    
                self.session_pnl += pnl
                self.active_positions[symbol] = None
                self.save_history()
                
                logger.info(
                    f"\\n[RL POLICY EXIT] CLOSE order triggered by neural network on {symbol}: {direction} | "
                    f"PnL: {pnl:+.4f} | Total Session PnL: {self.session_pnl:+.4f}\\n"
                )
                
                # Send reward back to RL Agent Process
                try:
                    self.rl_request_queue.put_nowait({
                        "type": "REWARD",
                        "symbol": symbol,
                        "reward": pnl
                    })
                except:
                    pass
                
                # Broadcast update
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.broadcast_agent_update())
                except RuntimeError:
                    pass
            elif action_label == "HOLD" or action_idx == 0:
                pass # HOLD active pos
            return
            
        # Widen dynamic TP and SL targets to prevent getting stopped out by noise
        tp_offset = last_price * 0.0020  # 0.20%
        sl_offset = last_price * 0.0010  # 0.10%
        
        # 2. LONG ENTRY
        if (action_label == "BUY" or action_idx == 1) and active_pos is None:
            tp = last_price + tp_offset
            sl = last_price - sl_offset
            
            # Place direct market LONG entry order on Delta Exchange Demo
            try:
                prod_id_map = {"BTCUSD": 84, "ETHUSD": 1699, "SOLUSD": 92572, "XRPUSD": 93723}
                p_id = prod_id_map.get(symbol)
                if p_id:
                    logger.info(f"[EXCHANGE ENTRY ORDER] Sending LONG market order for {symbol} (Product ID: {p_id}) to Delta Demo...")
                    order_res = self.rest_client.place_order(product_id=p_id, size=1, side="buy", order_type="market_order")
                    logger.info(f"[EXCHANGE ENTRY SUCCESS] Filled entry: {order_res}")
                    actual_fill = float(order_res.get("average_fill_price") or last_price)
            except Exception as ex:
                logger.error(f"[EXCHANGE ENTRY ERROR] Failed to place LONG order on Delta Demo: {str(ex)}")

            new_pos = {
                "asset": symbol,
                "direction": "LONG",
                "entry_price": actual_fill,
                "predicted_price": last_price,
                "slippage_pct": slippage_pct,
                "exec_mode": "MARKET",
                "tp": tp,
                "sl": sl,
                "entry_time": now,
                "trigger": f"Deep RL BUY (confidence {confidence:.2f})",
                "status": "ACTIVE",
                "pnl": 0.0,
                "exit_price": None
            }
            self.active_positions[symbol] = new_pos
            self.last_entry_times[symbol] = now
            self.trades_history.append(new_pos)
            self.save_history()
            logger.info(f"\\n[RL POLICY ENTRY] LONG Position Entered on {symbol} | Entry: {last_price:.4f} | TP: {tp:.4f} | SL: {sl:.4f} | Confidence: {confidence:.2f}\\n")
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast_agent_update())
            except RuntimeError:
                pass
                
        # 3. SHORT ENTRY
        elif (action_label == "SELL" or action_idx == 2) and active_pos is None:
            tp = last_price - tp_offset
            sl = last_price + sl_offset
            
            # Place direct market SHORT entry order on Delta Exchange Demo
            try:
                prod_id_map = {"BTCUSD": 84, "ETHUSD": 1699, "SOLUSD": 92572, "XRPUSD": 93723}
                p_id = prod_id_map.get(symbol)
                if p_id:
                    logger.info(f"[EXCHANGE ENTRY ORDER] Sending SHORT market order for {symbol} (Product ID: {p_id}) to Delta Demo...")
                    order_res = self.rest_client.place_order(product_id=p_id, size=1, side="sell", order_type="market_order")
                    logger.info(f"[EXCHANGE ENTRY SUCCESS] Filled entry: {order_res}")
                    actual_fill = float(order_res.get("average_fill_price") or last_price)
            except Exception as ex:
                logger.error(f"[EXCHANGE ENTRY ERROR] Failed to place SHORT order on Delta Demo: {str(ex)}")

            new_pos = {
                "asset": symbol,
                "direction": "SHORT",
                "entry_price": actual_fill,
                "predicted_price": last_price,
                "slippage_pct": slippage_pct,
                "exec_mode": "MARKET",
                "tp": tp,
                "sl": sl,
                "entry_time": now,
                "trigger": f"Deep RL SELL (confidence {confidence:.2f})",
                "status": "ACTIVE",
                "pnl": 0.0,
                "exit_price": None
            }
            self.active_positions[symbol] = new_pos
            self.last_entry_times[symbol] = now
            self.trades_history.append(new_pos)
            self.save_history()
            logger.info(f"\\n[RL POLICY ENTRY] SHORT Position Entered on {symbol} | Entry: {last_price:.4f} | TP: {tp:.4f} | SL: {sl:.4f} | Confidence: {confidence:.2f}\\n")
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast_agent_update())
            except RuntimeError:
                pass

    async def broadcast_agent_update(self) -> None:
        """Pushes comprehensive real-time statistics and trades list of the Deep RL agent over SSE."""
        try:
            # Dummy metrics since agent is isolated
            brain_metrics = {
                "trades": self.wins + self.losses,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": (self.wins / max(1, self.wins + self.losses)) * 100,
                "pnl": self.session_pnl,
                "epsilon": 0.0,
                "harness": "Multiprocess Isolation Active",
                "buffer_size": 100000
            }
            
            # Format all active positions
            active_list = {}
            for sym, pos in self.active_positions.items():
                if pos:
                    active_list[sym] = {
                        "direction": pos["direction"],
                        "entry_price": pos["entry_price"],
                        "tp": pos["tp"],
                        "sl": pos["sl"],
                        "pnl": pos["pnl"],
                        "entry_time": pos["entry_time"],
                        "asset": pos.get("asset", sym)
                    }
                    
            # Select last 5 completed trades for visual logs
            completed = [t for t in self.trades_history if t["status"] == "CLOSED"]
            recent = []
            for t in reversed(completed[-5:]):
                recent.append({
                    "direction": t["direction"],
                    "entry_price": t["entry_price"],
                    "predicted_price": t.get("predicted_price", t["entry_price"]),
                    "slippage_pct": t.get("slippage_pct", 0.0),
                    "exit_price": t.get("exit_price", 0.0),
                    "pnl": t["pnl"],
                    "trigger": t["trigger"],
                    "asset": t.get("asset", "BTCUSD")
                })
                
            total_trades = len(completed)
            win_rate = (self.wins / total_trades * 100.0) if total_trades > 0 else 0.0
            
            # Build Multi-Asset Scanner Payload
            scanner_state = {}
            symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "AVAXUSD"]
            for sym in symbols:
                lp = self.last_prices.get(sym, 0.0)
                
                # Fetch footprint metrics
                footprint = self.footprints.get(sym)
                bar_delta = 0.0
                if footprint:
                    active_bar = footprint.get_active_bar()
                    if active_bar:
                        bar_delta = active_bar.total_delta
                        
                metrics = self.metrics_calcs.get(sym)
                cvd_val = metrics.cvd_value if metrics else 0.0
                
                # Determine neural recommendation bias
                bias = "HOLD"
                probs = {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0, "CLOSE": 0.0}
                if sym in self.books:
                    if hasattr(self, "policy_probabilities") and sym in self.policy_probabilities:
                        probs = self.policy_probabilities[sym]
                        
                        # Find highest prob to set bias
                        max_act = max(probs.items(), key=lambda k: k[1])
                        bias = max_act[0]
                        
                scanner_state[sym] = {
                    "price": lp,
                    "delta": bar_delta,
                    "cvd": cvd_val,
                    "bias": bias,
                    "policy_probs": probs
                }
                
            await self.dashboard_server.broadcast_event({
                "type": "AGENT_UPDATE",
                "active_positions": active_list,
                "session_pnl": self.session_pnl,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "brain_harness": brain_metrics["harness"],
                "buffer_size": brain_metrics["buffer_size"],
                "recent_trades": recent,
                "scanner": scanner_state
            })
        except Exception as e:
            import traceback
            logger.error(f"Error in broadcast_agent_update: {str(e)}\n{traceback.format_exc()}")

    async def stop(self) -> None:
        """Stops orchestrator, connectors, dashboard server, and cancels pending workers."""
        logger.info("Shutting down Order Flow Engine...")
        self.is_running = False
        
        # Save cache
        for symbol, footprint in self.footprints.items():
            logger.info(f"Saving final footprint cache database for {symbol} before exit...")
            self.cache_manager.save_footprint_history(symbol, footprint.get_history())

        # Signal Multiprocessing RL Brain to shutdown and save checkpoint
        if self.rl_process is not None:
            self.rl_request_queue.put({"type": "SHUTDOWN"})
            self.rl_process.join(timeout=2.0)

        # Stop Connectors
        for conn in self.connectors:
            await conn.stop()
            
        # Stop Dashboard Server
        await self.dashboard_server.stop()
            
        # Cancel Processing Task
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
                
        # Cancel Action Task
        if getattr(self, "action_task", None):
            self.action_task.cancel()
            try:
                await self.action_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Order Flow Engine completely stopped.")

# Async helper to run the master orchestrator
async def main():
    engine = OrderFlowEngine()
    await engine.start()
    
    loop = asyncio.get_running_loop()
    
    def shutdown():
        logger.info("Exit signal received.")
        asyncio.create_task(engine.stop())
        
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    try:
        while engine.is_running:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        await engine.stop()

if __name__ == "__main__":
    mp.freeze_support()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Engine terminated by user.")
