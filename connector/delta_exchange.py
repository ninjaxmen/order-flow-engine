import asyncio
import json
import time
from typing import Optional
from connector.base import BaseConnector
from config import settings
from utils.logger import logger

class DeltaExchangeConnector(BaseConnector):
    """
    WebSocket client for the Delta Exchange API.
    Handles subscription and parsing of both legacy and new compact public feeds.
    """
    def __init__(self, event_queue: asyncio.Queue, simulation_mode: bool = True):
        super().__init__("DELTA", settings.DELTA_FEED_URL, event_queue)
        self.simulation_mode = simulation_mode
        self.websocket = None

    async def connect_and_subscribe(self) -> None:
        """
        Connects to Delta Exchange public socket.
        """
        if self.simulation_mode:
            logger.info("DELTA Connector: Simulation mode active. No remote WebSocket connection required.")
            return

        try:
            import websockets
            self.websocket = await websockets.connect(self.ws_url)
            
            # Subscribe using Delta's new payload layout
            channels = [
                {
                    "name": "trades",
                    "symbols": ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "AVAXUSD"]
                }
            ]
            for sym in ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "AVAXUSD"]:
                channels.append({
                    "name": "ob_l2",
                    "symbols": [sym]
                })

            sub_msg = {
                "type": "subscribe",
                "payload": {
                    "channels": channels
                }
            }
            await self.websocket.send(json.dumps(sub_msg))
            logger.info("DELTA Connector: Sent new compact subscription payload for BTCUSD, ETHUSD, SOLUSD, XRPUSD, AVAXUSD (ob_l2 subscribed per symbol, trades)")
        except ImportError:
            logger.warning("websockets package not found. Falling back to Simulation Mode for DELTA.")
            self.simulation_mode = True
        except Exception as e:
            logger.error(f"Failed to connect to Delta Exchange: {str(e)}. Falling back to simulation.")
            self.simulation_mode = True

    async def listen_loop(self) -> None:
        """
        Listens for incoming messages from Delta Exchange and decodes JSON.
        """
        if self.simulation_mode:
            await self._run_simulation()
            return

        try:
            while self.is_running and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Check for heartbeat and reply
                if isinstance(data, dict) and (data.get("type") == "heartbeat" or data.get("channel") == "heartbeat"):
                    await self.websocket.send(json.dumps({"type": "pong"}))
                    continue
                
                self._parse_json_message(data)
                
        except Exception as e:
            logger.error(f"DELTA socket reading error: {str(e)}")
            raise e

    def _parse_json_message(self, data: dict) -> None:
        """
        Parses JSON messages from the WebSocket.
        Gracefully resolves standard full-length field keys and compact abbreviated keys.
        """
        try:
            channel = data.get("channel") or data.get("type")
            if not channel:
                return
            if "ob_l2" in channel or "orderbook" in channel:
                logger.info(f"[DEBUG_WS] Parsing L2 depth event for {data.get('sy') or data.get('symbol') or 'unknown'}")

            # Abbreviated symbol key in compact feed is 'sy'
            symbol = data.get("symbol") or data.get("sy") or "BTCUSD"
            
            # Abbreviated timestamp key in compact feed is 'ts'
            timestamp_val = data.get("timestamp") or data.get("ts") or int(time.time() * 1_000_000)
            timestamp_ns = int(timestamp_val) * 1000

            # 1. TRADES CHANNEL
            if "trade" in channel:
                payload = data.get("data") or data
                trades_list = payload if isinstance(payload, list) else [payload]
                
                for trade in trades_list:
                    # Compact trade format: p = price, s = size, side = side
                    price_val = trade.get("price") or trade.get("p")
                    size_val = trade.get("size") or trade.get("volume") or trade.get("qty") or trade.get("s")
                    if price_val is None or size_val is None:
                        continue
                        
                    side = "BUY"
                    side_val = trade.get("side")
                    role = trade.get("buyer_role") or trade.get("r")
                    if side_val:
                        side = side_val.upper()
                    elif role == "taker" or role == "t":
                        side = "BUY"
                    elif role == "maker" or role == "m":
                        side = "SELL"
                        
                    event = {
                        "exchange": "DELTA",
                        "symbol": symbol,
                        "type": "TICK",
                        "timestamp_ns": timestamp_ns,
                        "price": float(price_val),
                        "volume": float(size_val),
                        "side": side
                    }
                    self.enqueue_event(event)

            # 2. LIMIT ORDER BOOK L2 CHANNEL
            elif "ob_l2" in channel or "orderbook" in channel:
                payload = data.get("data") or data
                
                # Compact order book format: b = bids, a = asks
                bids_raw = payload.get("bids") or payload.get("b") or []
                asks_raw = payload.get("asks") or payload.get("a") or []
                
                bids = [[float(level[0]), float(level[1])] for level in bids_raw]
                asks = [[float(level[0]), float(level[1])] for level in asks_raw]
                
                event = {
                    "exchange": "DELTA",
                    "symbol": symbol,
                    "type": "DEPTH",
                    "timestamp_ns": timestamp_ns,
                    "bids": bids,
                    "asks": asks
                }
                self.enqueue_event(event)

        except Exception as e:
            logger.error(f"DELTA JSON parsing failure: {str(e)}")

    async def _run_simulation(self) -> None:
        """
        Simulated high-frequency feed for multiple Delta Exchange assets.
        """
        symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "AVAXUSD"]
        base_prices = {
            "BTCUSD": 76600.0,
            "ETHUSD": 2100.0,
            "SOLUSD": 145.0,
            "XRPUSD": 0.52,
            "AVAXUSD": 35.5
        }
        tick_sizes = {
            "BTCUSD": 0.5,
            "ETHUSD": 0.05,
            "SOLUSD": 0.01,
            "XRPUSD": 0.0001,
            "AVAXUSD": 0.005
        }
        
        # Build initial order books
        books = {}
        for s in symbols:
            bp = base_prices[s]
            ts = tick_sizes[s]
            books[s] = {
                "bids": [[bp - i * ts, 100.0 + i * 20.0] for i in range(1, 6)],
                "asks": [[bp + i * ts, 110.0 + i * 18.0] for i in range(1, 6)]
            }

        logger.info("DELTA Multi-Asset Simulation Loop active.")
        import random
        
        while self.is_running:
            try:
                # Pick a random symbol to tick
                symbol = random.choice(symbols)
                bp = base_prices[symbol]
                ts = tick_sizes[symbol]
                book = books[symbol]
                
                if random.random() < 0.2:
                    sim_bids = [[p, size * random.uniform(0.7, 1.3)] for p, size in book["bids"]]
                    sim_asks = [[p, size * random.uniform(0.7, 1.3)] for p, size in book["asks"]]
                    
                    event = {
                        "exchange": "DELTA",
                        "symbol": symbol,
                        "type": "DEPTH",
                        "timestamp_ns": int(time.time() * 1_000_000_000),
                        "bids": sorted(sim_bids, key=lambda x: x[0], reverse=True),
                        "asks": sorted(sim_asks, key=lambda x: x[0])
                    }
                    self.enqueue_event(event)
                else:
                    is_buy = random.choice([True, False])
                    trade_price = book["asks"][0][0] if is_buy else book["bids"][0][0]
                    trade_vol = random.uniform(0.1, 5.0) if "BTC" in symbol else random.uniform(1.0, 50.0)
                    if random.random() < 0.05:
                        trade_vol *= 10.0  # Spike
                        
                    event = {
                        "exchange": "DELTA",
                        "symbol": symbol,
                        "type": "TICK",
                        "timestamp_ns": int(time.time() * 1_000_000_000),
                        "price": trade_price,
                        "volume": trade_vol,
                        "side": "BUY" if is_buy else "SELL"
                    }
                    self.enqueue_event(event)

                    price_move = ts if (is_buy and random.random() < 0.45) else (-ts if random.random() < 0.45 else 0)
                    if price_move != 0:
                        base_prices[symbol] += price_move
                        bp = base_prices[symbol]
                        book["bids"] = [[bp - i * ts, 100.0 + i * 20.0] for i in range(1, 6)]
                        book["asks"] = [[bp + i * ts, 110.0 + i * 18.0] for i in range(1, 6)]
                
                await asyncio.sleep(1.0 / settings.SIM_TICK_RATE)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DELTA simulation error: {str(e)}")
                await asyncio.sleep(1.0)
