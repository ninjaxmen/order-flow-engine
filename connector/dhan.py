import asyncio
import struct
import time
from typing import Optional, Dict
from connector.base import BaseConnector
from config import settings
from utils.logger import logger

class DhanConnector(BaseConnector):
    """
    WebSocket client for the Dhan API.
    Handles connection, subscription, and parsing of custom binary packets.
    Includes simulated tick flow for offline operation and backtesting.
    """
    def __init__(self, event_queue: asyncio.Queue, simulation_mode: bool = True):
        url = settings.DHAN_FEED_URL.format(
            token=settings.DHAN_ACCESS_TOKEN,
            client_id=settings.DHAN_CLIENT_ID
        )
        super().__init__("DHAN", url, event_queue)
        self.simulation_mode = simulation_mode
        self.websocket = None

    async def connect_and_subscribe(self) -> None:
        """
        Connects to Dhan WebSocket feed. If simulation_mode is enabled, it prepares simulated feed.
        """
        if self.simulation_mode:
            logger.info("DHAN Connector: Simulation mode active. No remote WebSocket connection required.")
            return

        # Live websocket connection code (standard websockets or dhanhq library)
        try:
            import websockets
            self.websocket = await websockets.connect(self.ws_url)
            # Subscribe to standard equity index/futures ids
            # Subscription structure: {"RequestCode": 15, "InstrumentCount": 1, "InstrumentList": [{"ExchangeSegment": "NSE_EQ", "SecurityId": "1333"}]}
            sub_msg = (
                '{"RequestCode": 15, "InstrumentCount": 1, "InstrumentList": '
                '[{"ExchangeSegment": "NSE_EQ", "SecurityId": "1333"}]}'
            )
            await self.websocket.send(sub_msg)
            logger.info("DHAN Connector: Subscribed to Nifty-50 (SecurityID: 1333)")
        except ImportError:
            logger.warning("websockets package not found. Falling back to Simulation Mode for DHAN.")
            self.simulation_mode = True
        except Exception as e:
            logger.error(f"Failed to connect to Dhan live server: {str(e)}. Falling back to simulation.")
            self.simulation_mode = True

    async def listen_loop(self) -> None:
        """
        Main listen loop. Routes binary frames to binary parser or runs mock generator.
        """
        if self.simulation_mode:
            await self._run_simulation()
            return

        try:
            while self.is_running and self.websocket:
                message = await self.websocket.recv()
                if isinstance(message, bytes):
                    self._parse_binary_packet(message)
                else:
                    # Heartbeat / text messages
                    logger.debug(f"DHAN Text Received: {message}")
        except Exception as e:
            logger.error(f"DHAN socket reading error: {str(e)}")
            raise e

    def _parse_binary_packet(self, data: bytes) -> None:
        """
        Decodes binary packets emitted by Dhan API feed.
        
        Binary Structure details:
        - FeedHeader (12 bytes):
            - FeedType (1 byte): 1=Ticker, 2=Quote, 3=Full/Depth
            - Length (2 bytes)
            - SecurityID (4 bytes)
            - ExchangeSegment (1 byte)
            - Timestamp (4 bytes)
        """
        try:
            if len(data) < 12:
                return

            feed_type, length, sec_id, segment_id, timestamp = struct.unpack("<BHI B I", data[:12])
            
            # Extract standard fields
            event = {
                "exchange": "DHAN",
                "symbol": f"DHAN_{sec_id}",
                "timestamp_ns": timestamp * 1_000_000_000, # convert epoch sec to ns
            }

            if feed_type == 1: # Ticker / LTP
                # Ticker payload: Price (4 bytes float), Volume (4 bytes float)
                price, volume = struct.unpack("<ff", data[12:20])
                event.update({
                    "type": "TICK",
                    "price": price,
                    "volume": volume,
                    "side": "BUY" if price % 2 == 0 else "SELL"  # Fallback aggressor determination
                })
                self.enqueue_event(event)

            elif feed_type == 3: # Market Depth (L2)
                # Parse bids and asks
                # Typically, Dhan returns depth with 5 levels of bids and asks.
                # Format: 5 * (BidPrice [4 bytes], BidQty [4 bytes], BidOrders [2 bytes], AskPrice [4 bytes], AskQty [4 bytes], AskOrders [2 bytes])
                offset = 12
                bids = []
                asks = []
                for _ in range(5):
                    if offset + 20 > len(data):
                        break
                    bid_p, bid_q, _, ask_p, ask_q, _ = struct.unpack("<ff H ff H", data[offset:offset+20])
                    if bid_p > 0:
                        bids.append([bid_p, bid_q])
                    if ask_p > 0:
                        asks.append([ask_p, ask_q])
                    offset += 20
                
                event.update({
                    "type": "DEPTH",
                    "bids": bids,
                    "asks": asks
                })
                self.enqueue_event(event)
                
        except Exception as e:
            logger.error(f"DHAN binary decoding failure: {str(e)}")

    async def _run_simulation(self) -> None:
        """
        High-fidelity simulated tick feed for Dhan.
        Simulates ticks around a realistic base price of 22000.0 (representing NIFTY futures)
        and maintains a 5-level Limit Order Book.
        """
        base_price = 22000.0
        sec_id = 1333
        symbol = f"DHAN_{sec_id}"
        
        # Initialize LOB Bids and Asks
        bids = [[base_price - i * 0.5, 100.0 + i * 20.0] for i in range(1, 6)]
        asks = [[base_price + i * 0.5, 120.0 + i * 15.0] for i in range(1, 6)]

        logger.info("DHAN Simulation Loop active.")
        
        import random
        
        while self.is_running:
            try:
                # 1. Periodically send Depth Updates (L2)
                if random.random() < 0.2:
                    # Randomize depth sizes slightly
                    sim_bids = [[p, size * random.uniform(0.8, 1.2)] for p, size in bids]
                    sim_asks = [[p, size * random.uniform(0.8, 1.2)] for p, size in asks]
                    
                    event = {
                        "exchange": "DHAN",
                        "symbol": symbol,
                        "type": "DEPTH",
                        "timestamp_ns": int(time.time() * 1_000_000_000),
                        "bids": sorted(sim_bids, key=lambda x: x[0], reverse=True),
                        "asks": sorted(sim_asks, key=lambda x: x[0])
                    }
                    self.enqueue_event(event)

                # 2. Generate trades (aggressive fills)
                else:
                    # Randomly pick if trade is buy or sell
                    is_buy = random.choice([True, False])
                    # Ticks execute at ask (buy) or bid (sell)
                    trade_price = asks[0][0] if is_buy else bids[0][0]
                    trade_vol = random.uniform(5.0, 80.0)
                    
                    # Larger trades occasionally (iceberg triggers)
                    if random.random() < 0.05:
                        trade_vol = random.uniform(250.0, 700.0)
                        
                    event = {
                        "exchange": "DHAN",
                        "symbol": symbol,
                        "type": "TICK",
                        "timestamp_ns": int(time.time() * 1_000_000_000),
                        "price": trade_price,
                        "volume": trade_vol,
                        "side": "BUY" if is_buy else "SELL"
                    }
                    self.enqueue_event(event)

                    # Walk LOB price slightly
                    price_move = 0.5 if (is_buy and random.random() < 0.45) else (-0.5 if random.random() < 0.45 else 0)
                    if price_move != 0:
                        base_price += price_move
                        bids = [[base_price - i * 0.5, 100.0 + i * 20.0] for i in range(1, 6)]
                        asks = [[base_price + i * 0.5, 120.0 + i * 15.0] for i in range(1, 6)]
                
                # Dynamic sleep to control feed rates
                await asyncio.sleep(1.0 / settings.SIM_TICK_RATE)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DHAN simulation error: {str(e)}")
                await asyncio.sleep(1.0)
