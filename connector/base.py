import asyncio
import abc
import time
from typing import Optional
from utils.logger import logger

class BaseConnector(abc.ABC):
    """
    Abstract Base class for all exchange WebSocket connections.
    Includes automated reconnection loops, heartbeat monitoring, and queue-based backpressure management.
    """
    def __init__(self, name: str, ws_url: str, event_queue: asyncio.Queue):
        self.name = name
        self.ws_url = ws_url
        self.event_queue = event_queue
        self.is_running = False
        self.conn_task: Optional[asyncio.Task] = None
        self.retry_count = 0
        self.max_backoff = 30.0  # seconds
        self.initial_backoff = 1.0  # seconds

    @abc.abstractmethod
    async def connect_and_subscribe(self) -> None:
        """
        Establishes the WebSocket connection and sends subscribe commands.
        Must be implemented by concrete classes.
        """
        pass

    @abc.abstractmethod
    async def listen_loop(self) -> None:
        """
        Listens for incoming messages from the socket and enqueues them.
        Must be implemented by concrete classes.
        """
        pass

    async def run(self) -> None:
        """
        Starts the connector execution and monitors the connection.
        """
        self.is_running = True
        self.conn_task = asyncio.create_task(self.reconnect_loop())
        logger.info(f"{self.name} connector started.")

    async def reconnect_loop(self) -> None:
        """
        Maintains connection state. Reconnects with exponential backoff on failure.
        """
        while self.is_running:
            try:
                logger.info(f"Connecting to {self.name} at {self.ws_url}...")
                await self.connect_and_subscribe()
                self.retry_count = 0  # reset on successful connection
                
                # Listen to incoming messages until connection breaks
                await self.listen_loop()
                
            except asyncio.CancelledError:
                logger.info(f"{self.name} connection loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in {self.name} connection: {str(e)}")
                
            if not self.is_running:
                break
                
            self.retry_count += 1
            backoff_sec = min(self.max_backoff, self.initial_backoff * (1.5 ** self.retry_count))
            logger.warning(f"{self.name} disconnected. Reconnecting in {backoff_sec:.2f} seconds (Attempt {self.retry_count})...")
            await asyncio.sleep(backoff_sec)

    async def stop(self) -> None:
        """
        Terminates the connector loop safely.
        """
        logger.info(f"Stopping {self.name} connector...")
        self.is_running = False
        if self.conn_task:
            self.conn_task.cancel()
            try:
                await self.conn_task
            except asyncio.CancelledError:
                pass
        logger.info(f"{self.name} connector stopped.")

    def enqueue_event(self, event: dict) -> None:
        """
        Pushes a decoded tick or order book update into the central queue.
        Implements logging/backpressure warnings if queue becomes congested.
        """
        if self.event_queue.full():
            logger.warning(f"Central Queue is Full! Backpressure detected. Dropping oldest event.")
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        
        self.event_queue.put_nowait(event)
