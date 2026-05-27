import asyncio
import os
import json
from typing import Set, Tuple
from utils.logger import logger

class OrderFlowDashboardServer:
    """
    Zero-dependency asynchronous HTTP and Server-Sent Events (SSE) server.
    Serves dashboard.html and pushes real-time order book, footprints, and signals to the client.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8080, on_kill_switch=None):
        self.host = host
        self.port = port
        self.on_kill_switch = on_kill_switch
        self.clients: Set[asyncio.StreamWriter] = set()
        self.server: Optional[asyncio.Server] = None
        self.dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")

    async def start(self) -> None:
        """Starts the asynchronous TCP server."""
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f"Visual Quant Dashboard server running at http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stops the server and closes all active client connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        for writer in list(self.clients):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self.clients.clear()
        logger.info("Visual Dashboard server stopped.")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Processes incoming HTTP requests."""
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return

            request_parts = request_line.decode('utf-8').strip().split(" ")
            if len(request_parts) < 2:
                writer.close()
                return

            method, path = request_parts[0], request_parts[1]

            # Read all request headers to clear the buffer
            while True:
                line = await reader.readline()
                if not line or line == b"\r\n":
                    break

            # Handle GET / (Main Dashboard UI)
            if method == "GET" and path == "/":
                if os.path.exists(self.dashboard_path):
                    with open(self.dashboard_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    response = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/html; charset=utf-8\r\n"
                        f"Content-Length: {len(content.encode('utf-8'))}\r\n"
                        "Connection: close\r\n\r\n"
                        f"{content}"
                    )
                    writer.write(response.encode('utf-8'))
                    await writer.drain()
                else:
                    response = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nDashboard HTML not found."
                    writer.write(response.encode('utf-8'))
                    await writer.drain()
                writer.close()

            # Handle GET /stream (Server-Sent Events)
            elif method == "GET" and path == "/stream":
                response_headers = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: keep-alive\r\n"
                    "Access-Control-Allow-Origin: *\r\n\r\n"
                )
                writer.write(response_headers.encode('utf-8'))
                await writer.drain()
                
                self.clients.add(writer)
                logger.info(f"New dashboard client connected. Total clients: {len(self.clients)}")
                
                # Keep the connection open indefinitely
                while True:
                    await asyncio.sleep(3600)

            # Handle POST /kill (Emergency Stop)
            elif method == "POST" and path == "/kill":
                if self.on_kill_switch:
                    self.on_kill_switch()
                response = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\nKill switch activated."
                writer.write(response.encode('utf-8'))
                await writer.drain()
                writer.close()

            else:
                response = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n"
                writer.write(response.encode('utf-8'))
                await writer.drain()
                writer.close()

        except (asyncio.CancelledError, ConnectionError):
            pass
        except Exception as e:
            logger.error(f"Error handling client request: {str(e)}")
        finally:
            if writer in self.clients:
                self.clients.remove(writer)
                logger.info(f"Dashboard client disconnected. Total clients: {len(self.clients)}")
            try:
                writer.close()
            except Exception:
                pass

    async def broadcast_event(self, event_data: dict) -> None:
        """Serializes and pushes event payload to all connected SSE clients."""
        if not self.clients:
            return

        serialized = json.dumps(event_data)
        sse_message = f"data: {serialized}\n\n".encode('utf-8')
        
        dead_clients = []
        for writer in self.clients:
            try:
                writer.write(sse_message)
                await writer.drain()
            except Exception:
                dead_clients.append(writer)

        for writer in dead_clients:
            if writer in self.clients:
                self.clients.remove(writer)
                try:
                    writer.close()
                except Exception:
                    pass
