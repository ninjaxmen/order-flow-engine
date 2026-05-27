import os

# API Credentials (loaded from environment or defaults)
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "MOCK_DHAN_CLIENT")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "MOCK_DHAN_TOKEN")

DELTA_API_KEY = "zY4xTDh0EtQrQcShwyJRGyet7rgofc"
DELTA_API_SECRET = "Dm42FX9RZ0XB4WIoLeLBwnDkRzfiZUrHW3iT9ZEzmMFfXYolecHURpTNeeMC"

# Endpoints
DHAN_FEED_URL = "wss://api-feed.dhan.co?version=2&token={token}&clientId={client_id}&authType=2"
DHAN_DEPTH_URL = "wss://depth-api-feed.dhan.co/twentydepth?token={token}&clientId={client_id}&authType=2"

DELTA_FEED_URL = "wss://socket-ind-pub.testnet.deltaex.org"
DELTA_TESTNET_FEED_URL = "wss://socket-ind-pub.testnet.deltaex.org"
DELTA_API_ENDPOINT = "https://cdn-ind.testnet.deltaex.org"

# Engine Parameters
FOOTPRINT_INTERVAL_SEC = int(os.getenv("FOOTPRINT_INTERVAL_SEC", "60"))  # 1-minute footprint bar
MAX_Footprint_Hist_Bars = 100  # Number of historical intervals to keep in memory

# Signal/Heuristics Thresholds
VOLUME_IMBALANCE_RATIO = 3.0  # e.g., Bid side volume must be >= 3x of Ask side (or vice versa)
ABSORPTION_MIN_VOL_THRESHOLD = 500  # Minimum contracts traded at a single level to evaluate absorption
EXHAUSTION_VOLUME_RATIO = 0.25  # Volume must fall below 25% of recent average to flag exhaustion at extremums

# Cache / Buffer Configuration
TICK_BUFFER_SIZE = 100000  # Limit buffer to avoid out-of-memory errors
LOB_DEPTH_LEVELS = 20  # Maintain standard L2 20 depth levels in memory

# Simulated Feeds Config
SIMULATION_MODE = False  # Set to False to disable simulation and run on real feeds
SIM_TICK_RATE = 100  # Rate of ticks per second generated in simulation mode
