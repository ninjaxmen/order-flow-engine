import threading
import time
from utils.logger import logger

class ContinuousTrainer(threading.Thread):
    """
    Background thread that continuously trains the neural network via ReplayBuffer.
    By moving PyTorch backpropagation to a separate OS thread, it prevents 
    the asyncio event loop (handling websocket ticks) from stuttering.
    """
    def __init__(self, agent, interval_seconds=5.0):
        super().__init__(daemon=True, name="OnlineLearnerThread")
        self.agent = agent
        self.interval = interval_seconds
        self.is_running = True

    def run(self):
        logger.info("[ONLINE LEARNER] Background training thread activated.")
        while self.is_running:
            time.sleep(self.interval)
            try:
                # Perform a single deep training step (batch backprop)
                self.agent._train_step()
            except Exception as e:
                logger.error(f"[ONLINE LEARNER] Error during backpropagation: {e}")

    def stop(self):
        self.is_running = False
