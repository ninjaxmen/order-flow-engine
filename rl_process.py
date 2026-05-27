import multiprocessing as mp
import time
import os

class RLProcess(mp.Process):
    def __init__(self, request_queue: mp.Queue, response_queue: mp.Queue):
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.agent = None

    def run(self):
        # Initialize agent inside the isolated process to ensure PyTorch tensors 
        # and memory are strictly bound to this CPU core.
        from agent.brain import DeepOrderFlowAgent
        import torch
        
        self.agent = DeepOrderFlowAgent(state_dim=36, action_dim=4, learning_rate=0.001)
        
        expert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "expert_weights.pth")
        if os.path.exists(expert_path):
            self.agent.load_checkpoint(expert_path)

        while True:
            try:
                # Block until we receive a request from the Execution Engine
                req = self.request_queue.get()
                
                if req is None or req.get("type") == "SHUTDOWN":
                    self.agent.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "brain_checkpoint.pth"))
                    break

                if req["type"] == "EVALUATE":
                    symbol = req["symbol"]
                    state = req["state"]
                    last_price = req["last_price"]
                    inst_sig = req["inst_sig"]
                    
                    # 1. Forward Pass (No GIL contention with the Execution Engine)
                    action_idx, action_label, confidence = self.agent.select_action(symbol, state, last_price, inst_sig)
                    
                    # 2. Extract policy probabilities for the dashboard
                    probs_dict = {}
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action_probs, _ = self.agent.network(state_tensor)
                        probs = action_probs.squeeze(0).numpy()
                        probs_dict = {
                            "HOLD": float(probs[0]),
                            "BUY": float(probs[1]),
                            "SELL": float(probs[2]),
                            "CLOSE": float(probs[3])
                        }
                    
                    # 3. Send response back to Execution Engine instantly
                    self.response_queue.put({
                        "type": "ACTION",
                        "symbol": symbol,
                        "action_idx": action_idx,
                        "action_label": action_label,
                        "confidence": confidence,
                        "probs": probs_dict
                    })
                    
                    # 4. Do background learning if memory is full
                    self.agent.train_step()

                elif req["type"] == "REWARD":
                    # Store reward in experience replay
                    symbol = req["symbol"]
                    reward = req["reward"]
                    next_state = req.get("next_state", [])
                    self.agent.learn_from_feedback(symbol, reward, next_state, done=True)
                    self.agent.train_step()

            except Exception as e:
                # Silently catch so the process doesn't die
                pass
