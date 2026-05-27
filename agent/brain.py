import os
import sys
import random
import time
import json
from collections import deque
from typing import List, Dict, Tuple, Optional

# Dynamically append parent directory to sys.path for absolute package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch (torch) is not installed in the active environment. "
        "The DeepRL Brain will fall back to a High-Performance Analytical Emulator. "
        "Run 'pip install torch' to activate the full Neural Network."
    )

# --- 1. NEURAL NETWORK DEFINITION (PYTORCH) ---
if TORCH_AVAILABLE:
    class ActorCriticNetwork(nn.Module):
        """
        Deep Neural Network implementing the Actor-Critic Architecture.
        - Input: High-dimensional Order Flow State (LOB depth, footprint delta, CVD metrics)
        - Actor Head: Outputs policy probabilities over actions [HOLD, BUY, SELL, CLOSE]
        - Critic Head: Outputs expected state value V(s) (expected scalp PnL)
        """
        def __init__(self, state_dim: int = 36, action_dim: int = 4):
            super(ActorCriticNetwork, self).__init__()
            
            # Common hidden layers (Feature extraction)
            self.common = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            
            # Actor head (policy distribution over actions)
            self.actor = nn.Sequential(
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Critic head (state value estimation)
            self.critic = nn.Linear(64, 1)
            
            # Forward Dynamics Model (Intrinsic Curiosity)
            self.forward_model = nn.Sequential(
                nn.Linear(state_dim + action_dim, 64),
                nn.ReLU(),
                nn.Linear(64, state_dim)
            )
            self.action_dim = action_dim

        def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            action_one_hot = torch.zeros(action.size(0), self.action_dim, device=state.device)
            action_one_hot.scatter_(1, action, 1.0)
            x = torch.cat([state, action_one_hot], dim=1)
            return self.forward_model(x)

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.common(state)
            policy = self.actor(x)
            value = self.critic(x)
            return policy, value


# --- 2. EXPERIENCE REPLAY BUFFER ---
class ReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer.
    Samples high-surprise and high-PnL events more frequently to focus learning on critical market shifts.
    """
    def __init__(self, max_size: int = 5000):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = 0.6  # PER prioritization factor

    def add(self, state: list, action: int, reward: float, next_state: list, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        # Assign high priority to terminal states or large rewards
        base_priority = abs(reward) if reward != 0 else 0.01
        if done:
            base_priority += 1.0  # Boost trade exits
        priority = (base_priority + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Optional[List[tuple]]:
        if len(self.buffer) < batch_size:
            return None
            
        probs = np.array(self.priorities)
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# --- 3. THE HIGH-INTELLIGENCE RL AGENT ---
class DeepOrderFlowAgent:
    """
    Autonomous Reinforcement Learning agent driving order flow decisions.
    Trained online via Actor-Critic policy updates using backpropagated market rewards.
    """
    def __init__(self, state_dim: int = 36, action_dim: int = 4, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        
        self.replay_buffer = ReplayBuffer(max_size=2000)
        self.batch_size = 32
        self.gamma = 0.99  # discount factor for future rewards
        
        # State tracking for training steps per symbol
        self.last_state: Dict[str, list] = {}
        self.last_action: Dict[str, int] = {}
        self.last_price: Dict[str, float] = {}
        
        # Regime Detection window
        self.price_history_window: Dict[str, deque] = {}
        
        # Action space mapping
        self.action_labels = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
        
        # Training Metrics
        self.metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "forward_loss": 0.0,
            "entropy": 0.0,
            "grad_norm": 0.0
        }

        if TORCH_AVAILABLE:
            self.network = ActorCriticNetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.harness_type = "Deep Actor-Critic Neural Network"
        else:
            self.network = None
            self.optimizer = None
            self.harness_type = "High-Performance Analytical Emulator"
            # Simple analytical weights fallback
            self.analytical_weights = [0.25] * state_dim

    def detect_regime(self, symbol: str, price: float) -> str:
        """Labels the market regime based on rolling price volatility."""
        if symbol not in self.price_history_window:
            self.price_history_window[symbol] = deque(maxlen=60)
        
        self.price_history_window[symbol].append(price)
        history = list(self.price_history_window[symbol])
        
        if len(history) < 20:
            return "SIDEWAYS"
            
        std_dev = np.std(history)
        mean_price = np.mean(history)
        coef_var = std_dev / mean_price if mean_price > 0 else 0
        
        if coef_var > 0.002:
            return "BREAKOUT"
        elif coef_var > 0.0005:
            return "MEAN_REVERSION"
        else:
            return "SIDEWAYS"

    def select_action(self, symbol: str, state_vector: list, price: float, institutional_signal_active: bool = False) -> Tuple[int, str, float]:
        """
        Encodes the state and queries the policy to output the optimal trading action.
        Returns: (Action_Index, Action_Label, Confidence)
        """
        self.last_price[symbol] = price
        self.last_state[symbol] = state_vector

        # --- PYTORCH NEURAL POLICY SELECTION ---
        if TORCH_AVAILABLE and self.network is not None:
            try:
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                
                with torch.no_grad():
                    policy, _ = self.network(state_tensor)
                    
                policy_np = policy.squeeze(0).numpy()
                
                # Expert Constraint Layer: Regime Detection biasing (REMOVED for full exploration freedom)
                # regime = self.detect_regime(symbol, price)
                # if regime == "SIDEWAYS":
                #     policy_np[0] += 0.4  # Heavily bias HOLD in sideways noise
                # elif regime == "MEAN_REVERSION":
                #     policy_np[0] += 0.1
                    
                # Re-normalize after regime bias
                # policy_np = policy_np / policy_np.sum()
                
                # Soft Expert Guidance: Gently reduce (but do not block) trade probabilities if there is no footprint evidence (REMOVED for full exploration freedom)
                # if not institutional_signal_active:
                #     policy_np[1] *= 0.3  # Penalize BUY
                #     policy_np[2] *= 0.3  # Penalize SELL
                #     policy_np = policy_np / policy_np.sum()  # Re-normalize
                
                # Stochastic Exploration Mode: Sample action to allow learning from mistakes
                action = int(np.random.choice(self.action_dim, p=policy_np))
                confidence = float(policy_np[action])
                
                self.last_action[symbol] = action
                return action, self.action_labels[action], confidence
                
            except Exception as e:
                logger.error(f"Error in deep policy action selection: {str(e)}")

        # --- HIGH-PERFORMANCE ANALYTICAL EMULATOR FALLBACK ---
        # Computes policy probabilities based on mathematical confluences
        score = sum(s * w for s, w in zip(state_vector, self.analytical_weights))
        
        # Determine action index
        if score > 0.6 and institutional_signal_active:
            action = 1  # BUY
        elif score < -0.6 and institutional_signal_active:
            action = 2  # SELL
        else:
            action = 0  # HOLD
            
        confidence = min(0.95, 0.5 + abs(score) * 0.5)
        
        # Full Autonomy Mode: No Confidence Gating
        MIN_CONFIDENCE_THRESHOLD = 0.0
        if action in [1, 2] and confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.info(f"[ANALYTICAL GATING] {self.action_labels[action]} suppressed on {symbol}. Confidence {confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD}.")
            action = 0
            confidence = 0.5
            
        self.last_action[symbol] = action
        return action, self.action_labels[action], confidence

    def learn_from_feedback(self, symbol: str, pnl: float, next_state_vector: list, done: bool) -> None:
        """
        Shapes the reward based on trade performance and triggers backpropagation updates.
        """
        if symbol not in self.last_state or symbol not in self.last_action:
            return

        # Action-Switching Penalty (Reward Function Refinement)
        if not hasattr(self, "action_history_queue"):
            from collections import deque
            import time
            self.action_history_queue = {}
            
        if symbol not in self.action_history_queue:
            from collections import deque
            self.action_history_queue[symbol] = deque(maxlen=60) # ~3 minutes of history
            
        import time
        self.action_history_queue[symbol].append((time.time(), self.last_action[symbol]))
        
        switching_penalty = 0.0
        now = time.time()
        # Keep only actions from the last 3 minutes
        valid_history = [a for t, a in self.action_history_queue[symbol] if now - t <= 180.0]
        
        if len(valid_history) >= 2:
            prev_action = valid_history[-2]
            curr_action = valid_history[-1]
            if prev_action in [1, 2] and curr_action in [1, 2] and prev_action != curr_action:
                switching_penalty = 0.5  # Heavy switching penalty
                logger.debug(f"[REWARD SHAPING] Applied -0.5 switching penalty on {symbol} (Flipped {prev_action} to {curr_action})")

        # 1. Cognitive Reward Shaping
        if done:
            # Profit reward + friction penalty to prevent fee-bleeding over-trading
            reward = pnl
            if abs(pnl) < 1.0:
                reward -= 0.05  # Turnover fee friction penalty
        else:
            # Continuous Intrinsic Reward: scaled down unrealized PnL step
            reward = pnl * 0.1
            
        reward -= switching_penalty
            
        # 2. Add transition to Replay Buffer
        self.replay_buffer.add(
            self.last_state[symbol], 
            self.last_action[symbol], 
            reward, 
            next_state_vector, 
            done
        )

        # 3. Training is now triggered periodically by the main orchestrator

    def _train_step(self) -> None:
        """
        Performs one gradient descent step over a sampled mini-batch of experiences.
        Updates both Actor and Critic weights simultaneously.
        """
        batch = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch Tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        try:
            # Curiosity / Forward model prediction
            pred_next_states = self.network.predict_next_state(states_t, actions_t)
            forward_loss = nn.MSELoss()(pred_next_states, next_states_t)
            
            # Intrinsic reward based on prediction error (surprise)
            with torch.no_grad():
                intrinsic_rewards = torch.mean((pred_next_states - next_states_t)**2, dim=1, keepdim=True)
            
            # Combine extrinsic and intrinsic rewards
            total_rewards = rewards_t + 0.1 * intrinsic_rewards
            
            # Get current policy and value estimations
            policy, values = self.network(states_t)
            
            # Get next state values for temporal difference target
            with torch.no_grad():
                _, next_values = self.network(next_states_t)
                
            # TD Target: Y = R + gamma * V(s') * (1 - done)
            targets = total_rewards + self.gamma * next_values * (1 - dones_t)
            
            # Critic Loss (Mean Squared Error between value prediction and TD Target)
            critic_loss = nn.MSELoss()(values, targets)

            # Actor Loss (Policy Gradient: -log(pi(a|s)) * Advantage)
            advantages = (targets - values).detach()
            selected_action_probs = policy.gather(1, actions_t)
            actor_loss = -(torch.log(selected_action_probs + 1e-10) * advantages).mean()

            # Entropy Bonus (to avoid policy collapse)
            entropy = -torch.sum(policy * torch.log(policy + 1e-10), dim=1).mean()

            # Total Loss
            total_loss = actor_loss + 0.5 * critic_loss + forward_loss - 0.01 * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping and norm tracking
            grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            self.metrics["actor_loss"] = float(actor_loss.item())
            self.metrics["critic_loss"] = float(critic_loss.item())
            self.metrics["forward_loss"] = float(forward_loss.item())
            self.metrics["entropy"] = float(entropy.item())
            self.metrics["grad_norm"] = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            
            # Periodically log to console (10% chance to avoid log spam)
            if random.random() < 0.10:
                logger.info(
                    f"[BRAIN] Training Step | Actor Loss: {self.metrics['actor_loss']:.4f} | "
                    f"Critic: {self.metrics['critic_loss']:.4f} | Curiosity: {self.metrics['forward_loss']:.4f} | "
                    f"Entropy: {self.metrics['entropy']:.4f}"
                )

        except Exception as e:
            logger.error(f"Error in deep neural network backpropagation step: {str(e)}")
            
    def export_policy_metrics(self) -> Dict:
        """Returns diagnostic details about the active brain."""
        return {
            "harness": self.harness_type,
            "buffer_size": len(self.replay_buffer),
            "torch_active": TORCH_AVAILABLE,
            "metrics": self.metrics
        }
        
    def save_checkpoint(self, filepath: str) -> None:
        """Saves the neural network weights to disk."""
        if TORCH_AVAILABLE and self.network is not None:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                torch.save(self.network.state_dict(), filepath)
                logger.info(f"Agent neural network checkpoint saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {str(e)}")
                
    def load_checkpoint(self, filepath: str) -> None:
        """Loads neural network weights from disk."""
        if TORCH_AVAILABLE and self.network is not None and os.path.exists(filepath):
            try:
                self.network.load_state_dict(torch.load(filepath))
                logger.info(f"Agent neural network checkpoint loaded from {filepath}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
