#!/usr/bin/env python3
"""
NumPy-only Phase 5 RL inference for production deployment.

Provides Phase 5 temporal architecture (TemporalEncoder + Asymmetric Actor/Critic)
without MLX dependency, enabling deployment on Linux servers.

Phase 5 Features:
- TemporalEncoder: Processes last 5 states → 32 temporal features
- Asymmetric Actor: 64 hidden units (smaller to prevent overfitting)
- Combined input: 18 current + 32 temporal = 50 dimensions
"""
import numpy as np
from safetensors import safe_open
from dataclasses import dataclass
from typing import Tuple, Dict, Deque
from collections import deque
from .base import Strategy, MarketState, Action


@dataclass
class NumpyTemporalEncoder:
    """
    NumPy implementation of TemporalEncoder.

    Processes stacked temporal features (5 states × 18 features = 90)
    → 32 temporal features

    Architecture: 90 → 64 (tanh) → 32 (tanh)
    With LayerNorm after each layer.
    """
    fc1_weight: np.ndarray  # (64, 90)
    fc1_bias: np.ndarray    # (64,)
    ln1_weight: np.ndarray  # (64,) LayerNorm scale
    ln1_bias: np.ndarray    # (64,) LayerNorm shift
    fc2_weight: np.ndarray  # (32, 64)
    fc2_bias: np.ndarray    # (32,)
    ln2_weight: np.ndarray  # (32,) LayerNorm scale
    ln2_bias: np.ndarray    # (32,) LayerNorm shift

    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply LayerNorm: (x - mean) / std * weight + bias"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return x_norm * weight + bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: stacked temporal features → temporal encoding.

        Args:
            x: Input features, shape (batch_size, 90) or (90,) [5 states × 18 features]

        Returns:
            Temporal features, shape (batch_size, 32) or (32,)
        """
        squeeze_output = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True

        # Layer 1: Linear → LayerNorm → Tanh
        h = x @ self.fc1_weight.T + self.fc1_bias
        h = self.layer_norm(h, self.ln1_weight, self.ln1_bias)
        h = np.tanh(h)

        # Layer 2: Linear → LayerNorm → Tanh
        h = h @ self.fc2_weight.T + self.fc2_bias
        h = self.layer_norm(h, self.ln2_weight, self.ln2_bias)
        h = np.tanh(h)

        if squeeze_output:
            h = h.squeeze(0)

        return h


@dataclass
class NumpyActor:
    """
    NumPy implementation of Phase 5 Actor network (Asymmetric - 64 hidden).

    Architecture: 50 (18 current + 32 temporal) → 64 (tanh) → 64 (tanh) → 3 (softmax)
    With LayerNorm after each hidden layer.
    """
    fc1_weight: np.ndarray  # (64, 50)
    fc1_bias: np.ndarray    # (64,)
    ln1_weight: np.ndarray  # (64,)
    ln1_bias: np.ndarray    # (64,)
    fc2_weight: np.ndarray  # (64, 64)
    fc2_bias: np.ndarray    # (64,)
    ln2_weight: np.ndarray  # (64,)
    ln2_bias: np.ndarray    # (64,)
    fc3_weight: np.ndarray  # (3, 64)
    fc3_bias: np.ndarray    # (3,)

    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply LayerNorm"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return x_norm * weight + bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: combined state → action probabilities.

        Args:
            x: Input features, shape (batch_size, 50) or (50,) [18 current + 32 temporal]

        Returns:
            Action probabilities, shape (batch_size, 3) or (3,)
        """
        squeeze_output = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True

        # Layer 1: Linear → LayerNorm → Tanh
        h = x @ self.fc1_weight.T + self.fc1_bias
        h = self.layer_norm(h, self.ln1_weight, self.ln1_bias)
        h = np.tanh(h)

        # Layer 2: Linear → LayerNorm → Tanh
        h = h @ self.fc2_weight.T + self.fc2_bias
        h = self.layer_norm(h, self.ln2_weight, self.ln2_bias)
        h = np.tanh(h)

        # Layer 3: Linear → Softmax
        logits = h @ self.fc3_weight.T + self.fc3_bias

        # Numerically stable softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if squeeze_output:
            probs = probs.squeeze(0)

        return probs


def load_phase5_from_safetensors(path: str) -> Tuple[NumpyTemporalEncoder, NumpyActor]:
    """
    Load Phase 5 TemporalEncoder and Actor weights from safetensors file.

    Args:
        path: Path to .safetensors file (e.g., 'rl_model.safetensors')

    Returns:
        Tuple of (NumpyTemporalEncoder, NumpyActor) with loaded weights
    """
    with safe_open(path, framework="numpy") as f:
        # Load TemporalEncoder weights
        temporal_encoder = NumpyTemporalEncoder(
            fc1_weight=f.get_tensor("actor.temporal_encoder.fc1.weight"),
            fc1_bias=f.get_tensor("actor.temporal_encoder.fc1.bias"),
            ln1_weight=f.get_tensor("actor.temporal_encoder.ln1.weight"),
            ln1_bias=f.get_tensor("actor.temporal_encoder.ln1.bias"),
            fc2_weight=f.get_tensor("actor.temporal_encoder.fc2.weight"),
            fc2_bias=f.get_tensor("actor.temporal_encoder.fc2.bias"),
            ln2_weight=f.get_tensor("actor.temporal_encoder.ln2.weight"),
            ln2_bias=f.get_tensor("actor.temporal_encoder.ln2.bias"),
        )

        # Load Actor weights
        actor = NumpyActor(
            fc1_weight=f.get_tensor("actor.fc1.weight"),
            fc1_bias=f.get_tensor("actor.fc1.bias"),
            ln1_weight=f.get_tensor("actor.ln1.weight"),
            ln1_bias=f.get_tensor("actor.ln1.bias"),
            fc2_weight=f.get_tensor("actor.fc2.weight"),
            fc2_bias=f.get_tensor("actor.fc2.bias"),
            ln2_weight=f.get_tensor("actor.ln2.weight"),
            ln2_bias=f.get_tensor("actor.ln2.bias"),
            fc3_weight=f.get_tensor("actor.fc3.weight"),
            fc3_bias=f.get_tensor("actor.fc3.bias"),
        )

        return temporal_encoder, actor


class NumpyRLStrategy(Strategy):
    """
    Phase 5 inference-only RL strategy using NumPy.

    Implements temporal architecture with state history tracking.
    For deployment on Linux servers where MLX is not available.
    """

    def __init__(self, model_path: str = "rl_model",
                 history_len: int = 5, input_dim: int = 18, temporal_dim: int = 32):
        """
        Initialize Phase 5 NumPy RL strategy.

        Args:
            model_path: Base path to model files (without extension)
            history_len: Number of historical states to track (5 for Phase 5)
            input_dim: Input feature dimension (18)
            temporal_dim: Temporal encoder output dimension (32)
        """
        super().__init__("rl-numpy-phase5")

        self.history_len = history_len
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim

        # Load Phase 5 networks
        safetensors_path = f"{model_path}.safetensors"
        self.temporal_encoder, self.actor = load_phase5_from_safetensors(safetensors_path)

        # State history per asset (deque of feature vectors)
        self.state_history: Dict[str, Deque[np.ndarray]] = {}

        # Load normalization stats
        stats_path = f"{model_path}_stats.npz"
        stats = np.load(stats_path)
        self.reward_mean = float(stats["reward_mean"])
        self.reward_std = float(stats["reward_std"])

        # Inference only
        self.training = False

    def _get_temporal_state(self, asset: str, current_features: np.ndarray) -> np.ndarray:
        """
        Get stacked temporal state for an asset.

        Updates history with current features, then returns stacked history.
        If history length < history_len, pads with zeros.

        Args:
            asset: Asset identifier (e.g., 'BTC')
            current_features: Current state features (18,)

        Returns:
            Stacked temporal features (90,) [5 states × 18 features]
        """
        # Initialize history for new assets
        if asset not in self.state_history:
            self.state_history[asset] = deque(maxlen=self.history_len)

        # Add current state to history
        self.state_history[asset].append(current_features.copy())

        # Stack history (most recent last)
        history_list = list(self.state_history[asset])

        # Pad if needed (zeros for missing history)
        while len(history_list) < self.history_len:
            history_list.insert(0, np.zeros(self.input_dim, dtype=np.float32))

        # Stack into single array
        temporal_state = np.concatenate(history_list)

        return temporal_state

    def act(self, state: MarketState) -> Action:
        """
        Select action using greedy policy with temporal context.

        Args:
            state: Current market state

        Returns:
            Selected action (HOLD, BUY, or SELL)
        """
        probs = self.get_action_probs(state)
        action_idx = int(np.argmax(probs))
        return Action(action_idx)

    def get_action_probs(self, state: MarketState) -> np.ndarray:
        """
        Get full action probability distribution with temporal context.

        Args:
            state: Current market state

        Returns:
            Array of shape (3,) with probabilities for [HOLD, BUY, SELL]
        """
        # Extract current features
        current_features = state.to_features()

        # Get temporal state (updates history)
        temporal_state = self._get_temporal_state(state.asset, current_features)

        # Encode temporal context
        temporal_features = self.temporal_encoder(temporal_state)

        # Combine current + temporal
        combined = np.concatenate([current_features, temporal_features])

        # Get action probabilities
        probs = self.actor(combined)

        return probs

    def get_action_with_info(self, state: MarketState) -> Tuple[Action, np.ndarray, float]:
        """
        Get action along with probabilities and confidence.

        Args:
            state: Current market state

        Returns:
            Tuple of (action, probabilities, confidence)
            where confidence is max_prob - second_max_prob
        """
        probs = self.get_action_probs(state)
        action_idx = int(np.argmax(probs))

        # Confidence: difference between top two probabilities
        sorted_probs = np.sort(probs)[::-1]
        confidence = sorted_probs[0] - sorted_probs[1]

        return Action(action_idx), probs, confidence


if __name__ == "__main__":
    # Quick test
    print("Testing Phase 5 NumPy RL Strategy...")

    strategy = NumpyRLStrategy("rl_model")

    # Create test state
    test_state = MarketState(asset="BTC", prob=0.55, time_remaining=0.5)
    test_state.returns_1m = 0.001
    test_state.order_book_imbalance_l1 = 0.15

    # Test with 5 different states to build history
    for i in range(5):
        test_state.returns_1m = 0.001 * (i + 1)
        action, probs, confidence = strategy.get_action_with_info(test_state)
        print(f"\nIteration {i+1}:")
        print(f"  Action: {action.name}")
        print(f"  Probs: HOLD={probs[0]:.3f}, BUY={probs[1]:.3f}, SELL={probs[2]:.3f}")
        print(f"  Confidence: {confidence:.3f}")

    print("\n✓ Phase 5 NumPy implementation working!")
