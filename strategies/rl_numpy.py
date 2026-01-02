#!/usr/bin/env python3
"""
NumPy-only RL inference for Railway deployment.

Provides Actor network inference without MLX dependency,
enabling deployment on Linux servers.
"""
import numpy as np
from safetensors import safe_open
from dataclasses import dataclass
from typing import Tuple
from .base import Strategy, MarketState, Action


@dataclass
class NumpyActor:
    """
    NumPy implementation of Actor network.

    Architecture: 18 → 128 (tanh) → 128 (tanh) → 3 (softmax)

    Mirrors the MLX Actor from rl_mlx.py but uses pure NumPy
    for Linux compatibility.
    """
    fc1_weight: np.ndarray  # (128, 18)
    fc1_bias: np.ndarray    # (128,)
    fc2_weight: np.ndarray  # (128, 128)
    fc2_bias: np.ndarray    # (128,)
    fc3_weight: np.ndarray  # (3, 128)
    fc3_bias: np.ndarray    # (3,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: state → action probabilities.

        Args:
            x: Input features, shape (batch_size, 18) or (18,)

        Returns:
            Action probabilities, shape (batch_size, 3) or (3,)
        """
        # Ensure 2D input
        squeeze_output = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True

        # Layer 1: Linear + tanh
        h = np.tanh(x @ self.fc1_weight.T + self.fc1_bias)

        # Layer 2: Linear + tanh
        h = np.tanh(h @ self.fc2_weight.T + self.fc2_bias)

        # Layer 3: Linear + softmax
        logits = h @ self.fc3_weight.T + self.fc3_bias

        # Numerically stable softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if squeeze_output:
            probs = probs.squeeze(0)

        return probs


def load_actor_from_safetensors(path: str) -> NumpyActor:
    """
    Load Actor weights from safetensors file.

    Args:
        path: Path to .safetensors file (e.g., 'rl_model.safetensors')

    Returns:
        NumpyActor with loaded weights
    """
    with safe_open(path, framework="numpy") as f:
        return NumpyActor(
            fc1_weight=f.get_tensor("actor.fc1.weight"),
            fc1_bias=f.get_tensor("actor.fc1.bias"),
            fc2_weight=f.get_tensor("actor.fc2.weight"),
            fc2_bias=f.get_tensor("actor.fc2.bias"),
            fc3_weight=f.get_tensor("actor.fc3.weight"),
            fc3_bias=f.get_tensor("actor.fc3.bias"),
        )


class NumpyRLStrategy(Strategy):
    """
    Inference-only RL strategy using NumPy.

    For deployment on Linux servers where MLX is not available.
    Loads trained weights from safetensors and runs inference only.
    """

    def __init__(self, model_path: str = "rl_model"):
        """
        Initialize NumPy RL strategy.

        Args:
            model_path: Base path to model files (without extension).
                        Expects {model_path}.safetensors and {model_path}_stats.npz
        """
        super().__init__("rl-numpy")

        # Load actor network
        safetensors_path = f"{model_path}.safetensors"
        self.actor = load_actor_from_safetensors(safetensors_path)

        # Load normalization stats
        stats_path = f"{model_path}_stats.npz"
        stats = np.load(stats_path)
        self.reward_mean = float(stats["reward_mean"])
        self.reward_std = float(stats["reward_std"])

        # Inference only - no training
        self.training = False

    def act(self, state: MarketState) -> Action:
        """
        Select action using greedy policy (argmax).

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
        Get full action probability distribution.

        Args:
            state: Current market state

        Returns:
            Array of shape (3,) with probabilities for [HOLD, BUY, SELL]
        """
        features = state.to_features()
        return self.actor(features)

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


# Verification function to compare with MLX output
def verify_numpy_matches_mlx(model_path: str = "rl_model") -> bool:
    """
    Verify NumPy inference matches MLX output.

    Only runs on macOS where MLX is available.
    Useful for testing before deployment.

    Returns:
        True if outputs match within tolerance, False otherwise
    """
    try:
        import mlx.core as mx
        from .rl_mlx import RLStrategy as MLXStrategy
    except ImportError:
        print("MLX not available - skipping verification")
        return True

    # Load both strategies
    numpy_strategy = NumpyRLStrategy(model_path)
    mlx_strategy = MLXStrategy()
    mlx_strategy.load(model_path)
    mlx_strategy.training = False

    # Create test state
    test_state = MarketState(
        asset="BTC",
        prob=0.55,
        time_remaining=0.5,
    )
    # Set some features
    test_state.returns_1m = 0.001
    test_state.returns_5m = 0.002
    test_state.order_book_imbalance_l1 = 0.1

    # Get outputs
    numpy_probs = numpy_strategy.get_action_probs(test_state)

    # MLX inference
    features = test_state.to_features().reshape(1, -1)
    mlx_probs = np.array(mlx_strategy.actor(mx.array(features))[0])

    # Compare
    max_diff = np.max(np.abs(numpy_probs - mlx_probs))
    matches = max_diff < 1e-5

    if matches:
        print(f"✓ NumPy matches MLX (max diff: {max_diff:.2e})")
    else:
        print(f"✗ Mismatch! Max diff: {max_diff:.2e}")
        print(f"  NumPy: {numpy_probs}")
        print(f"  MLX:   {mlx_probs}")

    return matches


if __name__ == "__main__":
    # Quick test
    print("Testing NumPy RL Strategy...")

    strategy = NumpyRLStrategy("rl_model")

    # Create test state
    test_state = MarketState(asset="BTC", prob=0.55, time_remaining=0.5)
    test_state.returns_1m = 0.001
    test_state.order_book_imbalance_l1 = 0.15

    action, probs, confidence = strategy.get_action_with_info(test_state)

    print(f"Action: {action.name}")
    print(f"Probs: HOLD={probs[0]:.3f}, BUY={probs[1]:.3f}, SELL={probs[2]:.3f}")
    print(f"Confidence: {confidence:.3f}")

    # Verify against MLX if available
    print("\nVerifying against MLX...")
    verify_numpy_matches_mlx()
