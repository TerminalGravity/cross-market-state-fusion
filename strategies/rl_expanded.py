#!/usr/bin/env python3
"""
Expanded PPO strategy with multi-source features.

Uses 42 features from:
- Original 18: Binance + Polymarket orderbook + position state
- Coinglass 6: Funding rates, OI, liquidations
- Deribit 4: Options IV, skew, put/call ratio
- On-chain 4: Exchange flows, whale alerts
- Sentiment 3: Fear & Greed, social volume
- Reserved 7: Future expansion

Architecture changes from rl_mlx.py:
- Input: 42 (expanded from 18)
- Hidden: 128 (increased from 64)
- Temporal encoder: 96 (expanded temporal context)
- Added feature group attention mechanism
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .base import Strategy, MarketState, Action

# Feature dimensions
ORIGINAL_DIM = 18
EXPANDED_DIM = 42
COINGLASS_DIM = 6
DERIBIT_DIM = 4
ONCHAIN_DIM = 4
SENTIMENT_DIM = 3
RESERVED_DIM = 7


@dataclass
class ExpandedExperience:
    """Experience tuple with expanded features."""
    state: np.ndarray
    temporal_state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_temporal_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class FeatureGroupEncoder(nn.Module):
    """Encodes each feature group separately before combining."""

    def __init__(self, output_dim: int = 64):
        super().__init__()

        self.core_encoder = nn.Sequential(
            nn.Linear(ORIGINAL_DIM, 32),
            nn.LayerNorm(32),
        )
        self.coinglass_encoder = nn.Sequential(
            nn.Linear(COINGLASS_DIM, 16),
            nn.LayerNorm(16),
        )
        self.deribit_encoder = nn.Sequential(
            nn.Linear(DERIBIT_DIM, 16),
            nn.LayerNorm(16),
        )
        self.onchain_encoder = nn.Sequential(
            nn.Linear(ONCHAIN_DIM, 8),
            nn.LayerNorm(8),
        )
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(SENTIMENT_DIM, 8),
            nn.LayerNorm(8),
        )

        combined_dim = 32 + 16 + 16 + 8 + 8
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        core = x[:, :ORIGINAL_DIM]
        coinglass = x[:, ORIGINAL_DIM:ORIGINAL_DIM + COINGLASS_DIM]
        deribit = x[:, ORIGINAL_DIM + COINGLASS_DIM:ORIGINAL_DIM + COINGLASS_DIM + DERIBIT_DIM]
        onchain = x[:, ORIGINAL_DIM + COINGLASS_DIM + DERIBIT_DIM:ORIGINAL_DIM + COINGLASS_DIM + DERIBIT_DIM + ONCHAIN_DIM]
        sentiment_start = ORIGINAL_DIM + COINGLASS_DIM + DERIBIT_DIM + ONCHAIN_DIM
        sentiment = x[:, sentiment_start:sentiment_start + SENTIMENT_DIM]

        core_enc = mx.tanh(self.core_encoder(core))
        coinglass_enc = mx.tanh(self.coinglass_encoder(coinglass))
        deribit_enc = mx.tanh(self.deribit_encoder(deribit))
        onchain_enc = mx.tanh(self.onchain_encoder(onchain))
        sentiment_enc = mx.tanh(self.sentiment_encoder(sentiment))

        combined = mx.concatenate([
            core_enc, coinglass_enc, deribit_enc, onchain_enc, sentiment_enc
        ], axis=-1)

        return mx.tanh(self.projection(combined))


class ExpandedTemporalEncoder(nn.Module):
    """Temporal encoder for expanded features."""

    def __init__(self, input_dim: int = EXPANDED_DIM, history_len: int = 5, output_dim: int = 48):
        super().__init__()
        self.history_len = history_len
        self.temporal_input = input_dim * history_len

        self.fc1 = nn.Linear(self.temporal_input, 96)
        self.ln1 = nn.LayerNorm(96)
        self.fc2 = nn.Linear(96, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.tanh(self.ln1(self.fc1(x)))
        h = mx.tanh(self.ln2(self.fc2(h)))
        return h


class ExpandedActor(nn.Module):
    """Policy network with expanded multi-source features."""

    def __init__(
        self,
        input_dim: int = EXPANDED_DIM,
        hidden_size: int = 128,
        output_dim: int = 3,
        history_len: int = 5,
        feature_enc_dim: int = 64,
        temporal_dim: int = 48
    ):
        super().__init__()
        self.feature_encoder = FeatureGroupEncoder(output_dim=feature_enc_dim)
        self.temporal_encoder = ExpandedTemporalEncoder(input_dim, history_len, temporal_dim)

        combined_dim = feature_enc_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def __call__(self, current_state: mx.array, temporal_state: mx.array) -> mx.array:
        feature_enc = self.feature_encoder(current_state)
        temporal_enc = self.temporal_encoder(temporal_state)
        combined = mx.concatenate([feature_enc, temporal_enc], axis=-1)

        h = mx.tanh(self.ln1(self.fc1(combined)))
        h = mx.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)

        return mx.softmax(logits, axis=-1)


class ExpandedCritic(nn.Module):
    """Value network with expanded features."""

    def __init__(
        self,
        input_dim: int = EXPANDED_DIM,
        hidden_size: int = 192,
        history_len: int = 5,
        feature_enc_dim: int = 64,
        temporal_dim: int = 48
    ):
        super().__init__()
        self.feature_encoder = FeatureGroupEncoder(output_dim=feature_enc_dim)
        self.temporal_encoder = ExpandedTemporalEncoder(input_dim, history_len, temporal_dim)

        combined_dim = feature_enc_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def __call__(self, current_state: mx.array, temporal_state: mx.array) -> mx.array:
        feature_enc = self.feature_encoder(current_state)
        temporal_enc = self.temporal_encoder(temporal_state)
        combined = mx.concatenate([feature_enc, temporal_enc], axis=-1)

        h = mx.tanh(self.ln1(self.fc1(combined)))
        h = mx.tanh(self.ln2(self.fc2(h)))
        return self.fc3(h)


def materialize_params(params):
    """Force computation of lazy MLX parameters."""
    for key, val in params.items():
        if hasattr(val, 'tolist'):
            _ = val.tolist()


class ExpandedRLStrategy(Strategy):
    """PPO strategy with expanded 42-feature input."""

    def __init__(
        self,
        name: str = "ExpandedRL",
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.10,
        buffer_size: int = 512,
        batch_size: int = 64,
        ppo_epochs: int = 10,
        history_len: int = 5,
    ):
        super().__init__(name)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.history_len = history_len

        self.actor = ExpandedActor(input_dim=EXPANDED_DIM, hidden_size=128, history_len=history_len)
        self.critic = ExpandedCritic(input_dim=EXPANDED_DIM, hidden_size=192, history_len=history_len)

        self.actor_opt = optim.Adam(learning_rate=actor_lr)
        self.critic_opt = optim.Adam(learning_rate=critic_lr)

        self.buffer: List[ExpandedExperience] = []
        self.state_history: deque = deque(maxlen=history_len)

        self.update_count = 0
        self.total_reward = 0.0

        self._init_networks()

    def _init_networks(self):
        """Initialize networks with dummy data."""
        dummy_state = mx.zeros((1, EXPANDED_DIM))
        dummy_temporal = mx.zeros((1, self.history_len * EXPANDED_DIM))

        _ = self.actor(dummy_state, dummy_temporal)
        _ = self.critic(dummy_state, dummy_temporal)

        materialize_params(self.actor.parameters())
        materialize_params(self.critic.parameters())

    def _get_temporal_state(self) -> np.ndarray:
        """Get stacked temporal state from history."""
        if len(self.state_history) < self.history_len:
            padding = [np.zeros(EXPANDED_DIM) for _ in range(self.history_len - len(self.state_history))]
            states = padding + list(self.state_history)
        else:
            states = list(self.state_history)

        return np.concatenate(states)

    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        features = state.to_expanded_features()
        temporal = self._get_temporal_state()

        self.state_history.append(features.copy())

        state_mx = mx.array(features.reshape(1, -1))
        temporal_mx = mx.array(temporal.reshape(1, -1))

        probs = self.actor(state_mx, temporal_mx)
        probs = np.array(probs[0])

        if self.training:
            action_idx = np.random.choice(3, p=probs)
        else:
            action_idx = np.argmax(probs)

        return Action(action_idx)

    def store_experience(
        self,
        state: MarketState,
        action: Action,
        reward: float,
        next_state: MarketState,
        done: bool
    ):
        """Store experience in buffer."""
        if not self.training:
            return

        state_feat = state.to_expanded_features()
        next_feat = next_state.to_expanded_features()
        temporal = self._get_temporal_state()
        next_temporal = temporal.copy()

        state_mx = mx.array(state_feat.reshape(1, -1))
        temporal_mx = mx.array(temporal.reshape(1, -1))

        probs = np.array(self.actor(state_mx, temporal_mx)[0])
        value = float(self.critic(state_mx, temporal_mx)[0, 0])
        log_prob = np.log(probs[action.value] + 1e-8)

        exp = ExpandedExperience(
            state=state_feat,
            temporal_state=temporal,
            action=action.value,
            reward=reward,
            next_state=next_feat,
            next_temporal_state=next_temporal,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.buffer.append(exp)
        self.total_reward += reward

        if len(self.buffer) >= self.buffer_size:
            self.update()

    def update(self) -> Dict[str, float]:
        """Perform PPO update."""
        if len(self.buffer) < self.batch_size:
            return {}

        advantages, returns = self._compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = mx.array(np.stack([e.state for e in self.buffer]))
        temporals = mx.array(np.stack([e.temporal_state for e in self.buffer]))
        actions = mx.array(np.array([e.action for e in self.buffer]))
        old_log_probs = mx.array(np.array([e.log_prob for e in self.buffer]))
        advantages_mx = mx.array(advantages)
        returns_mx = mx.array(returns)

        metrics = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'kl_div': 0.0}

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(self.buffer))

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                batch_states = states[batch_idx]
                batch_temporals = temporals[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_mx[batch_idx]
                batch_returns = returns_mx[batch_idx]

                actor_loss, actor_grads = self._actor_loss_and_grad(
                    batch_states, batch_temporals, batch_actions,
                    batch_old_log_probs, batch_advantages
                )
                self.actor_opt.update(self.actor, actor_grads)
                materialize_params(self.actor.parameters())

                critic_loss, critic_grads = self._critic_loss_and_grad(
                    batch_states, batch_temporals, batch_returns
                )
                self.critic_opt.update(self.critic, critic_grads)
                materialize_params(self.critic.parameters())

                metrics['actor_loss'] += float(actor_loss)
                metrics['critic_loss'] += float(critic_loss)

        num_batches = (len(self.buffer) // self.batch_size) * self.ppo_epochs
        for k in metrics:
            metrics[k] /= max(1, num_batches)

        self.buffer.clear()
        self.update_count += 1

        return metrics

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n = len(self.buffer)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            exp = self.buffer[t]

            if exp.done:
                delta = exp.reward - exp.value
                last_gae = delta
            else:
                next_value = self.buffer[t + 1].value if t + 1 < n else last_value
                delta = exp.reward + self.gamma * next_value - exp.value
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

            advantages[t] = last_gae
            returns[t] = advantages[t] + exp.value

        return advantages, returns

    def _actor_loss_and_grad(self, states, temporals, actions, old_log_probs, advantages):
        """Compute actor loss and gradients."""
        def loss_fn(model):
            probs = model(states, temporals)
            action_probs = mx.take_along_axis(probs, actions.reshape(-1, 1), axis=1).squeeze()
            log_probs = mx.log(action_probs + 1e-8)

            ratio = mx.exp(log_probs - old_log_probs)
            clipped_ratio = mx.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -mx.mean(mx.minimum(surrogate1, surrogate2))

            entropy = -mx.mean(mx.sum(probs * mx.log(probs + 1e-8), axis=-1))

            return policy_loss - self.entropy_coef * entropy

        loss, grads = mx.value_and_grad(loss_fn)(self.actor)
        return loss, grads

    def _critic_loss_and_grad(self, states, temporals, returns):
        """Compute critic loss and gradients."""
        def loss_fn(model):
            values = model(states, temporals).squeeze()
            return mx.mean((values - returns) ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(self.critic)
        return loss, grads

    def reset(self):
        """Reset state history."""
        self.state_history.clear()

    def save(self, path: str):
        """Save model weights."""
        import safetensors.numpy as sf

        actor_params = {f"actor.{k}": np.array(v) for k, v in self.actor.parameters().items()}
        critic_params = {f"critic.{k}": np.array(v) for k, v in self.critic.parameters().items()}

        all_params = {**actor_params, **critic_params}
        sf.save_file(all_params, path)

    def load(self, path: str):
        """Load model weights."""
        import safetensors.numpy as sf

        params = sf.load_file(path)

        actor_params = {k.replace("actor.", ""): mx.array(v) for k, v in params.items() if k.startswith("actor.")}
        critic_params = {k.replace("critic.", ""): mx.array(v) for k, v in params.items() if k.startswith("critic.")}

        self.actor.update(actor_params)
        self.critic.update(critic_params)

        materialize_params(self.actor.parameters())
        materialize_params(self.critic.parameters())
