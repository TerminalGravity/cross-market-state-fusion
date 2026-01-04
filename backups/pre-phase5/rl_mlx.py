#!/usr/bin/env python3
"""
MLX-based PPO (Proximal Policy Optimization) strategy.

Uses Apple's MLX framework for proper automatic differentiation
instead of manual NumPy backprop.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base import Strategy, MarketState, Action


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class Actor(nn.Module):
    """Policy network: state -> action probabilities."""

    def __init__(self, input_dim: int = 18, hidden_size: int = 128, output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Returns action probabilities."""
        h = mx.tanh(self.fc1(x))
        h = mx.tanh(self.fc2(h))
        logits = self.fc3(h)
        probs = mx.softmax(logits, axis=-1)
        return probs


class Critic(nn.Module):
    """Value network: state -> expected return."""

    def __init__(self, input_dim: int = 18, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Returns value estimate."""
        h = mx.tanh(self.fc1(x))
        h = mx.tanh(self.fc2(h))
        value = self.fc3(h)
        return value


class RLStrategy(Strategy):
    """PPO-based strategy with actor-critic architecture using MLX."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_size: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,  # Shorter horizon for 15-min markets
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.10,  # Higher entropy to prevent premature convergence
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 512,  # Faster updates (4x more frequent)
        batch_size: int = 64,  # Smaller batches for smaller buffer
        n_epochs: int = 10,
        target_kl: float = 0.02,
    ):
        super().__init__("rl")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = 3  # BUY, HOLD, SELL (simplified)

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Networks
        self.actor = Actor(input_dim, hidden_size, self.output_dim)
        self.critic = Critic(input_dim, hidden_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(learning_rate=lr_actor)
        self.critic_optimizer = optim.Adam(learning_rate=lr_critic)

        # Experience buffer
        self.experiences: List[Experience] = []

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # For storing last action's log prob and value
        self._last_log_prob = 0.0
        self._last_value = 0.0

        # Eval networks on init
        mx.eval(self.actor.parameters(), self.critic.parameters())

    def act(self, state: MarketState) -> Action:
        """Select action using current policy."""
        features = state.to_features()
        features_mx = mx.array(features.reshape(1, -1))

        # Get action probabilities and value
        probs = self.actor(features_mx)
        value = self.critic(features_mx)

        # Eval to get values
        mx.eval(probs, value)

        probs_np = np.array(probs[0])
        value_np = float(np.array(value[0, 0]))

        if self.training:
            # Sample from distribution
            action_idx = np.random.choice(self.output_dim, p=probs_np)
        else:
            # Greedy
            action_idx = int(np.argmax(probs_np))

        # Store for experience collection
        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np

        return Action(action_idx)

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        """Store experience for training."""
        # Update running reward stats for normalization
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * (reward - self.reward_mean))
            / max(1, self.reward_count)
        )

        # Normalize reward
        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        exp = Experience(
            state=state.to_features(),
            action=action.value,
            reward=norm_reward,
            next_state=next_state.to_features(),
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self.experiences.append(exp)

        # Limit buffer size
        if len(self.experiences) > self.buffer_size:
            self.experiences = self.experiences[-self.buffer_size:]

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     dones: np.ndarray, next_value: float) -> tuple:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _clip_grad_norm(self, grads: dict, max_norm: float) -> dict:
        """Clip gradients by global norm."""
        # Compute global norm
        total_norm_sq = 0.0
        for key, val in grads.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    total_norm_sq += float(mx.sum(subval ** 2))
            else:
                total_norm_sq += float(mx.sum(val ** 2))

        total_norm = np.sqrt(total_norm_sq)
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            def scale_grad(g):
                if isinstance(g, dict):
                    return {k: scale_grad(v) for k, v in g.items()}
                return g * clip_coef

            grads = {k: scale_grad(v) for k, v in grads.items()}

        return grads

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO with proper MLX autograd."""
        if len(self.experiences) < self.buffer_size:
            return None

        # Convert experiences to arrays
        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        actions = np.array([e.action for e in self.experiences], dtype=np.int32)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array([e.log_prob for e in self.experiences], dtype=np.float32)
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Compute next value for GAE
        next_state_mx = mx.array(self.experiences[-1].next_state.reshape(1, -1))
        next_value = float(np.array(self.critic(next_state_mx)[0, 0]))

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to MLX arrays
        states_mx = mx.array(states)
        actions_mx = mx.array(actions)
        old_log_probs_mx = mx.array(old_log_probs)
        advantages_mx = mx.array(advantages.astype(np.float32))
        returns_mx = mx.array(returns.astype(np.float32))
        old_values_mx = mx.array(old_values)

        n_samples = len(self.experiences)
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)

            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = mx.array(indices[start:end].astype(np.int32))

                # Get batch using mx.take (MLX doesn't support numpy fancy indexing)
                batch_states = mx.take(states_mx, batch_idx, axis=0)
                batch_actions = mx.take(actions_mx, batch_idx, axis=0)
                batch_old_log_probs = mx.take(old_log_probs_mx, batch_idx, axis=0)
                batch_advantages = mx.take(advantages_mx, batch_idx, axis=0)
                batch_returns = mx.take(returns_mx, batch_idx, axis=0)
                batch_old_values = mx.take(old_values_mx, batch_idx, axis=0)

                # Define loss function for actor (takes model, not params)
                def actor_loss_fn(model):
                    probs = model(batch_states)

                    # Get log probs for taken actions
                    batch_size_local = batch_actions.shape[0]
                    action_indices = mx.arange(batch_size_local)
                    selected_probs = probs[action_indices, batch_actions]
                    log_probs = mx.log(selected_probs + 1e-8)

                    # PPO clipped objective
                    ratio = mx.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = mx.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -mx.mean(mx.minimum(surr1, surr2))

                    # Entropy bonus (encourages exploration)
                    entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
                    entropy_mean = mx.mean(entropy)
                    policy_loss = policy_loss - self.entropy_coef * entropy_mean

                    # Metrics
                    approx_kl = mx.mean(batch_old_log_probs - log_probs)
                    clip_frac = mx.mean(
                        ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).astype(mx.float32)
                    )

                    return policy_loss, (entropy_mean, approx_kl, clip_frac)

                # Define loss function for critic (takes model, not params)
                def critic_loss_fn(model):
                    values = model(batch_states).squeeze()

                    # Value loss with clipping (PPO2 style)
                    values_clipped = batch_old_values + mx.clip(
                        values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss1 = (batch_returns - values) ** 2
                    value_loss2 = (batch_returns - values_clipped) ** 2
                    value_loss = 0.5 * mx.mean(mx.maximum(value_loss1, value_loss2))

                    return value_loss

                # Compute actor gradients and update
                actor_loss_and_grad = nn.value_and_grad(self.actor, actor_loss_fn)
                (actor_loss, (entropy, approx_kl, clip_frac)), actor_grads = actor_loss_and_grad(self.actor)

                # Clip actor gradients
                actor_grads = self._clip_grad_norm(actor_grads, self.max_grad_norm)

                # Update actor
                self.actor_optimizer.update(self.actor, actor_grads)

                # Compute critic gradients and update
                critic_loss_and_grad = nn.value_and_grad(self.critic, critic_loss_fn)
                critic_loss, critic_grads = critic_loss_and_grad(self.critic)

                # Clip critic gradients
                critic_grads = self._clip_grad_norm(critic_grads, self.max_grad_norm)

                # Update critic
                self.critic_optimizer.update(self.critic, critic_grads)

                # Eval to commit updates (include optimizer state)
                mx.eval(
                    self.actor.parameters(),
                    self.critic.parameters(),
                    self.actor_optimizer.state,
                    self.critic_optimizer.state,
                )

                # Record metrics
                all_metrics["policy_loss"].append(float(np.array(actor_loss)))
                all_metrics["value_loss"].append(float(np.array(critic_loss)))
                all_metrics["entropy"].append(float(np.array(entropy)))
                all_metrics["approx_kl"].append(float(np.array(approx_kl)))
                all_metrics["clip_fraction"].append(float(np.array(clip_frac)))

                epoch_kl += float(np.array(approx_kl))
                n_batches += 1

            # Early stopping on KL divergence
            avg_kl = epoch_kl / max(1, n_batches)
            if avg_kl > self.target_kl:
                print(f"  [RL] Early stop epoch {epoch}, KL={avg_kl:.4f}")
                break

        # Clear buffer after update
        self.experiences.clear()

        # Compute explained variance
        y_pred = old_values
        y_true = returns
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0

        return {
            "policy_loss": np.mean(all_metrics["policy_loss"]),
            "value_loss": np.mean(all_metrics["value_loss"]),
            "entropy": np.mean(all_metrics["entropy"]),
            "approx_kl": np.mean(all_metrics["approx_kl"]),
            "clip_fraction": np.mean(all_metrics["clip_fraction"]),
            "explained_variance": explained_var,
        }

    def reset(self):
        """Clear experience buffer."""
        self.experiences.clear()

    def save(self, path: str):
        """Save model and training state."""
        # Flatten params for saving
        def flatten_params(params, prefix=""):
            result = {}
            for key, val in params.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(val, dict):
                    result.update(flatten_params(val, f"{full_key}."))
                else:
                    result[full_key] = val
            return result

        actor_flat = flatten_params(self.actor.parameters(), "actor.")
        critic_flat = flatten_params(self.critic.parameters(), "critic.")

        # Save weights using MLX safetensors
        weights = {**actor_flat, **critic_flat}
        weights_path = path.replace(".npz", "") + ".safetensors"
        mx.save_safetensors(weights_path, weights)

        # Save stats separately
        stats_path = path.replace(".npz", "") + "_stats.npz"
        np.savez(
            stats_path,
            reward_mean=self.reward_mean,
            reward_std=self.reward_std,
            reward_count=self.reward_count,
        )

    def load(self, path: str):
        """Load model and training state."""
        # Load weights
        weights_path = path.replace(".npz", "") + ".safetensors"
        weights = mx.load(weights_path)

        # Unflatten and load actor params
        def unflatten_params(flat_dict, prefix, template):
            result = {}
            for key, val in template.items():
                full_key = f"{prefix}{key}"
                if isinstance(val, dict):
                    result[key] = unflatten_params(flat_dict, f"{full_key}.", val)
                else:
                    result[key] = flat_dict[full_key]
            return result

        actor_params = unflatten_params(weights, "actor.", self.actor.parameters())
        critic_params = unflatten_params(weights, "critic.", self.critic.parameters())

        self.actor.update(actor_params)
        self.critic.update(critic_params)

        # Load stats
        stats_path = path.replace(".npz", "") + "_stats.npz"
        stats = np.load(stats_path)
        self.reward_mean = float(stats["reward_mean"])
        self.reward_std = float(stats["reward_std"])
        self.reward_count = int(stats["reward_count"])

        # Eval to commit
        mx.eval(self.actor.parameters(), self.critic.parameters())
