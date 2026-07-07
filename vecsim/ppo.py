from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import PPOConfig
from .environment import VehicularEdgeEnv


class ActorCritic(nn.Module):

    def __init__(
        self,
        observation_dim: int,
        critic_observation_dim: int,
        action_dim: int,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.critic_observation_dim = int(critic_observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.base_feature_dim = 7
        self.action_feature_dim = 3
        self.structured_actor = (
            observation_dim
            == self.base_feature_dim + self.action_feature_dim * action_dim
        )
        if self.structured_actor:
            self.action_actor = nn.Sequential(
                nn.Linear(
                    self.base_feature_dim + self.action_feature_dim, hidden_size
                ),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(observation_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, action_dim),
            )
        self.critic = nn.Sequential(
            nn.Linear(critic_observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(self._initialize)
        if self.structured_actor:
            nn.init.orthogonal_(self.action_actor[-1].weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def _actor_logits(self, observations: torch.Tensor):
        if not self.structured_actor:
            return self.actor(observations)
        batch_size = observations.shape[0]
        base = observations[:, : self.base_feature_dim]
        action_features = observations[:, self.base_feature_dim :].reshape(
            batch_size, self.action_dim, self.action_feature_dim
        )
        expanded_base = base.unsqueeze(1).expand(-1, self.action_dim, -1)
        action_inputs = torch.cat(
            (expanded_base, action_features), dim=-1
        )
        return self.action_actor(action_inputs).squeeze(-1)

    def distribution(self, observations: torch.Tensor, masks: torch.Tensor):
        logits = self._actor_logits(observations)
        if masks.dtype != torch.bool:
            masks = masks.bool()
        if torch.any(~torch.any(masks, dim=-1)):
            raise ValueError("Every action mask must contain at least one valid action")
        masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits)

    def act(
        self,
        observations: torch.Tensor,
        critic_observations: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ):
        distribution = self.distribution(observations, masks)
        actions = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        log_probabilities = distribution.log_prob(actions)
        values = self.critic(critic_observations).squeeze(-1)
        return actions, log_probabilities, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        critic_observations: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
    ):
        distribution = self.distribution(observations, masks)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(critic_observations).squeeze(-1),
        )


class PPOTrainer:
    def __init__(
        self,
        environment: VehicularEdgeEnv,
        config: PPOConfig,
        seed: int,
        device: Optional[str] = None,
    ):
        self.environment = environment
        self.config = config
        self.seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(
            environment.observation_dim,
            environment.critic_observation_dim,
            environment.action_dim,
            config.hidden_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def _tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def train(self, updates: Optional[int] = None):
        updates = int(updates if updates is not None else self.config.updates)
        if updates <= 0:
            raise ValueError("updates must be positive")
        observations, critic_observations, masks = self.environment.reset()
        history: List[Dict[str, float]] = []
        for update in range(updates):
            obs_buffer = []
            critic_buffer = []
            mask_buffer = []
            action_buffer = []
            log_probability_buffer = []
            value_buffer = []
            reward_buffer = []
            done_buffer = []

            for _ in range(self.config.rollout_steps):
                observation_tensor = self._tensor(observations)
                critic_tensor = self._tensor(critic_observations)
                mask_tensor = self._tensor(masks, torch.bool)
                with torch.no_grad():
                    actions, log_probabilities, values = self.model.act(
                        observation_tensor, critic_tensor, mask_tensor
                    )
                next_obs, next_critic, next_masks, rewards, done, _ = self.environment.step(
                    actions.cpu().numpy()
                )
                obs_buffer.append(observations)
                critic_buffer.append(critic_observations)
                mask_buffer.append(masks)
                action_buffer.append(actions.cpu().numpy())
                log_probability_buffer.append(log_probabilities.cpu().numpy())
                value_buffer.append(values.cpu().numpy())
                reward_buffer.append(rewards)
                done_buffer.append(
                    np.full(self.environment.scenario.clients, done, dtype=np.float32)
                )
                if done:
                    next_obs, next_critic, next_masks = self.environment.reset()
                observations, critic_observations, masks = next_obs, next_critic, next_masks

            with torch.no_grad():
                bootstrap = (
                    self.model.critic(self._tensor(critic_observations))
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
            raw_rewards_array = np.asarray(reward_buffer, dtype=np.float32)
            rewards_array = raw_rewards_array / self.config.reward_scale
            values_array = np.asarray(value_buffer, dtype=np.float32)
            dones_array = np.asarray(done_buffer, dtype=np.float32)
            advantages = np.zeros_like(rewards_array)
            generalized_advantage = np.zeros(self.environment.scenario.clients, dtype=np.float32)
            next_value = bootstrap
            for step in reversed(range(self.config.rollout_steps)):
                active = 1.0 - dones_array[step]
                delta = (
                    rewards_array[step]
                    + self.config.gamma * next_value * active
                    - values_array[step]
                )
                generalized_advantage = (
                    delta
                    + self.config.gamma
                    * self.config.gae_lambda
                    * active
                    * generalized_advantage
                )
                advantages[step] = generalized_advantage
                next_value = values_array[step]
            returns = advantages + values_array

            flat_obs = self._tensor(np.asarray(obs_buffer).reshape(-1, self.environment.observation_dim))
            flat_critic = self._tensor(
                np.asarray(critic_buffer).reshape(
                    -1, self.environment.critic_observation_dim
                )
            )
            flat_masks = self._tensor(
                np.asarray(mask_buffer).reshape(-1, self.environment.action_dim), torch.bool
            )
            flat_actions = self._tensor(
                np.asarray(action_buffer).reshape(-1), torch.long
            )
            flat_old_log_probabilities = self._tensor(
                np.asarray(log_probability_buffer).reshape(-1)
            )
            flat_returns = self._tensor(returns.reshape(-1))
            normalized_advantages = advantages.reshape(-1)
            normalized_advantages = (
                normalized_advantages - normalized_advantages.mean()
            ) / (normalized_advantages.std() + 1e-8)
            flat_advantages = self._tensor(normalized_advantages)

            sample_count = len(flat_actions)
            losses = []
            value_losses = []
            entropies = []
            for _ in range(self.config.epochs):
                order = np.random.permutation(sample_count)
                for start in range(0, sample_count, self.config.minibatch_size):
                    indices = self._tensor(
                        order[start : start + self.config.minibatch_size], torch.long
                    )
                    new_log_probabilities, entropy, predicted_values = (
                        self.model.evaluate_actions(
                            flat_obs[indices],
                            flat_critic[indices],
                            flat_masks[indices],
                            flat_actions[indices],
                        )
                    )
                    ratio = torch.exp(
                        new_log_probabilities - flat_old_log_probabilities[indices]
                    )
                    unclipped = ratio * flat_advantages[indices]
                    clipped = (
                        torch.clamp(
                            ratio,
                            1.0 - self.config.clip_ratio,
                            1.0 + self.config.clip_ratio,
                        )
                        * flat_advantages[indices]
                    )
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (
                        predicted_values - flat_returns[indices]
                    ).pow(2).mean()
                    entropy_mean = entropy.mean()
                    total_loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy_mean
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    losses.append(float(policy_loss.detach().cpu()))
                    value_losses.append(float(value_loss.detach().cpu()))
                    entropies.append(float(entropy_mean.detach().cpu()))

            history.append(
                {
                    "update": float(update + 1),
                    "mean_rollout_reward": float(np.mean(raw_rewards_array)),
                    "policy_loss": float(np.mean(losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "entropy": float(np.mean(entropies)),
                }
            )
        return history

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "observation_dim": self.model.observation_dim,
                "critic_observation_dim": self.model.critic_observation_dim,
                "action_dim": self.model.action_dim,
                "hidden_size": self.model.hidden_size,
                "ppo_config": asdict(self.config),
            },
            path,
        )


def load_actor_critic(
    path: Union[str, Path], device: Optional[str] = None
):
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(path), map_location=selected_device)
    model = ActorCritic(
        checkpoint["observation_dim"],
        checkpoint["critic_observation_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(selected_device)
    model.eval()
    return model
