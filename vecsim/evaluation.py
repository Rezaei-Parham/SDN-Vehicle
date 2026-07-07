from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import ScenarioConfig, SimulationConfig
from .environment import VehicularEdgeEnv
from .mobility import TrafficTrace
from .ppo import ActorCritic


class GreedyOffloader:

    def select(self, environment: VehicularEdgeEnv):
        actions = np.zeros(environment.scenario.clients, dtype=np.int64)
        for client_index, vehicle_id in enumerate(environment.current_frame.ids):
            queue = environment.queues[vehicle_id]
            if not queue:
                continue
            task = queue[0]
            scores = np.full(environment.action_dim, np.inf)
            scores[0] = task.cycles / environment.config.local_cpu_hz
            for action in np.flatnonzero(environment.last_masks[client_index, 1:]) + 1:
                spec = environment.action_specs[action]
                transmission = environment._transmission_delay(client_index, action, task)
                if spec.destination == "provider":
                    compute = task.cycles / environment.config.provider_cpu_hz
                else:
                    compute = task.cycles / environment.config.edge_cpu_hz
                estimated_loss = environment.estimated_action_loss[client_index, action]
                expected_retries = 1.0 / max(1.0 - estimated_loss, 0.05)
                scores[action] = expected_retries * transmission + compute + 1.5 * estimated_loss
            actions[client_index] = int(np.argmin(scores))
        return actions


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    metric_std: Dict[str, float]
    provider_history: np.ndarray
    average_provider_locations: np.ndarray


def evaluate(
    trace: TrafficTrace,
    scenario: ScenarioConfig,
    simulation: SimulationConfig,
    episodes: int,
    seed: int,
    placement_mode: str,
    model: Optional[ActorCritic] = None,
):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    per_episode: List[Dict[str, float]] = []
    histories = []
    greedy = GreedyOffloader()
    max_start = len(trace.frames) - simulation.episode_steps - 1
    for episode in range(episodes):
        environment = VehicularEdgeEnv(
            trace,
            scenario,
            simulation,
            seed=seed + episode * 101,
            placement_mode=placement_mode,
        )
        start = int(round(episode * max_start / max(1, episodes - 1)))
        observations, critic_observations, masks = environment.reset(start_index=start)
        done = False
        info: Dict[str, float] = {}
        reward_sum = 0.0
        reward_count = 0
        invalid_actions = 0.0
        while not done:
            if model is None:
                actions = greedy.select(environment)
            else:
                device = next(model.parameters()).device
                with torch.no_grad():
                    actions_tensor, _, _ = model.act(
                        torch.as_tensor(observations, dtype=torch.float32, device=device),
                        torch.as_tensor(
                            critic_observations, dtype=torch.float32, device=device
                        ),
                        torch.as_tensor(masks, dtype=torch.bool, device=device),
                        deterministic=True,
                    )
                actions = actions_tensor.cpu().numpy()
            (
                observations,
                critic_observations,
                masks,
                rewards,
                done,
                info,
            ) = environment.step(actions)
            reward_sum += float(np.sum(rewards))
            reward_count += len(rewards)
            invalid_actions += info["invalid_actions"]
        info["mean_reward"] = reward_sum / max(1, reward_count)
        info["invalid_actions"] = invalid_actions
        per_episode.append(info)
        histories.append(np.asarray(environment.provider_history))
    metric_names = sorted(per_episode[0])
    metrics = {
        name: float(np.mean([episode[name] for episode in per_episode]))
        for name in metric_names
    }
    metric_std = {
        name: float(np.std([episode[name] for episode in per_episode]))
        for name in metric_names
    }
    provider_history = np.stack(histories)
    return EvaluationResult(
        metrics=metrics,
        metric_std=metric_std,
        provider_history=provider_history,
        average_provider_locations=np.mean(provider_history, axis=(0, 1)),
    )
