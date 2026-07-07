from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from .config import ScenarioConfig, SimulationConfig
from .hexgrid import HexGrid
from .kalman import LinkKalmanBank
from .mobility import ClientFrame, TrafficTrace
from .network import LinkModel, SpatialRiskMap, Task, sample_task
from .placement import (
    ConvexProviderPlacer,
    RandomProviderPlacer,
    fixed_edge_positions,
    move_toward_targets,
)


@dataclass(frozen=True)
class ActionSpec:
    destination: str
    destination_index: int
    path: str


@dataclass
class NetworkMetrics:
    transmission_attempts: int = 0
    packet_losses: int = 0
    completed_tasks: int = 0
    total_delay_s: float = 0.0
    missed_deadlines: int = 0
    blind_spots: int = 0
    generated_tasks: int = 0
    queue_drops: int = 0

    def as_dict(self, steps: int):
        return {
            "packet_loss_ratio": self.packet_losses / max(1, self.transmission_attempts),
            "average_delay_s": self.total_delay_s / max(1, self.completed_tasks),
            "missed_deadlines": float(self.missed_deadlines),
            "blind_spot_occurrences": float(self.blind_spots),
            "blind_spots_per_step": self.blind_spots / max(1, steps),
            "completed_tasks": float(self.completed_tasks),
            "generated_tasks": float(self.generated_tasks),
            "queue_drops": float(self.queue_drops),
        }


class VehicularEdgeEnv:

    BASE_FEATURE_DIM = 7
    ACTION_FEATURE_DIM = 3

    def __init__(
        self,
        trace: TrafficTrace,
        scenario: ScenarioConfig,
        config: SimulationConfig,
        seed: int = 0,
        placement_mode: str = "convex",
    ):
        trace.validate_scenario(scenario.clients, config.episode_steps)
        if placement_mode not in {"convex", "random"}:
            raise ValueError("placement_mode must be 'convex' or 'random'")
        self.trace = trace
        self.scenario = scenario
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.task_rng = np.random.default_rng(seed + 11)
        self.weather_rng = np.random.default_rng(seed + 23)
        self.measurement_rng = np.random.default_rng(seed + 37)
        self.event_rng = np.random.default_rng(seed + 53)
        self.grid = HexGrid.from_bounds(trace.bounds, config.hex_radius_m)
        self.risk_map = SpatialRiskMap(trace.bounds, seed=seed + 1009)
        self.link_model = LinkModel(config, self.risk_map)
        self.edge_positions = fixed_edge_positions(self.grid, scenario.edge_servers)
        if placement_mode == "convex":
            self.placer = ConvexProviderPlacer(
                grid=self.grid,
                provider_range=config.provider_range_m,
                provider_speed=config.provider_speed_mps,
                placement_interval=config.placement_interval_s,
                risk_map=self.risk_map,
            )
        else:
            self.placer = RandomProviderPlacer(
                grid=self.grid,
                provider_speed=config.provider_speed_mps,
                placement_interval=config.placement_interval_s,
                rng=np.random.default_rng(seed + 3001),
            )
        self.placement_mode = placement_mode
        self.action_specs = self._make_action_specs()
        self.action_dim = len(self.action_specs)
        self.observation_dim = (
            self.BASE_FEATURE_DIM + self.ACTION_FEATURE_DIM * self.action_dim
        )
        self.central_dim = 4 + scenario.providers + scenario.edge_servers
        self.critic_observation_dim = self.observation_dim + self.central_dim

        self.queues: Dict[str, Deque[Task]] = {}
        self.kalman = LinkKalmanBank()
        self.metrics = NetworkMetrics()
        self.provider_history: list[np.ndarray] = []
        self.current_frame: ClientFrame
        self.provider_positions: np.ndarray
        self.provider_targets: np.ndarray
        self.provider_loads = np.zeros(scenario.providers, dtype=np.float64)
        self.edge_loads = np.zeros(scenario.edge_servers, dtype=np.float64)
        self.weather = 0.0
        self.start_index = 0
        self.elapsed_steps = 0
        self.true_action_loss = np.zeros((scenario.clients, self.action_dim))
        self.estimated_action_loss = np.zeros_like(self.true_action_loss)
        self.path_distances: list[list[Tuple[float, ...]]] = []
        self.last_masks = np.zeros((scenario.clients, self.action_dim), dtype=bool)
        self.last_connectivity_masks = self.last_masks.copy()

    def _make_action_specs(self):
        actions = [ActionSpec("local", 0, "local")]
        for index in range(self.scenario.providers):
            actions.append(ActionSpec("provider", index, "direct"))
            actions.append(ActionSpec("provider", index, "vehicle_relay"))
        for index in range(self.scenario.edge_servers):
            actions.append(ActionSpec("edge", index, "direct"))
            actions.append(ActionSpec("edge", index, "provider_relay"))
        return tuple(actions)

    def reset(
        self, start_index: Optional[int] = None
    ):
        max_start = len(self.trace.frames) - self.config.episode_steps - 1
        if start_index is None:
            start_index = int(self.rng.integers(0, max_start + 1))
        if not 0 <= start_index <= max_start:
            raise ValueError(f"start_index must be between 0 and {max_start}")
        self.start_index = int(start_index)
        self.elapsed_steps = 0
        self.current_frame = self.trace.client_frame(
            self.start_index, self.scenario.clients, previous_ids=()
        )
        self.queues = {vehicle_id: deque() for vehicle_id in self.current_frame.ids}
        self.kalman = LinkKalmanBank()
        self.metrics = NetworkMetrics()
        self.weather = float(
            self.weather_rng.choice((0.0, 0.5, 1.0), p=(0.65, 0.25, 0.10))
        )
        self.provider_loads.fill(0.0)
        self.edge_loads.fill(0.0)
        self._enqueue_arrivals()
        demand = self._placement_demand()
        self.provider_positions = self.placer.choose_targets(
            self.scenario.providers,
            self.current_frame.positions,
            demand,
            self.weather,
            current_positions=None,
        )
        self.provider_targets = self.provider_positions.copy()
        self.provider_history = [self.provider_positions.copy()]
        return self._observations()

    def _enqueue_arrivals(self):
        for vehicle_id in self.current_frame.ids:
            queue = self.queues.setdefault(vehicle_id, deque())
            if self.task_rng.random() < self.config.task_arrival_probability:
                self.metrics.generated_tasks += 1
                if len(queue) >= self.config.max_queue:
                    self.metrics.queue_drops += 1
                    self.metrics.missed_deadlines += 1
                else:
                    queue.append(
                        sample_task(
                            self.task_rng, self.current_frame.time, self.config
                        )
                    )

    def _placement_demand(self):
        demand = np.empty(self.scenario.clients, dtype=np.float64)
        for index, vehicle_id in enumerate(self.current_frame.ids):
            queue = self.queues.get(vehicle_id, ())
            head_cycles = queue[0].cycles / 2.2e9 if queue else 0.0
            demand[index] = 1.0 + len(queue) / self.config.max_queue + head_cycles
        return demand

    def _calculate_paths(self):
        clients = self.scenario.clients
        masks = np.zeros((clients, self.action_dim), dtype=bool)
        masks[:, 0] = True
        self.true_action_loss.fill(0.0)
        self.path_distances = [[() for _ in range(self.action_dim)] for _ in range(clients)]
        positions = self.current_frame.positions
        speeds = self.current_frame.speeds
        providers = self.provider_positions
        edges = self.edge_positions
        provider_count = self.scenario.providers
        edge_count = self.scenario.edge_servers

        client_provider_distance = np.linalg.norm(
            positions[:, None, :] - providers[None, :, :], axis=2
        )
        client_provider_loss = self.link_model.packet_loss(
            np.repeat(positions, provider_count, axis=0),
            np.tile(providers, (clients, 1)),
            self.config.provider_range_m,
            self.weather,
            np.repeat(speeds, provider_count),
        ).reshape(clients, provider_count)
        client_edge_distance = np.linalg.norm(
            positions[:, None, :] - edges[None, :, :], axis=2
        )
        client_edge_loss = self.link_model.packet_loss(
            np.repeat(positions, edge_count, axis=0),
            np.tile(edges, (clients, 1)),
            self.config.edge_range_m,
            self.weather,
            np.repeat(speeds, edge_count),
        ).reshape(clients, edge_count)

        client_client_distance = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        client_client_loss = self.link_model.packet_loss(
            np.repeat(positions, clients, axis=0),
            np.tile(positions, (clients, 1)),
            self.config.relay_range_m,
            self.weather,
            np.repeat(speeds, clients),
        ).reshape(clients, clients)
        relay_valid = (
            (client_client_distance[:, :, None] <= self.config.relay_range_m)
            & (
                client_provider_distance[None, :, :]
                <= self.config.provider_range_m
            )
            & (~np.eye(clients, dtype=bool)[:, :, None])
        )
        relay_loss = 1.0 - (
            (1.0 - client_client_loss[:, :, None])
            * (1.0 - client_provider_loss[None, :, :])
        )
        relay_loss = np.where(relay_valid, relay_loss, np.inf)
        relay_cost = (
            client_client_distance[:, :, None] / self.config.relay_range_m
            + client_provider_distance[None, :, :] / self.config.provider_range_m
        )
        relay_cost = np.where(relay_valid, relay_cost, np.inf)
        best_vehicle_relay = np.argmin(relay_cost, axis=1)
        best_vehicle_cost = np.min(relay_cost, axis=1)
        has_vehicle_relay = np.isfinite(best_vehicle_cost)

        provider_edge_distance = np.linalg.norm(
            providers[:, None, :] - edges[None, :, :], axis=2
        )
        provider_edge_loss = self.link_model.packet_loss(
            np.repeat(providers, edge_count, axis=0),
            np.tile(edges, (provider_count, 1)),
            self.config.edge_range_m,
            self.weather,
            np.full(provider_count * edge_count, self.config.provider_speed_mps),
        ).reshape(provider_count, edge_count)
        provider_relay_valid = (
            (client_provider_distance[:, :, None] <= self.config.provider_range_m)
            & (provider_edge_distance[None, :, :] <= self.config.edge_range_m)
        )
        provider_relay_loss = 1.0 - (
            (1.0 - client_provider_loss[:, :, None])
            * (1.0 - provider_edge_loss[None, :, :])
        )
        provider_relay_loss = np.where(provider_relay_valid, provider_relay_loss, np.inf)
        provider_relay_cost = (
            client_provider_distance[:, :, None] / self.config.provider_range_m
            + provider_edge_distance[None, :, :] / self.config.edge_range_m
        )
        provider_relay_cost = np.where(
            provider_relay_valid, provider_relay_cost, np.inf
        )
        best_provider_relay = np.argmin(provider_relay_cost, axis=1)
        best_provider_cost = np.min(provider_relay_cost, axis=1)
        has_provider_relay = np.isfinite(best_provider_cost)

        for client_index in range(clients):
            action = 1
            for provider_index in range(provider_count):
                distance = client_provider_distance[client_index, provider_index]
                direct_valid = distance <= self.config.provider_range_m
                masks[client_index, action] = direct_valid
                self.true_action_loss[client_index, action] = client_provider_loss[
                    client_index, provider_index
                ]
                if direct_valid:
                    self.path_distances[client_index][action] = (float(distance),)
                action += 1

                relay_index = best_vehicle_relay[client_index, provider_index]
                valid = has_vehicle_relay[client_index, provider_index]
                masks[client_index, action] = valid
                if valid:
                    self.true_action_loss[client_index, action] = relay_loss[
                        client_index, relay_index, provider_index
                    ]
                    self.path_distances[client_index][action] = (
                        float(client_client_distance[client_index, relay_index]),
                        float(client_provider_distance[relay_index, provider_index]),
                    )
                action += 1

            for edge_index in range(edge_count):
                distance = client_edge_distance[client_index, edge_index]
                direct_valid = distance <= self.config.edge_range_m
                masks[client_index, action] = direct_valid
                self.true_action_loss[client_index, action] = client_edge_loss[
                    client_index, edge_index
                ]
                if direct_valid:
                    self.path_distances[client_index][action] = (float(distance),)
                action += 1

                relay_index = best_provider_relay[client_index, edge_index]
                valid = has_provider_relay[client_index, edge_index]
                masks[client_index, action] = valid
                if valid:
                    self.true_action_loss[client_index, action] = provider_relay_loss[
                        client_index, relay_index, edge_index
                    ]
                    self.path_distances[client_index][action] = (
                        float(client_provider_distance[client_index, relay_index]),
                        float(provider_edge_distance[relay_index, edge_index]),
                    )
                action += 1
        return masks

    def _observations(self):
        connectivity_masks = self._calculate_paths()
        masks = connectivity_masks.copy()
        observations = np.zeros(
            (self.scenario.clients, self.observation_dim), dtype=np.float32
        )
        for client_index, vehicle_id in enumerate(self.current_frame.ids):
            queue = self.queues[vehicle_id]
            if not queue:
                masks[client_index, 1:] = False
            now = self.current_frame.time
            if queue:
                task = queue[0]
                age = max(0.0, now - task.created_at)
                remaining = max(0.0, task.deadline_s - age)
                deadline = max(task.deadline_s, 0.1)
                task_features = (
                    task.data_bits / 4.0e6,
                    task.cycles / 2.2e9,
                    np.clip(remaining / deadline, 0.0, 1.0),
                    np.clip(age / deadline, 0.0, 2.0),
                )
            else:
                task = None
                deadline = 1.0
                task_features = (0.0, 0.0, 0.0, 0.0)
            base = [
                len(queue) / self.config.max_queue,
                *task_features,
                min(self.current_frame.speeds[client_index] / 35.0, 1.5),
                self.weather,
            ]
            observations[
                client_index, : self.BASE_FEATURE_DIM
            ] = np.asarray(base, dtype=np.float32)
            feature_index = self.BASE_FEATURE_DIM
            for action, spec in enumerate(self.action_specs):
                if task is None:
                    loss_estimate = 0.0 if action == 0 else 1.0
                    transmission_delay = 0.0
                    compute_delay = 0.0
                elif action == 0:
                    loss_estimate = 0.0
                    transmission_delay = 0.0
                    compute_delay = task.cycles / self.config.local_cpu_hz
                elif connectivity_masks[client_index, action]:
                    true_loss = self.true_action_loss[client_index, action]
                    measured = float(
                        self.link_model.noisy_measurement(
                            np.asarray([true_loss]), self.measurement_rng
                        )[0]
                    )
                    loss_estimate = self.kalman.estimate(
                        (vehicle_id, action, spec.destination_index),
                        measured,
                    )
                    transmission_delay = self._transmission_delay(
                        client_index, action, task
                    )
                    if spec.destination == "provider":
                        load = max(
                            1.0, self.provider_loads[spec.destination_index]
                        )
                        compute_delay = (
                            task.cycles * load / self.config.provider_cpu_hz
                        )
                    else:
                        load = max(1.0, self.edge_loads[spec.destination_index])
                        compute_delay = task.cycles * load / self.config.edge_cpu_hz
                else:
                    loss_estimate = 1.0
                    transmission_delay = 4.0 * deadline
                    compute_delay = 4.0 * deadline

                self.estimated_action_loss[client_index, action] = loss_estimate
                observations[
                    client_index, feature_index : feature_index + 3
                ] = (
                    loss_estimate,
                    min(transmission_delay / deadline, 4.0),
                    min(compute_delay / deadline, 4.0),
                )
                feature_index += self.ACTION_FEATURE_DIM
        self.last_masks = masks
        self.last_connectivity_masks = connectivity_masks
        central = self._central_state()
        critic = np.concatenate(
            (observations, np.broadcast_to(central, (self.scenario.clients, len(central)))),
            axis=1,
        ).astype(np.float32)
        return observations, critic, masks

    def _central_state(self):
        mean_queue = np.mean([len(self.queues[vehicle_id]) for vehicle_id in self.current_frame.ids])
        area_km2 = np.prod(self.trace.bounds[1] - self.trace.bounds[0]) / 1.0e6
        density = self.scenario.clients / max(area_km2, 1e-9)
        time_fraction = self.elapsed_steps / max(1, self.config.episode_steps)
        return np.asarray(
            [
                mean_queue / self.config.max_queue,
                min(density / 100.0, 2.0),
                self.weather,
                time_fraction,
                *(
                    self.provider_loads
                    / max(1.0, self.scenario.clients / self.scenario.providers)
                ),
                *(
                    self.edge_loads
                    / max(1.0, self.scenario.clients / self.scenario.edge_servers)
                ),
            ],
            dtype=np.float32,
        )

    def _transmission_delay(self, client_index: int, action: int, task: Task):
        distances = self.path_distances[client_index][action]
        spec = self.action_specs[action]
        if not distances:
            return 1.0
        if len(distances) == 1:
            communication_range = (
                self.config.provider_range_m
                if spec.destination == "provider"
                else self.config.edge_range_m
            )
            return task.data_bits / self.link_model.data_rate_bps(
                distances[0], communication_range, self.weather
            )
        first_range = (
            self.config.relay_range_m
            if spec.path == "vehicle_relay"
            else self.config.provider_range_m
        )
        second_range = (
            self.config.provider_range_m
            if spec.destination == "provider"
            else self.config.edge_range_m
        )
        return sum(
            (
                task.data_bits
                / self.link_model.data_rate_bps(distances[0], first_range, self.weather),
                task.data_bits
                / self.link_model.data_rate_bps(distances[1], second_range, self.weather),
            )
        )

    def step(
        self, actions: Sequence[int]
    ):
        actions_array = np.asarray(actions, dtype=np.int64)
        if actions_array.shape != (self.scenario.clients,):
            raise ValueError(f"actions must have shape ({self.scenario.clients},)")
        valid_indices = (actions_array >= 0) & (actions_array < self.action_dim)
        valid_masks = np.zeros_like(valid_indices)
        valid_masks[valid_indices] = self.last_masks[
            np.flatnonzero(valid_indices), actions_array[valid_indices]
        ]
        invalid = ~valid_masks
        actions_array[invalid] = 0
        rewards = np.zeros(self.scenario.clients, dtype=np.float32)
        rewards[invalid] -= 1.0

        successful: list[Tuple[int, int, Task, float]] = []
        provider_counts = np.zeros(self.scenario.providers, dtype=np.float64)
        edge_counts = np.zeros(self.scenario.edge_servers, dtype=np.float64)
        now = self.current_frame.time
        active_remote = np.any(self.last_connectivity_masks[:, 1:], axis=1)
        self.metrics.blind_spots += int(np.count_nonzero(~active_remote))

        for client_index, (vehicle_id, action) in enumerate(
            zip(self.current_frame.ids, actions_array)
        ):
            queue = self.queues[vehicle_id]
            if not queue:
                continue
            task = queue[0]
            spec = self.action_specs[int(action)]
            if spec.destination == "local":
                successful.append((client_index, int(action), task, 0.0))
                continue
            self.metrics.transmission_attempts += 1
            transmission_delay = self._transmission_delay(client_index, int(action), task)
            loss = self.true_action_loss[client_index, int(action)]
            rewards[client_index] -= (
                self.config.loss_penalty
                * self.estimated_action_loss[client_index, int(action)]
            )
            if self.event_rng.random() < loss:
                self.metrics.packet_losses += 1
                age_after_step = now + 1.0 - task.created_at
                rewards[client_index] -= (
                    self.config.delay_weight / max(task.deadline_s, 0.1)
                )
                if age_after_step >= task.deadline_s:
                    queue.popleft()
                    self.metrics.missed_deadlines += 1
                    rewards[client_index] -= self.config.deadline_penalty
                continue
            successful.append((client_index, int(action), task, transmission_delay))
            if spec.destination == "provider":
                provider_counts[spec.destination_index] += 1.0
            else:
                edge_counts[spec.destination_index] += 1.0

        for client_index, action, task, transmission_delay in successful:
            vehicle_id = self.current_frame.ids[client_index]
            queue = self.queues[vehicle_id]
            spec = self.action_specs[action]
            if spec.destination == "local":
                compute_delay = task.cycles / self.config.local_cpu_hz
            elif spec.destination == "provider":
                compute_delay = (
                    task.cycles
                    * max(1.0, provider_counts[spec.destination_index])
                    / self.config.provider_cpu_hz
                )
            else:
                compute_delay = (
                    task.cycles
                    * max(1.0, edge_counts[spec.destination_index])
                    / self.config.edge_cpu_hz
                )
            delay = max(0.0, now - task.created_at) + transmission_delay + compute_delay
            if queue and queue[0] is task:
                queue.popleft()
            self.metrics.completed_tasks += 1
            self.metrics.total_delay_s += delay
            normalized_delay = delay / max(task.deadline_s, 0.1)
            rewards[client_index] -= self.config.delay_weight * min(normalized_delay, 4.0)
            if delay > task.deadline_s:
                self.metrics.missed_deadlines += 1
                rewards[client_index] -= self.config.deadline_penalty

        self.provider_loads = provider_counts
        self.edge_loads = edge_counts
        self._advance()
        done = (
            self.elapsed_steps >= self.config.episode_steps
            or self.start_index + self.elapsed_steps >= len(self.trace.frames) - 1
        )
        info = self.metrics.as_dict(self.elapsed_steps)
        info["mean_reward"] = float(np.mean(rewards))
        info["invalid_actions"] = float(np.count_nonzero(invalid))
        if done:
            observations = np.zeros(
                (self.scenario.clients, self.observation_dim), dtype=np.float32
            )
            critic = np.zeros(
                (self.scenario.clients, self.critic_observation_dim), dtype=np.float32
            )
            masks = np.zeros((self.scenario.clients, self.action_dim), dtype=bool)
            masks[:, 0] = True
        else:
            observations, critic, masks = self._observations()
        return observations, critic, masks, rewards, done, info

    def _advance(self):
        self.elapsed_steps += 1
        self.provider_positions = move_toward_targets(
            self.provider_positions,
            self.provider_targets,
            self.config.provider_speed_mps,
        )
        self.provider_history.append(self.provider_positions.copy())
        if (
            self.elapsed_steps >= self.config.episode_steps
            or self.start_index + self.elapsed_steps >= len(self.trace.frames) - 1
        ):
            return
        next_index = min(
            self.start_index + self.elapsed_steps, len(self.trace.frames) - 1
        )
        previous_ids = self.current_frame.ids
        self.current_frame = self.trace.client_frame(
            next_index, self.scenario.clients, previous_ids
        )
        active = set(self.current_frame.ids)
        departed_tasks = sum(
            len(self.queues.get(vehicle_id, ()))
            for vehicle_id in previous_ids
            if vehicle_id not in active
        )
        if departed_tasks:
            self.metrics.queue_drops += departed_tasks
            self.metrics.missed_deadlines += departed_tasks
        self.queues = {
            vehicle_id: self.queues.get(vehicle_id, deque())
            for vehicle_id in self.current_frame.ids
        }
        self.kalman.discard_vehicles(active)
        self._drop_expired_tasks()
        if self.weather_rng.random() < self.config.weather_change_probability:
            direction = int(self.weather_rng.choice((-1, 1)))
            self.weather = float(np.clip(self.weather + 0.5 * direction, 0.0, 1.0))
        self._enqueue_arrivals()
        if self.elapsed_steps % self.config.placement_interval_s == 0:
            self.provider_targets = self.placer.choose_targets(
                self.scenario.providers,
                self.current_frame.positions,
                self._placement_demand(),
                self.weather,
                self.provider_positions,
            )

    def _drop_expired_tasks(self):
        now = self.current_frame.time
        for queue in self.queues.values():
            retained = deque(
                task for task in queue if now - task.created_at < task.deadline_s
            )
            expired = len(queue) - len(retained)
            if expired:
                self.metrics.missed_deadlines += expired
                queue.clear()
                queue.extend(retained)
