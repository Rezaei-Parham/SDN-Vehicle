from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    clients: int
    providers: int
    edge_servers: int


@dataclass
class SimulationConfig:
    hex_radius_m: float = 140.0
    placement_interval_s: int = 20
    provider_speed_mps: float = 25.0
    provider_range_m: float = 360.0
    edge_range_m: float = 700.0
    relay_range_m: float = 260.0
    bandwidth_hz: float = 10.0e6
    tx_power_dbm: float = 23.0
    noise_dbm: float = -96.0
    local_cpu_hz: float = 0.8e9
    provider_cpu_hz: float = 12.0e9
    edge_cpu_hz: float = 25.0e9
    task_arrival_probability: float = 0.72
    task_deadline_min_s: float = 1.5
    task_deadline_max_s: float = 5.0
    max_queue: int = 12
    episode_steps: int = 160
    weather_change_probability: float = 0.06
    deadline_penalty: float = 4.0
    loss_penalty: float = 0.75
    delay_weight: float = 1.0
    blind_spot_threshold: float = 0.55


@dataclass
class PPOConfig:
    updates: int = 100
    rollout_steps: int = 96
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    reward_scale: float = 4.0


@dataclass
class ExperimentConfig:
    seed: int = 17
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    scenarios: Dict[str, ScenarioConfig] = field(
        default_factory=lambda: {
            "scenario_20": ScenarioConfig("scenario_20", 20, 3, 2),
            "scenario_50": ScenarioConfig("scenario_50", 50, 6, 4),
        }
    )

    def to_dict(self):
        return asdict(self)


def _merge_dataclass(instance: Any, values: Dict[str, Any]):
    known = {field_name for field_name in instance.__dataclass_fields__}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(instance).__name__}: {sorted(unknown)}")
    return type(instance)(**{**asdict(instance), **values})


def load_config(path: Optional[Union[str, Path]] = None):
    config = ExperimentConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"seed", "simulation", "ppo", "scenarios"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    simulation = _merge_dataclass(config.simulation, raw.get("simulation", {}))
    ppo = _merge_dataclass(config.ppo, raw.get("ppo", {}))
    scenarios_raw = raw.get("scenarios", {})
    scenarios = dict(config.scenarios)
    for name, values in scenarios_raw.items():
        base = scenarios.get(name, ScenarioConfig(name=name, clients=20, providers=3, edge_servers=2))
        values = dict(values)
        values.setdefault("name", name)
        scenarios[name] = _merge_dataclass(base, values)
    return ExperimentConfig(
        seed=int(raw.get("seed", config.seed)),
        simulation=simulation,
        ppo=ppo,
        scenarios=scenarios,
    )
