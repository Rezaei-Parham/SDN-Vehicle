from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .config import SimulationConfig


@dataclass(frozen=True)
class Task:
    data_bits: float
    cycles: float
    deadline_s: float
    created_at: float


class SpatialRiskMap:

    def __init__(self, bounds: np.ndarray, seed: int, hot_spots: int = 6):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        rng = np.random.default_rng(seed)
        span = self.bounds[1] - self.bounds[0]
        self.centers = rng.uniform(self.bounds[0] + 0.08 * span, self.bounds[1] - 0.08 * span, (hot_spots, 2))
        self.widths = rng.uniform(0.07, 0.18, (hot_spots, 2)) * span
        self.amplitudes = rng.uniform(0.25, 0.75, hot_spots)

    def risk(self, positions: np.ndarray):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        scaled = (positions[:, None, :] - self.centers[None, :, :]) / self.widths[None, :, :]
        fields = self.amplitudes[None, :] * np.exp(-0.5 * np.sum(scaled * scaled, axis=2))
        return np.clip(np.max(fields, axis=1), 0.0, 1.0)


class LinkModel:
    def __init__(self, config: SimulationConfig, risk_map: SpatialRiskMap):
        self.config = config
        self.risk_map = risk_map

    def packet_loss(
        self,
        transmitters: np.ndarray,
        receivers: np.ndarray,
        communication_range: float,
        weather: float,
        speed: Union[np.ndarray, float] = 0.0,
    ):
        transmitters = np.asarray(transmitters, dtype=np.float64).reshape(-1, 2)
        receivers = np.asarray(receivers, dtype=np.float64).reshape(-1, 2)
        distance = np.linalg.norm(transmitters - receivers, axis=1)
        midpoint = 0.5 * (transmitters + receivers)
        spatial = self.risk_map.risk(midpoint)
        speed_array = np.broadcast_to(np.asarray(speed, dtype=np.float64), distance.shape)
        logit = (
            -4.2
            + 5.7 * distance / max(communication_range, 1e-6)
            + 1.15 * weather
            + 1.65 * spatial
            + 0.010 * speed_array
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
        probability = np.where(distance <= communication_range, probability, 1.0)
        return np.clip(probability, 0.005, 1.0)

    def noisy_measurement(self, true_loss: np.ndarray, rng: np.random.Generator):
        noise = rng.normal(0.0, 0.08, np.asarray(true_loss).shape)
        return np.clip(np.asarray(true_loss) + noise, 0.0, 1.0)

    def data_rate_bps(self, distance: float, communication_range: float, weather: float):
        # A bounded link-budget approximation used only inside the communication range.
        normalized = max(distance / max(communication_range, 1e-6), 0.02)
        snr_db = 25.0 - 24.0 * np.log10(1.0 + 8.0 * normalized) - 5.0 * weather
        snr_linear = 10.0 ** (snr_db / 10.0)
        return float(max(1.0e5, self.config.bandwidth_hz * np.log2(1.0 + snr_linear)))


def sample_task(
    rng: np.random.Generator, created_at: float, config: SimulationConfig
):
    if not 0.0 < config.task_deadline_min_s <= config.task_deadline_max_s:
        raise ValueError("Task deadline bounds must satisfy 0 < min <= max")
    return Task(
        data_bits=float(rng.uniform(0.6e6, 4.0e6)),
        cycles=float(rng.uniform(0.4e9, 2.2e9)),
        deadline_s=float(
            rng.uniform(config.task_deadline_min_s, config.task_deadline_max_s)
        ),
        created_at=float(created_at),
    )
