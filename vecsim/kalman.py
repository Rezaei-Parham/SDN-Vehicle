from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np


@dataclass
class ScalarKalmanFilter:

    mean: float = 0.25
    variance: float = 0.20
    process_variance: float = 0.015
    measurement_variance: float = 0.040

    def predict(self):
        self.variance += self.process_variance
        return float(np.clip(self.mean, 0.0, 1.0))

    def update(self, measurement: float):
        measurement = float(np.clip(measurement, 0.0, 1.0))
        gain = self.variance / (self.variance + self.measurement_variance)
        self.mean += gain * (measurement - self.mean)
        self.variance = (1.0 - gain) * self.variance
        self.mean = float(np.clip(self.mean, 0.0, 1.0))
        return self.mean


class LinkKalmanBank:
    def __init__(self, process_variance: float = 0.015, measurement_variance: float = 0.040):
        self.filters: dict[Hashable, ScalarKalmanFilter] = {}
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def estimate(self, key: Hashable, noisy_measurement: float):
        link_filter = self.filters.get(key)
        if link_filter is None:
            link_filter = ScalarKalmanFilter(
                mean=float(np.clip(noisy_measurement, 0.0, 1.0)),
                process_variance=self.process_variance,
                measurement_variance=self.measurement_variance,
            )
            self.filters[key] = link_filter
        else:
            link_filter.predict()
        return link_filter.update(noisy_measurement)

    def discard_vehicles(self, active_vehicle_ids: set[str]):
        stale = [
            key
            for key in self.filters
            if isinstance(key, tuple) and key and key[0] not in active_vehicle_ids
        ]
        for key in stale:
            del self.filters[key]

