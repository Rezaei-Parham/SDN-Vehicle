from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class HexGrid:

    bounds: np.ndarray
    radius: float
    centers: np.ndarray

    @classmethod
    def from_bounds(cls, bounds: np.ndarray, radius: float):
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (2, 2) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("bounds must have shape (2, 2) with max > min")
        if radius <= 0:
            raise ValueError("radius must be positive")
        minimum, maximum = bounds
        horizontal = np.sqrt(3.0) * radius
        vertical = 1.5 * radius
        centers: list[tuple[float, float]] = []
        row = 0
        y = minimum[1]
        while y <= maximum[1] + 1e-9:
            offset = 0.5 * horizontal if row % 2 else 0.0
            x = minimum[0] + offset
            while x <= maximum[0] + 1e-9:
                centers.append((x, y))
                x += horizontal
            row += 1
            y = minimum[1] + row * vertical
        if not centers:
            centers = [tuple(np.mean(bounds, axis=0))]
        return cls(bounds=bounds, radius=float(radius), centers=np.asarray(centers))

    def nearest_indices(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return cKDTree(self.centers).query(points, k=1)[1].astype(np.int64)

    def vertices(self, center: np.ndarray):
        angles = np.deg2rad(np.arange(30.0, 390.0, 60.0))
        return np.asarray(center) + self.radius * np.column_stack((np.cos(angles), np.sin(angles)))

    @property
    def diagonal(self):
        return float(np.linalg.norm(self.bounds[1] - self.bounds[0]))

