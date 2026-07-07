from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from scipy.optimize import linear_sum_assignment

from .hexgrid import HexGrid
from .network import SpatialRiskMap


def fixed_edge_positions(grid: HexGrid, count: int):
    if count <= 0 or count > len(grid.centers):
        raise ValueError("Invalid edge-server count")
    first = int(np.argmin(np.sum(grid.centers, axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(grid.centers - grid.centers[first], axis=1)
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(grid.centers - grid.centers[index], axis=1)
        )
    return grid.centers[selected].copy()


@dataclass
class ConvexProviderPlacer:

    grid: HexGrid
    provider_range: float
    provider_speed: float
    placement_interval: float
    risk_map: SpatialRiskMap
    movement_penalty: float = 0.15

    def _coverage(
        self,
        client_positions: np.ndarray,
        weather: float,
    ):
        centers = self.grid.centers
        clients = np.asarray(client_positions, dtype=np.float64)
        distance = np.linalg.norm(
            centers[:, None, :] - clients[None, :, :], axis=2
        )
        midpoints = 0.5 * (
            centers[:, None, :] + clients[None, :, :]
        )
        risk = self.risk_map.risk(midpoints.reshape(-1, 2)).reshape(distance.shape)
        loss_logit = (
            -4.2
            + 5.7 * distance / max(self.provider_range, 1e-6)
            + 1.15 * weather
            + 1.65 * risk
        )
        success = 1.0 - 1.0 / (
            1.0 + np.exp(-np.clip(loss_logit, -20.0, 20.0))
        )
        return np.where(distance <= self.provider_range, success, 0.0)

    def choose_targets(
        self,
        provider_count: int,
        client_positions: np.ndarray,
        demand: np.ndarray,
        weather: float,
        current_positions: Optional[np.ndarray],
    ):
        if provider_count <= 0 or provider_count > len(self.grid.centers):
            raise ValueError("Invalid provider count")
        if current_positions is not None and len(current_positions) != provider_count:
            raise ValueError("current_positions must contain one row per provider")
        centers = self.grid.centers
        client_positions = np.asarray(client_positions, dtype=np.float64)
        demand = np.asarray(demand, dtype=np.float64)
        coverage = self._coverage(client_positions, weather)
        if current_positions is None:
            movement = np.zeros((provider_count, len(centers)), dtype=np.float64)
            allowed = np.ones_like(movement, dtype=bool)
        else:
            maximum_move = self.provider_speed * self.placement_interval + self.grid.radius
            movement = np.linalg.norm(
                np.asarray(current_positions)[:, None, :] - centers[None, :, :],
                axis=2,
            )
            allowed = movement <= maximum_move
            for provider in range(provider_count):
                if not np.any(allowed[provider]):
                    allowed[provider, int(np.argmin(movement[provider]))] = True

        assignment = cp.Variable((provider_count, len(centers)))
        served = cp.Variable(len(client_positions))
        occupancy = cp.sum(assignment, axis=0)
        movement_cost = movement / max(self.provider_range, 1e-6)
        objective = cp.Maximize(
            demand @ served
            - self.movement_penalty * cp.sum(cp.multiply(movement_cost, assignment))
        )
        constraints = [
            assignment >= 0.0,
            assignment <= allowed.astype(float),
            cp.sum(assignment, axis=1) == 1.0,
            occupancy <= 1.0,
            served >= 0.0,
            served <= 1.0,
            served <= coverage.T @ occupancy,
        ]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver="SCIPY", scipy_options={"method": "highs"})
        except (cp.error.SolverError, TypeError):
            problem.solve(solver="CLARABEL")
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or assignment.value is None:
            raise RuntimeError(f"Provider-placement optimization failed: {problem.status}")
        occupancy_value = np.clip(np.sum(assignment.value, axis=0), 0.0, 1.0)
        standalone_benefit = coverage @ demand
        rounded_score = (
            assignment.value
            + occupancy_value[None, :] / provider_count
            + 1e-7 * standalone_benefit[None, :]
        )
        rounded_score = np.where(allowed, rounded_score, -1e12)
        rows, columns = linear_sum_assignment(-rounded_score)
        if len(rows) != provider_count or not np.all(allowed[rows, columns]):
            raise RuntimeError("Could not round provider-placement solution to valid distinct hexes")
        ordered = np.empty(provider_count, dtype=np.int64)
        ordered[rows] = columns
        return self.grid.centers[ordered].copy()


@dataclass
class RandomProviderPlacer:
    grid: HexGrid
    provider_speed: float
    placement_interval: float
    rng: np.random.Generator

    def choose_targets(
        self,
        provider_count: int,
        client_positions: np.ndarray,
        demand: np.ndarray,
        weather: float,
        current_positions: Optional[np.ndarray],
    ):
        del client_positions, demand, weather
        centers = self.grid.centers
        if current_positions is None:
            indices = self.rng.choice(len(centers), provider_count, replace=False)
            return centers[indices].copy()
        maximum_move = self.provider_speed * self.placement_interval + self.grid.radius
        selected: list[int] = []
        for position in current_positions:
            distance = np.linalg.norm(centers - position, axis=1)
            candidates = np.flatnonzero(distance <= maximum_move)
            candidates = np.setdiff1d(candidates, np.asarray(selected, dtype=int), assume_unique=False)
            if len(candidates) == 0:
                candidates = np.setdiff1d(np.argsort(distance), np.asarray(selected, dtype=int))
            chosen = int(self.rng.choice(candidates))
            selected.append(chosen)
        return centers[selected].copy()


def move_toward_targets(
    positions: np.ndarray, targets: np.ndarray, maximum_distance: float
):
    positions = np.asarray(positions, dtype=np.float64)
    delta = np.asarray(targets, dtype=np.float64) - positions
    distance = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, maximum_distance / np.maximum(distance, 1e-12))
    return positions + delta * scale[:, None]
