from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union
from xml.etree.ElementTree import iterparse

import numpy as np


@dataclass(frozen=True)
class TrafficFrame:
    time: float
    ids: tuple[str, ...]
    positions: np.ndarray
    speeds: np.ndarray
    angles: np.ndarray

    def index(self):
        return {vehicle_id: i for i, vehicle_id in enumerate(self.ids)}


@dataclass(frozen=True)
class ClientFrame:
    time: float
    ids: tuple[str, ...]
    positions: np.ndarray
    speeds: np.ndarray
    angles: np.ndarray
    replaced: np.ndarray


class TrafficTrace:

    def __init__(self, frames: Sequence[TrafficFrame], bounds: np.ndarray, source: Path):
        if not frames:
            raise ValueError("The XML trace contains no <timestep> elements")
        self.frames = tuple(frames)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.source = source
        counts = [len(frame.ids) for frame in self.frames]
        self.min_vehicles = min(counts)
        self.max_vehicles = max(counts)
        self.unique_vehicles = len(
            {vehicle_id for frame in self.frames for vehicle_id in frame.ids}
        )

    @classmethod
    def from_xml(cls, path: Union[str, Path]):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frames: list[TrafficFrame] = []
        minimum = np.array([np.inf, np.inf], dtype=np.float64)
        maximum = np.array([-np.inf, -np.inf], dtype=np.float64)
        previous_time = -np.inf
        for _, element in iterparse(path, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "timestep":
                continue
            time = float(element.attrib["time"])
            if time <= previous_time:
                raise ValueError("SUMO timesteps must be strictly increasing")
            previous_time = time
            ids: list[str] = []
            positions: list[tuple[float, float]] = []
            speeds: list[float] = []
            angles: list[float] = []
            for vehicle in element:
                if vehicle.tag.rsplit("}", 1)[-1] != "vehicle":
                    continue
                try:
                    vehicle_id = vehicle.attrib["id"]
                    x = float(vehicle.attrib["x"])
                    y = float(vehicle.attrib["y"])
                    speed = float(vehicle.attrib["speed"])
                    angle = float(vehicle.attrib.get("angle", 0.0))
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Malformed vehicle at timestep {time}") from exc
                ids.append(vehicle_id)
                positions.append((x, y))
                speeds.append(speed)
                angles.append(angle)
                minimum = np.minimum(minimum, (x, y))
                maximum = np.maximum(maximum, (x, y))
            frames.append(
                TrafficFrame(
                    time=time,
                    ids=tuple(ids),
                    positions=np.asarray(positions, dtype=np.float32).reshape(-1, 2),
                    speeds=np.asarray(speeds, dtype=np.float32),
                    angles=np.asarray(angles, dtype=np.float32),
                )
            )
            element.clear()
        if not np.all(np.isfinite(np.concatenate((minimum, maximum)))):
            raise ValueError("The XML trace contains no valid vehicle coordinates")
        return cls(frames, np.stack((minimum, maximum)), path)

    def validate_scenario(self, clients: int, episode_steps: int):
        if clients <= 0:
            raise ValueError("clients must be positive")
        if clients > self.min_vehicles:
            raise ValueError(
                f"Scenario requests {clients} clients, but the least-populated timestep has "
                f"only {self.min_vehicles}"
            )
        if episode_steps >= len(self.frames):
            raise ValueError(
                f"episode_steps={episode_steps} must be less than trace length={len(self.frames)}"
            )

    def client_frame(
        self, frame_index: int, clients: int, previous_ids: Sequence[str] = ()
    ):
        frame = self.frames[frame_index]
        lookup = frame.index()
        retained = [vehicle_id for vehicle_id in previous_ids if vehicle_id in lookup]
        retained_set = set(retained)
        available = sorted(vehicle_id for vehicle_id in frame.ids if vehicle_id not in retained_set)
        chosen = (retained + available[: max(0, clients - len(retained))])[:clients]
        if len(chosen) != clients:
            raise RuntimeError(f"Timestep {frame.time} cannot supply {clients} client vehicles")
        old_ids = set(previous_ids)
        indices = np.fromiter((lookup[vehicle_id] for vehicle_id in chosen), dtype=np.int64)
        return ClientFrame(
            time=frame.time,
            ids=tuple(chosen),
            positions=frame.positions[indices].astype(np.float64, copy=True),
            speeds=frame.speeds[indices].astype(np.float64, copy=True),
            angles=frame.angles[indices].astype(np.float64, copy=True),
            replaced=np.asarray([vehicle_id not in old_ids for vehicle_id in chosen], dtype=bool),
        )

    def iter_client_frames(
        self, start: int, steps: int, clients: int
    ):
        ids: tuple[str, ...] = ()
        for index in range(start, min(start + steps, len(self.frames))):
            selected = self.client_frame(index, clients, ids)
            ids = selected.ids
            yield selected
