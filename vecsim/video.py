from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

_cache_root = Path(tempfile.gettempdir()) / "vecsim-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from matplotlib.collections import LineCollection

from .config import ScenarioConfig, SimulationConfig
from .environment import VehicularEdgeEnv
from .evaluation import GreedyOffloader
from .mobility import TrafficTrace
from .ppo import ActorCritic


@dataclass(frozen=True)
class VideoResult:
    output: Path
    frames: int
    fps: int
    simulated_clients: int
    raw_vehicles_min: int
    raw_vehicles_max: int
    final_metrics: Dict[str, float]

    def as_dict(self):
        return {
            "output": str(self.output),
            "frames": self.frames,
            "fps": self.fps,
            "duration_s": self.frames / self.fps,
            "simulated_clients": self.simulated_clients,
            "raw_vehicles_min": self.raw_vehicles_min,
            "raw_vehicles_max": self.raw_vehicles_max,
            "final_metrics": self.final_metrics,
        }


def _policy_actions(
    model: Optional[ActorCritic],
    greedy: GreedyOffloader,
    environment: VehicularEdgeEnv,
    observations: np.ndarray,
    critic_observations: np.ndarray,
    masks: np.ndarray,
):
    if model is None:
        return greedy.select(environment)
    device = next(model.parameters()).device
    with torch.no_grad():
        actions, _, _ = model.act(
            torch.as_tensor(observations, dtype=torch.float32, device=device),
            torch.as_tensor(critic_observations, dtype=torch.float32, device=device),
            torch.as_tensor(masks, dtype=torch.bool, device=device),
            deterministic=True,
        )
    return actions.cpu().numpy()


def _empty_offsets():
    return np.empty((0, 2), dtype=np.float64)


def render_simulation_video(
    trace: TrafficTrace,
    scenario: ScenarioConfig,
    simulation: SimulationConfig,
    output: Union[str, Path],
    frames: int,
    fps: int = 10,
    start_index: int = 0,
    seed: int = 17,
    model: Optional[ActorCritic] = None,
    placement_mode: str = "convex",
    trail_steps: int = 35,
    max_link_lines: int = 45,
):
    output = Path(output).expanduser().resolve()
    if frames <= 0:
        raise ValueError("frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0 <= start_index < len(trace.frames) - 1:
        raise ValueError("start_index is outside the trace")
    available = len(trace.frames) - start_index - 1
    frames = min(frames, available)
    video_config = SimulationConfig(**{**simulation.__dict__, "episode_steps": frames})
    environment = VehicularEdgeEnv(
        trace,
        scenario,
        video_config,
        seed=seed,
        placement_mode=placement_mode,
    )
    observations, critic_observations, masks = environment.reset(start_index=start_index)
    if model is not None:
        expected = (
            environment.observation_dim,
            environment.critic_observation_dim,
            environment.action_dim,
        )
        actual = (
            model.observation_dim,
            model.critic_observation_dim,
            model.action_dim,
        )
        if actual != expected:
            raise ValueError(
                f"Checkpoint dimensions {actual} do not match video scenario {expected}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    elif output.suffix.lower() == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required to create MP4 video")
        writer = FFMpegWriter(
            fps=fps,
            codec="h264",
            bitrate=3200,
            metadata={
                "title": "SUMO vehicular edge simulation",
                "comment": "All raw XML vehicles plus PPO clients and convex providers",
            },
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    else:
        raise ValueError("Video output must end in .mp4 or .gif")

    figure = plt.figure(figsize=(13.2, 7.4), constrained_layout=True)
    grid_spec = figure.add_gridspec(1, 2, width_ratios=(4.5, 1.35))
    axis = figure.add_subplot(grid_spec[0, 0])
    panel = figure.add_subplot(grid_spec[0, 1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(trace.bounds[:, 0])
    axis.set_ylim(trace.bounds[:, 1])
    axis.set_xlabel("SUMO x coordinate (m)")
    axis.set_ylabel("SUMO y coordinate (m)")
    axis.set_title("SUMO traffic and SDN-controlled provider vehicles", fontsize=14)

    hex_segments = [environment.grid.vertices(center) for center in environment.grid.centers]
    axis.add_collection(
        LineCollection(hex_segments, colors="#d9dee7", linewidths=0.45, zorder=0)
    )
    risk_x = np.linspace(trace.bounds[0, 0], trace.bounds[1, 0], 100)
    risk_y = np.linspace(trace.bounds[0, 1], trace.bounds[1, 1], 75)
    risk_points = np.stack(np.meshgrid(risk_x, risk_y), axis=-1).reshape(-1, 2)
    risk = environment.risk_map.risk(risk_points).reshape(len(risk_y), len(risk_x))
    axis.imshow(
        risk,
        extent=(
            trace.bounds[0, 0],
            trace.bounds[1, 0],
            trace.bounds[0, 1],
            trace.bounds[1, 1],
        ),
        origin="lower",
        cmap="Reds",
        alpha=0.12,
        vmin=0.0,
        vmax=1.0,
        zorder=-1,
    )

    raw_scatter = axis.scatter(
        [], [], s=10, c="#a9b1bd", alpha=0.62, linewidths=0, label="Other XML traffic"
    )
    local_scatter = axis.scatter(
        [], [], s=19, c="#2276b9", alpha=0.90, linewidths=0, label="PPO: local / idle"
    )
    provider_action_scatter = axis.scatter(
        [], [], s=23, c="#f08a24", alpha=0.95, linewidths=0, label="PPO: provider"
    )
    edge_action_scatter = axis.scatter(
        [], [], s=23, c="#8459b3", alpha=0.95, linewidths=0, label="PPO: edge"
    )
    provider_scatter = axis.scatter(
        [],
        [],
        s=145,
        c="#d62f2f",
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        zorder=8,
        label="Provider vehicle",
    )
    target_scatter = axis.scatter(
        [],
        [],
        s=85,
        facecolors="none",
        edgecolors="#9a1f1f",
        marker="X",
        linewidths=1.4,
        zorder=7,
        label="Provider target",
    )
    axis.scatter(
        environment.edge_positions[:, 0],
        environment.edge_positions[:, 1],
        s=70,
        c="#1f2933",
        marker="s",
        edgecolors="white",
        linewidths=0.7,
        zorder=7,
        label="Fixed edge server",
    )
    action_links = LineCollection([], linewidths=0.55, alpha=0.22, zorder=2)
    axis.add_collection(action_links)
    trail_lines = [
        axis.plot([], [], color="#b51f1f", linewidth=1.4, alpha=0.72, zorder=5)[0]
        for _ in range(scenario.providers)
    ]
    axis.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)

    panel.axis("off")
    panel.set_title("Live simulation state", loc="left", fontsize=13, pad=12)
    status_text = panel.text(
        0.0,
        0.98,
        "",
        transform=panel.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
        linespacing=1.42,
    )

    greedy = GreedyOffloader()
    raw_counts = []
    final_info: Dict[str, float] = {}
    with writer.saving(figure, str(output), dpi=125):
        for frame_number in range(frames):
            raw_frame = trace.frames[start_index + environment.elapsed_steps]
            raw_counts.append(len(raw_frame.ids))
            actions = _policy_actions(
                model,
                greedy,
                environment,
                observations,
                critic_observations,
                masks,
            )
            specs = environment.action_specs
            local_indices = np.asarray(
                [i for i, action in enumerate(actions) if specs[action].destination == "local"],
                dtype=np.int64,
            )
            provider_indices = np.asarray(
                [i for i, action in enumerate(actions) if specs[action].destination == "provider"],
                dtype=np.int64,
            )
            edge_indices = np.asarray(
                [i for i, action in enumerate(actions) if specs[action].destination == "edge"],
                dtype=np.int64,
            )
            client_id_set = set(environment.current_frame.ids)
            raw_lookup = np.asarray(
                [vehicle_id not in client_id_set for vehicle_id in raw_frame.ids], dtype=bool
            )
            raw_scatter.set_offsets(raw_frame.positions[raw_lookup])
            local_scatter.set_offsets(
                environment.current_frame.positions[local_indices]
                if len(local_indices)
                else _empty_offsets()
            )
            provider_action_scatter.set_offsets(
                environment.current_frame.positions[provider_indices]
                if len(provider_indices)
                else _empty_offsets()
            )
            edge_action_scatter.set_offsets(
                environment.current_frame.positions[edge_indices]
                if len(edge_indices)
                else _empty_offsets()
            )
            provider_scatter.set_offsets(environment.provider_positions)
            target_scatter.set_offsets(environment.provider_targets)

            history = np.asarray(environment.provider_history)
            history = history[max(0, len(history) - trail_steps) :]
            for provider_index, line in enumerate(trail_lines):
                line.set_data(history[:, provider_index, 0], history[:, provider_index, 1])

            link_segments = []
            link_colors = []
            for client_index, action in enumerate(actions):
                spec = specs[action]
                if spec.destination == "local":
                    continue
                destination = (
                    environment.provider_positions[spec.destination_index]
                    if spec.destination == "provider"
                    else environment.edge_positions[spec.destination_index]
                )
                link_segments.append(
                    [environment.current_frame.positions[client_index], destination]
                )
                link_colors.append("#f08a24" if spec.destination == "provider" else "#8459b3")
                if len(link_segments) >= max_link_lines:
                    break
            action_links.set_segments(link_segments)
            action_links.set_color(link_colors)

            weather_label = {0.0: "clear", 0.5: "rain", 1.0: "severe"}.get(
                environment.weather, f"{environment.weather:.1f}"
            )
            queued = sum(len(queue) for queue in environment.queues.values())
            status_text.set_text(
                f"Frame             {frame_number + 1:>4}/{frames}\n"
                f"SUMO time         {raw_frame.time:>7.1f} s\n"
                f"XML vehicles      {len(raw_frame.ids):>7}\n"
                f"PPO clients       {scenario.clients:>7}\n"
                f"Queued tasks      {queued:>7}\n"
                f"Weather           {weather_label:>7}\n"
                f"\nACTIONS\n"
                f"Local / idle      {len(local_indices):>7}\n"
                f"To provider       {len(provider_indices):>7}\n"
                f"To edge           {len(edge_indices):>7}\n"
                f"\nCUMULATIVE METRICS\n"
                f"Packet attempts   {environment.metrics.transmission_attempts:>7}\n"
                f"Packet losses     {environment.metrics.packet_losses:>7}\n"
                f"Tasks completed   {environment.metrics.completed_tasks:>7}\n"
                f"Missed deadlines  {environment.metrics.missed_deadlines:>7}\n"
                f"Blind spots       {environment.metrics.blind_spots:>7}"
            )
            writer.grab_frame()
            (
                observations,
                critic_observations,
                masks,
                _,
                done,
                final_info,
            ) = environment.step(actions)
            if done and frame_number + 1 < frames:
                break
    plt.close(figure)
    return VideoResult(
        output=output,
        frames=len(raw_counts),
        fps=fps,
        simulated_clients=scenario.clients,
        raw_vehicles_min=min(raw_counts),
        raw_vehicles_max=max(raw_counts),
        final_metrics=final_info,
    )
