from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Union

_cache_root = Path(tempfile.gettempdir()) / "vecsim-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from .hexgrid import HexGrid


def plot_metric_comparison(
    results: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_path: Union[str, Path],
):
    metric_labels = [
        ("packet_loss_ratio", "Packet-loss ratio"),
        ("average_delay_s", "Average delay (s)"),
        ("missed_deadlines", "Missed deadlines"),
        ("blind_spot_occurrences", "Blind-spot occurrences"),
    ]
    scenarios = list(results)
    methods = list(next(iter(results.values())))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    for axis, (metric, title) in zip(axes.flat, metric_labels):
        for method_index, method in enumerate(methods):
            values = [results[scenario][method][metric] for scenario in scenarios]
            axis.bar(
                x + (method_index - (len(methods) - 1) / 2) * width,
                values,
                width,
                label=method,
            )
        axis.set_title(title)
        axis.set_xticks(x, [name.replace("_", " ") for name in scenarios])
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _provider_occupancy(grid: HexGrid, history: np.ndarray):
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 4 or history.shape[-1] != 2:
        raise ValueError(
            "provider history must have shape (episodes, steps, providers, 2)"
        )
    positions = history.reshape(-1, 2)
    nearest = grid.nearest_indices(positions)
    counts = np.bincount(nearest, minlength=len(grid.centers)).astype(np.float64)
    return 100.0 * counts / max(1, len(positions))


def plot_provider_locations(
    grid: HexGrid,
    proposed_history: np.ndarray,
    baseline_history: np.ndarray,
    edge_positions: np.ndarray,
    scenario_name: str,
    output_path: Union[str, Path],
):
    histories = (proposed_history, baseline_history)
    occupancies = tuple(_provider_occupancy(grid, history) for history in histories)
    maximum_occupancy = max(float(np.max(values)) for values in occupancies)
    normalization = Normalize(vmin=0.0, vmax=max(maximum_occupancy, 1e-9))
    polygons = [grid.vertices(center) for center in grid.centers]
    padding = 0.02 * (grid.bounds[1] - grid.bounds[0])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    heatmaps = []
    for axis, history, occupancy, title in zip(
        axes,
        histories,
        occupancies,
        ("Convex placement", "Random placement"),
    ):
        heatmap = PolyCollection(
            polygons,
            array=occupancy,
            cmap="YlOrRd",
            norm=normalization,
            edgecolors="#bcc5cf",
            linewidths=0.45,
        )
        axis.add_collection(heatmap)
        heatmaps.append(heatmap)
        mean_locations = np.mean(history, axis=(0, 1))
        axis.scatter(
            mean_locations[:, 0],
            mean_locations[:, 1],
            marker="*",
            s=125,
            color="#1764ab",
            edgecolor="white",
            linewidth=0.8,
            label="Mean provider location",
            zorder=3,
        )
        axis.scatter(
            edge_positions[:, 0],
            edge_positions[:, 1],
            marker="s",
            s=45,
            color="black",
            label="Fixed edge server",
            zorder=3,
        )
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(grid.bounds[0, 0] - padding[0], grid.bounds[1, 0] + padding[0])
        axis.set_ylim(grid.bounds[0, 1] - padding[1], grid.bounds[1, 1] + padding[1])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    axes[0].legend(loc="upper right", framealpha=0.92)
    figure.colorbar(
        heatmaps[0],
        ax=axes,
        shrink=0.86,
        pad=0.02,
        label="Occupancy (% of provider-time samples)",
    )
    figure.suptitle(
        f"Provider occupancy heat map and mean locations — "
        f"{scenario_name.replace('_', ' ')}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_convergence(
    histories: Mapping[str, List[Dict[str, float]]], output_path: Union[str, Path]
):
    fields = [
        ("mean_rollout_reward", "Mean rollout reward"),
        ("policy_loss", "PPO policy loss"),
        ("value_loss", "PPO value loss"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for scenario, history in histories.items():
        updates = [row["update"] for row in history]
        for axis, (field, _) in zip(axes, fields):
            axis.plot(
                updates,
                [row[field] for row in history],
                marker="o",
                markersize=3,
                label=scenario,
            )
    for axis, (_, title) in zip(axes, fields):
        axis.set_title(title)
        axis.set_xlabel("Training update")
        axis.grid(alpha=0.25)
    axes[0].legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
