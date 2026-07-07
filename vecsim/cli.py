from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import ExperimentConfig, load_config
from .environment import VehicularEdgeEnv
from .evaluation import evaluate
from .mobility import TrafficTrace
from .ppo import PPOTrainer, load_actor_critic


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_history(path: Path, history: list[Dict[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _scenario(config: ExperimentConfig, name: str):
    try:
        return config.scenarios[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scenario {name!r}; available: {', '.join(config.scenarios)}"
        ) from exc


def command_analyze(args: argparse.Namespace, config: ExperimentConfig):
    trace = TrafficTrace.from_xml(args.xml)
    result = {
        "source": str(trace.source),
        "timesteps": len(trace.frames),
        "time_start_s": trace.frames[0].time,
        "time_end_s": trace.frames[-1].time,
        "bounds": {
            "x_min": float(trace.bounds[0, 0]),
            "y_min": float(trace.bounds[0, 1]),
            "x_max": float(trace.bounds[1, 0]),
            "y_max": float(trace.bounds[1, 1]),
        },
        "vehicles_per_timestep_min": trace.min_vehicles,
        "vehicles_per_timestep_max": trace.max_vehicles,
        "unique_vehicle_ids": trace.unique_vehicles,
    }
    for scenario in config.scenarios.values():
        trace.validate_scenario(scenario.clients, config.simulation.episode_steps)
    print(json.dumps(result, indent=2))


def command_train(args: argparse.Namespace, config: ExperimentConfig):
    from .plotting import plot_convergence

    trace = TrafficTrace.from_xml(args.xml)
    scenario = _scenario(config, args.scenario)
    environment = VehicularEdgeEnv(
        trace, scenario, config.simulation, config.seed, placement_mode="convex"
    )
    trainer = PPOTrainer(environment, config.ppo, config.seed, device=args.device)
    history = trainer.train(updates=args.updates)
    output = Path(args.output).expanduser().resolve()
    trainer.save(output / f"{scenario.name}_ppo.pt")
    _write_history(output / f"{scenario.name}_training.csv", history)
    _write_json(output / f"{scenario.name}_config.json", config.to_dict())
    plot_convergence(
        {scenario.name: history}, output / f"{scenario.name}_convergence.png"
    )
    print(f"Saved checkpoint and convergence outputs to {output}")


def command_evaluate(args: argparse.Namespace, config: ExperimentConfig):
    from .plotting import plot_metric_comparison, plot_provider_locations

    trace = TrafficTrace.from_xml(args.xml)
    scenario = _scenario(config, args.scenario)
    model = load_actor_critic(args.checkpoint, device=args.device)
    expected = VehicularEdgeEnv(
        trace, scenario, config.simulation, config.seed, placement_mode="convex"
    )
    dimensions = (
        model.observation_dim,
        model.critic_observation_dim,
        model.action_dim,
    )
    expected_dimensions = (
        expected.observation_dim,
        expected.critic_observation_dim,
        expected.action_dim,
    )
    if dimensions != expected_dimensions:
        raise ValueError(
            f"Checkpoint dimensions {dimensions} do not match scenario dimensions "
            f"{expected_dimensions}"
        )
    methods = {
        "Convex + CTDE-PPO": ("convex", model),
        "Random + CTDE-PPO": ("random", model),
        "Convex + greedy": ("convex", None),
        "Random + greedy": ("random", None),
    }
    evaluations = {
        name: evaluate(
            trace,
            scenario,
            config.simulation,
            args.episodes,
            config.seed + 5000,
            placement,
            selected_model,
        )
        for name, (placement, selected_model) in methods.items()
    }
    proposed = evaluations["Convex + CTDE-PPO"]
    baseline = evaluations["Random + greedy"]
    output = Path(args.output).expanduser().resolve()
    comparison = {
        scenario.name: {
            name: result.metrics for name, result in evaluations.items()
        }
    }
    _write_json(
        output / f"{scenario.name}_evaluation.json",
        {
            "means": comparison[scenario.name],
            "standard_deviations": {
                name: result.metric_std for name, result in evaluations.items()
            },
            "average_provider_locations": {
                name: result.average_provider_locations.tolist()
                for name, result in evaluations.items()
            },
        },
    )
    plot_metric_comparison(comparison, output / f"{scenario.name}_metrics.png")
    grid_environment = VehicularEdgeEnv(
        trace, scenario, config.simulation, config.seed, placement_mode="convex"
    )
    plot_provider_locations(
        grid_environment.grid,
        proposed.provider_history,
        baseline.provider_history,
        grid_environment.edge_positions,
        scenario.name,
        output / f"{scenario.name}_provider_locations.png",
    )
    print(json.dumps(comparison, indent=2))


def command_run_all(args: argparse.Namespace, config: ExperimentConfig):
    from .plotting import plot_convergence, plot_metric_comparison, plot_provider_locations

    trace = TrafficTrace.from_xml(args.xml)
    output = Path(args.output).expanduser().resolve()
    histories: Dict[str, list[Dict[str, float]]] = {}
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    deviations: Dict[str, Dict[str, Dict[str, float]]] = {}
    provider_locations: Dict[str, Dict[str, list[list[float]]]] = {}
    for scenario_index, scenario in enumerate(config.scenarios.values()):
        environment = VehicularEdgeEnv(
            trace,
            scenario,
            config.simulation,
            config.seed + scenario_index,
            placement_mode="convex",
        )
        trainer = PPOTrainer(
            environment,
            config.ppo,
            config.seed + scenario_index,
            device=args.device,
        )
        history = trainer.train(updates=args.updates)
        histories[scenario.name] = history
        trainer.save(output / "checkpoints" / f"{scenario.name}_ppo.pt")
        _write_history(output / "training" / f"{scenario.name}.csv", history)

        evaluation_seed = config.seed + 5000 + scenario_index * 1000
        methods = {
            "Convex + CTDE-PPO": ("convex", trainer.model),
            "Random + CTDE-PPO": ("random", trainer.model),
            "Convex + greedy": ("convex", None),
            "Random + greedy": ("random", None),
        }
        evaluations = {
            name: evaluate(
                trace,
                scenario,
                config.simulation,
                args.episodes,
                evaluation_seed,
                placement,
                selected_model,
            )
            for name, (placement, selected_model) in methods.items()
        }
        proposed = evaluations["Convex + CTDE-PPO"]
        baseline = evaluations["Random + greedy"]
        results[scenario.name] = {
            name: result.metrics for name, result in evaluations.items()
        }
        deviations[scenario.name] = {
            name: result.metric_std for name, result in evaluations.items()
        }
        provider_locations[scenario.name] = {
            name: result.average_provider_locations.tolist()
            for name, result in evaluations.items()
        }
        plot_provider_locations(
            environment.grid,
            proposed.provider_history,
            baseline.provider_history,
            environment.edge_positions,
            scenario.name,
            output / "figures" / f"{scenario.name}_provider_locations.png",
        )
    _write_json(
        output / "evaluation_summary.json",
        {
            "means": results,
            "standard_deviations": deviations,
            "average_provider_locations": provider_locations,
            "configuration": config.to_dict(),
            "trace": str(trace.source),
        },
    )
    plot_metric_comparison(results, output / "figures" / "network_metrics.png")
    plot_convergence(histories, output / "figures" / "ppo_convergence.png")
    print(json.dumps(results, indent=2))
    print(f"All artifacts saved to {output}")


def command_video(args: argparse.Namespace, config: ExperimentConfig):
    from .video import render_simulation_video

    trace = TrafficTrace.from_xml(args.xml)
    base_scenario = _scenario(config, args.scenario)
    clients = args.clients if args.clients is not None else trace.min_vehicles
    if clients > trace.min_vehicles:
        raise ValueError(
            f"--clients cannot exceed {trace.min_vehicles}, the minimum active XML count"
        )
    scenario = replace(
        base_scenario,
        name=f"{base_scenario.name}_video_{clients}",
        clients=clients,
    )
    model = (
        load_actor_critic(args.checkpoint, device=args.device)
        if args.checkpoint
        else None
    )
    result = render_simulation_video(
        trace=trace,
        scenario=scenario,
        simulation=config.simulation,
        output=args.output,
        frames=args.frames,
        fps=args.fps,
        start_index=args.start,
        seed=config.seed + 7000,
        model=model,
        placement_mode=args.placement,
        trail_steps=args.trail_steps,
        max_link_lines=args.max_links,
    )
    metadata_path = result.output.with_suffix(".json")
    _write_json(metadata_path, result.as_dict())
    print(json.dumps(result.as_dict(), indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vecsim",
        description="SUMO-driven SDN vehicular edge simulation",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
        help="YAML experiment configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Validate and summarize a SUMO FCD XML")
    analyze.add_argument("--xml", required=True, help="Path to simulation.out.xml")

    train = subparsers.add_parser("train", help="Train CTDE-PPO for one scenario")
    train.add_argument("--xml", required=True)
    train.add_argument("--scenario", default="scenario_20")
    train.add_argument("--output", default="outputs")
    train.add_argument("--updates", type=int)
    train.add_argument("--device", choices=("cpu", "cuda", "mps"))

    evaluation = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint against the random/greedy baseline"
    )
    evaluation.add_argument("--xml", required=True)
    evaluation.add_argument("--scenario", default="scenario_20")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", default="outputs")
    evaluation.add_argument("--episodes", type=int, default=5)
    evaluation.add_argument("--device", choices=("cpu", "cuda", "mps"))

    run_all = subparsers.add_parser(
        "run-all", help="Train/evaluate both required scenarios and generate all plots"
    )
    run_all.add_argument("--xml", required=True)
    run_all.add_argument("--output", default="outputs")
    run_all.add_argument("--updates", type=int)
    run_all.add_argument("--episodes", type=int, default=5)
    run_all.add_argument("--device", choices=("cpu", "cuda", "mps"))

    video = subparsers.add_parser(
        "video",
        help="Render all XML traffic, client offloading, and provider movement",
    )
    video.add_argument("--xml", required=True)
    video.add_argument("--scenario", default="scenario_50")
    video.add_argument(
        "--checkpoint",
        help="PPO checkpoint; when omitted, the greedy offloader is used",
    )
    video.add_argument("--output", default="outputs/simulation_video.mp4")
    video.add_argument(
        "--clients",
        type=int,
        help="Decision-making clients; default is the XML minimum (172 for the supplied trace)",
    )
    video.add_argument("--frames", type=int, default=160)
    video.add_argument("--fps", type=int, default=10)
    video.add_argument("--start", type=int, default=0)
    video.add_argument("--trail-steps", type=int, default=35)
    video.add_argument("--max-links", type=int, default=45)
    video.add_argument("--placement", choices=("convex", "random"), default="convex")
    video.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        if args.command == "analyze":
            command_analyze(args, config)
        elif args.command == "train":
            command_train(args, config)
        elif args.command == "evaluate":
            command_evaluate(args, config)
        elif args.command == "run-all":
            command_run_all(args, config)
        elif args.command == "video":
            command_video(args, config)
        else:
            parser.error(f"Unsupported command: {args.command}")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
