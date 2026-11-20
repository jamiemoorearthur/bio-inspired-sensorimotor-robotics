"""Run the four CPG/reservoir comparisons and export episode-level CSV data."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.simulations.run_controller import run_simulation


# Each experiment changes one factor at a time: controller or robot model.
EXPERIMENTS = (
    ("EXP-01", "Ant-v5", "cpg", "ant_cpg.csv"),
    ("EXP-02", "Ant-v5", "reservoir", "ant_reservoir.csv"),
    ("EXP-03", "Swimmer-v5", "cpg", "swimmer_cpg.csv"),
    ("EXP-04", "Swimmer-v5", "reservoir", "swimmer_reservoir.csv"),
)

CSV_COLUMNS = (
    "experiment_id",
    "environment",
    "controller",
    "controller_label",
    "seed",
    "reward",
    "displacement",
    "velocity",
    "episode_length",
    "control_effort",
)


def _append_result(path: Path, row: dict[str, Any]) -> None:
    """Append one episode to its condition's CSV, writing headers if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _log_line(log_path: Path, message: str) -> None:
    """Write a timestamped progress or error message to the run log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {message}\n")


def run_experiments(
    episodes: int = 1,
    steps: int = 500,
    seeds: list[int] | None = None,
    render: bool = False,
    results_dir: str | Path | None = None,
    record_video: bool = False,
    video_dir: str | Path | None = None,
    video_episodes: int = 1,
) -> list[dict[str, Any]]:
    """Run each condition for every requested seed and export each episode.

    CSV rows are episode-level measurements. The files are appended to so
    separate experiment runs remain available for later analysis. Any one
    failed episode is logged and collected while the remaining runs continue.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if seeds is None:
        seeds = [42]
    if not seeds:
        raise ValueError("Provide at least one random seed.")
    if video_episodes < 0:
        raise ValueError("video_episodes must be zero or greater.")
    if record_video and render:
        raise ValueError("Choose either human rendering or video recording, not both.")

    # Resolve outputs from the repository root, not the caller's current folder.
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(results_dir) if results_dir else repo_root / "results"
    csv_dir = output_root / "csv"
    log_path = output_root / "logs" / "experiments.log"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed = []
    failures = []
    _log_line(
        log_path,
        f"Started experiment batch: episodes={episodes}, steps={steps}, seeds={seeds}",
    )

    # Each seed is repeated for each condition, enabling paired comparisons.
    for experiment_id, env_name, controller_name, filename in EXPERIMENTS:
        csv_path = csv_dir / filename
        evaluation_index = 0
        for seed in seeds:
            for episode_index in range(episodes):
                # Give repeated episodes distinct deterministic seeds. Using
                # the number of requested seeds as the stride keeps adjacent
                # seed lists (for example 42, 43, 44) from overlapping repeats.
                episode_seed = int(seed) + episode_index * len(seeds)
                # Video is limited to the first N evaluation episodes per
                # condition, so long batches do not create videos for every run.
                record_this_video = (
                    record_video and evaluation_index < video_episodes
                )
                video_prefix = (
                    f"{env_name.split('-')[0].lower()}_"
                    f"{controller_name}_seed{episode_seed}"
                )
                evaluation_index += 1
                try:
                    metrics = run_simulation(
                        env_name=env_name,
                        controller_name=controller_name,
                        steps=steps,
                        seed=episode_seed,
                        render=render,
                        record_video=record_this_video,
                        video_dir=video_dir,
                        video_prefix=video_prefix,
                    )
                    row = {"experiment_id": experiment_id, **metrics}
                    _append_result(csv_path, row)
                    completed.append(row)
                    _log_line(
                        log_path,
                        f"Completed {experiment_id} seed={episode_seed} "
                        f"episode={episode_index + 1} reward={metrics['reward']:.6g}",
                    )
                    print(
                        f"{experiment_id} {env_name} {controller_name} "
                        f"seed={episode_seed} episode={episode_index + 1}/{episodes} "
                        f"reward={metrics['reward']:.4f}"
                    )
                except Exception as error:
                    detail = (
                        f"Failed {experiment_id} seed={episode_seed} "
                        f"episode={episode_index + 1}: {type(error).__name__}: {error}"
                    )
                    failures.append(detail)
                    _log_line(log_path, detail)
                    print(detail)

    _log_line(
        log_path,
        f"Finished experiment batch: completed={len(completed)}, failures={len(failures)}",
    )
    if failures:
        # Keep good results, but ensure automation can detect an incomplete batch.
        raise RuntimeError(
            f"{len(failures)} episode(s) failed; see {log_path}. "
            f"Successfully completed {len(completed)} episode(s)."
        )
    return completed


def main(argv=None):
    """Parse experiment options and execute the four comparison conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--render", action="store_true", help="Show each MuJoCo simulation."
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record the first selected evaluation episodes for each condition.",
    )
    parser.add_argument(
        "--video-dir", default=None, help="Video root (default: results/videos)."
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=1,
        help="Maximum number of episodes to record per experiment condition.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override the default repository results directory.",
    )
    args = parser.parse_args(argv)
    rows = run_experiments(
        episodes=args.episodes,
        steps=args.steps,
        seeds=args.seeds,
        render=args.render,
        results_dir=args.results_dir,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    print(f"Exported {len(rows)} episode rows.")
    return rows


if __name__ == "__main__":
    main()
