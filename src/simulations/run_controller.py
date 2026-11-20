"""Run one controller in a Gymnasium MuJoCo environment."""

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from src.controllers.cpg import CPGController
from src.controllers.reservoir import ReservoirController


SUPPORTED_ENVIRONMENTS = ("Ant-v5", "Swimmer-v5")
SUPPORTED_CONTROLLERS = ("cpg", "reservoir")


def make_controller(name: str, action_space, observation, seed: int):
    """Build the selected controller using this environment's spaces.

    Gymnasium supplies observations to ``act`` and checks the returned action
    against its action space. The reservoir needs the flattened observation
    size to create its sensor-to-reservoir weights on construction.
    """
    normalized_name = name.lower()
    if normalized_name == "cpg":
        return CPGController(action_space=action_space)
    if normalized_name == "reservoir":
        return ReservoirController(
            action_space=action_space,
            input_dim=int(np.asarray(observation).size),
            seed=seed,
        )
    raise ValueError(
        f"Unsupported controller {name!r}; choose from {SUPPORTED_CONTROLLERS}."
    )


def _mujoco_forward_state(env) -> tuple[float | None, float | None]:
    """Read forward position and velocity from MuJoCo state when available.

    Both supported models use world x as their forward axis. Reading qpos and
    qvel directly avoids assuming that Ant and Swimmer encode these values at
    the same indices in their observation vectors.
    """
    unwrapped = env.unwrapped
    data = getattr(unwrapped, "data", None)
    if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
        return None, None
    position = float(data.qpos[0]) if len(data.qpos) else None
    velocity = float(data.qvel[0]) if len(data.qvel) else None
    return position, velocity


def run_simulation(
    env_name: str = "Ant-v5",
    controller_name: str = "cpg",
    steps: int = 500,
    seed: int = 42,
    render: bool = False,
    record_video: bool = False,
    video_dir: str | Path | None = None,
    video_prefix: str | None = None,
) -> dict[str, Any]:
    """Run one episode (up to ``steps``) and return its performance metrics.

    The loop gets an observation from ``reset``/``step``, asks the controller
    to turn it into an action, and sends that action back to Gymnasium. Either
    Gymnasium termination or truncation ends the episode early. The environment
    is always closed, including when controller or simulation code raises.
    Optional recording wraps that same environment once, so the recorded RGB
    frames correspond to the actions produced by the selected controller.
    """
    if env_name not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            f"Unsupported environment {env_name!r}; choose from {SUPPORTED_ENVIRONMENTS}."
        )
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if record_video and render:
        raise ValueError("Choose either human rendering or video recording, not both.")

    # RGB rendering is enabled only for recording; ordinary experiment runs
    # keep render_mode=None and avoid the extra frame-rendering overhead.
    render_mode = "rgb_array" if record_video else "human" if render else None
    env_kwargs = {"render_mode": render_mode}
    if record_video:
        # MuJoCo accepts these output dimensions where its renderer supports them.
        env_kwargs.update(width=1280, height=720)
    env = gym.make(env_name, **env_kwargs)
    video_path = None
    video_stem = None
    recording_dir = None
    controller = None
    try:
        if record_video:
            # RecordVideo encodes the RGB images returned by MuJoCo's render()
            # into an MP4. Check the encoder packages first for a useful error.
            missing = [
                package
                for package in ("moviepy", "imageio_ffmpeg")
                if importlib.util.find_spec(package) is None
            ]
            if missing:
                raise RuntimeError(
                    "Video recording needs Gymnasium's video encoder dependencies. "
                    "Install them with: pip install moviepy imageio-ffmpeg"
                )

            from gymnasium.wrappers import RecordVideo

            # A fresh environment is used for each run, so episode zero is the
            # selected evaluation episode and the wrapper records its full life.
            project_root = Path(__file__).resolve().parents[2]
            output_root = (
                Path(video_dir)
                if video_dir is not None
                else project_root / "results" / "videos"
            )
            robot_dir = output_root / env_name.split("-")[0].lower()
            robot_dir.mkdir(parents=True, exist_ok=True)
            base_stem = video_prefix or (
                f"{env_name.split('-')[0].lower()}_"
                f"{controller_name.lower()}_seed{seed}"
            )
            video_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", base_stem)

            # Avoid overwriting earlier videos when the same seed is rerun.
            suffix = 1
            while (
                (robot_dir / f"{video_stem}.mp4").exists()
                or list(robot_dir.glob(f"{video_stem}-episode-*.mp4"))
            ):
                video_stem = f"{base_stem}_run{suffix:02d}"
                suffix += 1
            video_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video_stem)
            video_path = robot_dir / f"{video_stem}.mp4"
            # RecordVideo warns whenever its target folder already exists.
            # Give each run a private new folder, then move the final MP4 up.
            recording_dir = robot_dir / f".{video_stem}_recording"

            # One wrapper owns capture and encoding. The lambda selects episode
            # zero explicitly, avoiding Gymnasium's periodic default schedule.
            env = RecordVideo(
                env,
                video_folder=str(recording_dir),
                episode_trigger=lambda episode_id: episode_id == 0,
                video_length=0,
                name_prefix=video_stem,
                disable_logger=True,
            )

        observation, _info = env.reset(seed=seed)
        controller = make_controller(
            controller_name, env.action_space, observation, seed
        )
        controller.reset()

        # Capture the actual MuJoCo root position before the first action.
        initial_x, _initial_velocity = _mujoco_forward_state(env)
        total_reward = 0.0
        control_effort = 0.0
        velocity_samples = []
        episode_length = 0
        terminated = truncated = False

        # Each pass is one control interval: observation -> action -> physics.
        for _step in range(steps):
            action = controller.act(observation)
            observation, reward, terminated, truncated, _info = env.step(action)

            # Sum squared actuator commands as a simple cumulative effort cost.
            action_values = np.asarray(action, dtype=np.float64)
            control_effort += float(np.sum(np.square(action_values)))
            total_reward += float(reward)
            episode_length += 1

            # Sample actual forward speed after physics advances this interval.
            _position, velocity = _mujoco_forward_state(env)
            if velocity is not None:
                velocity_samples.append(velocity)

            if render:
                env.render()
            if terminated or truncated:
                # Clear recurrent/oscillator state at an episode boundary.
                controller.reset()
                break

        final_x, _final_velocity = _mujoco_forward_state(env)
        displacement = (
            final_x - initial_x
            if initial_x is not None and final_x is not None
            else None
        )
        mean_velocity = (
            float(np.mean(velocity_samples)) if velocity_samples else None
        )

        # Reaching the requested step cap is also an episode boundary for the
        # caller, even if Gymnasium itself did not signal truncation.
        if not (terminated or truncated):
            controller.reset()

        return {
            "environment": env_name,
            "controller": controller_name.lower(),
            "seed": int(seed),
            "reward": total_reward,
            "displacement": displacement,
            "velocity": mean_velocity,
            "episode_length": episode_length,
            "control_effort": control_effort,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "controller_label": (
                "untrained/random ESN baseline"
                if controller_name.lower() == "reservoir"
                else "CPG"
            ),
            "video_path": str(video_path) if video_path is not None else None,
        }
    finally:
        # Closing RecordVideo finalises even a rollout stopped at the step cap.
        # Gymnasium closes its wrapped MuJoCo environment as part of this call.
        env.close()
        if record_video and video_stem is not None:
            recorded_files = sorted(
                recording_dir.glob(f"{video_stem}-episode-*.mp4")
            )
            if not recorded_files:
                if recording_dir.exists() and not any(recording_dir.iterdir()):
                    recording_dir.rmdir()
                raise RuntimeError(
                    "Video recording produced no MP4. Confirm the environment "
                    "rendered frames and the video encoder is installed."
                )
            recorded_file = recorded_files[0]
            if recorded_file.stat().st_size == 0:
                raise RuntimeError(f"Recorded video is empty: {recorded_file}")
            # RecordVideo adds '-episode-0' to its filename; simplify the final
            # output to the requested env/controller/seed name after finalising.
            recorded_file.replace(video_path)
            recording_dir.rmdir()


def main(argv=None):
    """Parse command-line options and print one episode's metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=SUPPORTED_ENVIRONMENTS, default="Ant-v5")
    parser.add_argument(
        "--controller", choices=SUPPORTED_CONTROLLERS, default="cpg"
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render", action="store_true", help="Open a human-view MuJoCo window."
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Save this rollout as an MP4."
    )
    parser.add_argument(
        "--video-dir", default=None, help="Video root (default: results/videos)."
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=1,
        help="Number of eligible evaluation episodes; this command runs one episode.",
    )
    args = parser.parse_args(argv)
    if args.video_episodes < 0:
        parser.error("--video-episodes must be zero or greater.")
    metrics = run_simulation(
        env_name=args.env,
        controller_name=args.controller,
        steps=args.steps,
        seed=args.seed,
        render=args.render,
        record_video=args.record_video and args.video_episodes > 0,
        video_dir=args.video_dir,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


if __name__ == "__main__":
    main()
