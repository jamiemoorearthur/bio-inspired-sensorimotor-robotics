"""Run one controller in a Gymnasium MuJoCo environment."""

import argparse
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
) -> dict[str, Any]:
    """Run one episode (up to ``steps``) and return its performance metrics.

    The loop gets an observation from ``reset``/``step``, asks the controller
    to turn it into an action, and sends that action back to Gymnasium. Either
    Gymnasium termination or truncation ends the episode early. The environment
    is always closed, including when controller or simulation code raises.
    """
    if env_name not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            f"Unsupported environment {env_name!r}; choose from {SUPPORTED_ENVIRONMENTS}."
        )
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    # Human rendering is opt-in, so batch experiments remain headless by default.
    env = gym.make(env_name, render_mode="human" if render else None)
    controller = None
    try:
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
        }
    finally:
        # Close MuJoCo viewers and native resources regardless of success.
        env.close()


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
    args = parser.parse_args(argv)
    metrics = run_simulation(
        env_name=args.env,
        controller_name=args.controller,
        steps=args.steps,
        seed=args.seed,
        render=args.render,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


if __name__ == "__main__":
    main()
