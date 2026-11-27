"""Run the saved Hopf-trained ESN readout in a native MuJoCo viewer.

Examples:
    python -m src.simulations.run_hopf_esn_viewer --env Ant-v5 --surface even
    python -m src.simulations.run_hopf_esn_viewer --env Ant-v5 --surface uneven
    python -m src.simulations.run_hopf_esn_viewer --env Swimmer-v5 --surface water

The network learned to predict the next Hopf oscillator position from the
current four oscillator positions. Those predictions drive the actuators,
with no robot observations fed into the reservoir. This is an open-loop
Hopf-to-ESN imitation demo, not a robot-trained adaptive policy.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import mujoco.viewer
import numpy as np

from src.models.esn_model_reservoir_computing import (
    HOPF_ALPHA,
    HOPF_DT,
    HOPF_FREQUENCY_HZ,
    HOPF_MU,
    rk4_hopf_step,
)
from src.simulations.benchmark_controller_surfaces import (
    SURFACES,
    write_surface_xml,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = REPO_ROOT / "results" / "models" / "esn_hopf_best_weights.npz"
PHASE_OFFSETS = np.array([0.0, np.pi, np.pi, 0.0])  # FL, FR, HL, HR.


class HopfDrivenESN:
    """Load trained reservoir/readout weights and advance Hopf inputs live."""

    def __init__(self, weights_path: Path):
        """Load the best sweep model and initialize its recurrent state."""
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Trained weights missing at {weights_path}. Run "
                "python -m src.models.esn_model_reservoir_computing first."
            )
        with np.load(weights_path) as saved:
            self.rho = float(saved["rho"])
            self.input_weights = saved["input_weights"].copy()
            self.reservoir_weights = saved["reservoir_weights"].copy()
            self.readout_weights = saved["readout_weights"].copy()
            self.readout_intercept = saved["readout_intercept"].copy()
        self.leak_rate = 0.3  # Must match LEAK_RATE used to train the model.
        expected = (64, 4)
        if self.input_weights.shape != expected:
            raise ValueError(
                f"Expected saved ESN input weights shaped {expected}; "
                f"found {self.input_weights.shape}. Regenerate Hopf weights."
            )
        self.reset()

    def reset(self) -> None:
        """Clear the reservoir and restart the four Hopf oscillators."""
        radius = np.sqrt(HOPF_MU)
        self.hopf_state = np.stack(
            (radius * np.cos(PHASE_OFFSETS), radius * np.sin(PHASE_OFFSETS))
        )
        self.reservoir_state = np.zeros(self.reservoir_weights.shape[0])
        self.latest_action = np.zeros(4)

    def advance(self, control_dt: float) -> np.ndarray:
        """Integrate Hopf states at training dt and produce the next-step ESN output."""
        elapsed = 0.0
        omega = 2.0 * np.pi * HOPF_FREQUENCY_HZ
        # Preserve the ESN's 100 Hz training cadence while matching robot control dt.
        while elapsed < control_dt - 1e-12:
            dt = min(HOPF_DT, control_dt - elapsed)
            self.hopf_state = rk4_hopf_step(
                self.hopf_state, dt, HOPF_ALPHA, HOPF_MU, omega
            )
            hopf_position = self.hopf_state[0] / np.sqrt(HOPF_MU)
            candidate = np.tanh(
                self.input_weights @ hopf_position
                + self.reservoir_weights @ self.reservoir_state
            )
            self.reservoir_state = (
                (1.0 - self.leak_rate) * self.reservoir_state
                + self.leak_rate * candidate
            )
            self.latest_action = np.clip(
                self.readout_weights @ self.reservoir_state
                + self.readout_intercept,
                -1.0,
                1.0,
            )
            elapsed += dt
        return self.latest_action.copy()


def robot_action(env_name: str, oscillator_output: np.ndarray, action_space) -> np.ndarray:
    """Map the four learned Hopf predictions to the selected robot's actuators."""
    if env_name == "Ant-v5":
        fl, fr, hl, hr = np.clip(oscillator_output, -1.0, 1.0)
        # Gymnasium Ant motor order is HR, FL, FR, HL, with opposing hip/ankle signs.
        action = np.asarray((hr, -hr, fl, -fl, fr, -fr, hl, -hl), dtype=np.float32)
    else:
        # Swimmer has two segment motors; use the first two learned oscillator outputs.
        action = np.asarray(oscillator_output[:2], dtype=np.float32)
    return np.clip(action, action_space.low, action_space.high).reshape(action_space.shape)


def main(argv: list[str] | None = None) -> None:
    """Launch a visible MuJoCo simulation and drive it with saved Hopf ESN weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("Ant-v5", "Swimmer-v5"), default="Ant-v5")
    parser.add_argument("--surface", choices=SURFACES, default="even")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the viewer closes.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_FILE)
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be zero or greater.")

    policy = HopfDrivenESN(args.weights)
    with tempfile.TemporaryDirectory(prefix="hopf_esn_viewer_") as temp:
        xml_path = write_surface_xml(args.env, args.surface, Path(temp))
        env = gym.make(args.env, **({"xml_file": str(xml_path)} if xml_path else {}))
        viewer = None
        try:
            observation, _ = env.reset(seed=args.seed)
            del observation  # The Hopf-trained demonstration is intentionally open-loop.
            control_dt = float(env.unwrapped.dt)
            viewer = mujoco.viewer.launch_passive(
                env.unwrapped.model,
                env.unwrapped.data,
            )
            print(f"Environment: {args.env}; profile: {args.surface}")
            print(f"Control interval: {control_dt:.3f} s; trained reservoir rho: {policy.rho:.2f}")
            print("Live view uses Hopf oscillator states -> trained ESN -> robot actuators.")
            print("Robot sensor observations are not inputs to this trained model.")

            total_reward = 0.0
            episode_steps = 0
            total_steps = 0
            next_tick = time.perf_counter()
            while viewer.is_running() and (args.steps == 0 or total_steps < args.steps):
                oscillator_action = policy.advance(control_dt)
                action = robot_action(args.env, oscillator_action, env.action_space)
                _obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                episode_steps += 1
                total_steps += 1
                viewer.sync()

                if total_steps % 100 == 0:
                    print(
                        f"step={total_steps:5d} episode_step={episode_steps:4d} "
                        f"reward={total_reward:8.2f} "
                        f"ESN action={np.array2string(oscillator_action, precision=2)}"
                    )
                if terminated or truncated:
                    print(
                        f"Episode ended after {episode_steps} steps; "
                        f"episode reward={total_reward:.2f}; resetting."
                    )
                    _obs, _info = env.reset()
                    policy.reset()
                    total_reward = 0.0
                    episode_steps = 0

                # Pace the simulation at real time; this avoids a busy CPU loop.
                next_tick += control_dt
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        finally:
            if viewer is not None:
                viewer.close()
            env.close()


if __name__ == "__main__":
    main()
