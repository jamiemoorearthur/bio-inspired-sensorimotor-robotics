"""Educational grid search for Hopf CPG parameters on Ant and Swimmer.

Run from the repository root with:
    python "CPG tuning.py" --steps 500 --trials 3

This standalone example uses Hopf oscillators; ``src/controllers/cpg.py`` uses
sinusoids, so the script makes the difference explicit while reusing the same
action interface, Gymnasium environments, and episode metrics.
"""

import argparse
from itertools import product
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import mujoco.viewer 

def make_rough_ant_xml(directory):
    """Create a temporary Ant model with raised blocks along its forward path."""
    import gymnasium.envs.mujoco.ant_v5 as ant_module

    source = Path(ant_module.__file__).parent / "assets" / "ant.xml"
    tree = ET.parse(source)
    worldbody = tree.getroot().find("worldbody")
    # Each tuple is (forward position, block height, half-length).
    for index, (x_pos, height, half_length) in enumerate((
        (2.5, 0.10, 0.55), (3.4, 0.18, 0.35),
        (5.0, 0.22, 0.40), (6.2, 0.08, 0.65),
        (8.0, 0.16, 0.50), (9.1, 0.24, 0.30),
    )):
        ET.SubElement(worldbody, "geom", {
            "name": f"terrain_block_{index}", "type": "box",
            "pos": f"{x_pos} 0 {height / 2:.3f}",
            "size": f"{half_length} 1.5 {height / 2:.3f}",
            "rgba": "0.55 0.38 0.22 1", "friction": "1.2 0.5 0.5",
            "condim": "3",
        })
    path = Path(directory) / "ant_rough_track.xml"
    tree.write(path, encoding="unicode")
    return path


def run_adaptive_ant(seed, steps, amplitude, frequency, mu, alpha):
    """Run a GUI Ant episode with terrain and posture feedback adjustments."""
    with tempfile.TemporaryDirectory(prefix="hopf_ant_track_") as temp_dir:
        xml_file = make_rough_ant_xml(temp_dir)
        env = gym.make("Ant-v5", xml_file=str(xml_file), render_mode="human")
        try:
            env.reset(seed=seed)
            n = int(np.prod(env.action_space.shape))
            dt = float(env.unwrapped.dt)
            phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            state = np.column_stack((np.sqrt(mu) * np.cos(phases),
                                     np.sqrt(mu) * np.sin(phases)))
            total_reward = 0.0
            velocity_samples = []
            active_section = -1

            for step in range(steps):
                data = env.unwrapped.data
                x_pos, z_pos = float(data.qpos[0]), float(data.qpos[2])
                roughness = 0.0
                for start, end, level in ((2.0, 3.9, 0.5), (4.5, 6.6, 1.0),
                                          (7.5, 9.6, 0.75)):
                    if start <= x_pos < end:
                        roughness = level
                        break
                qw, qx, qy, qz = map(float, data.qpos[3:7])
                pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
                speed = float(data.qvel[0])
                amp = np.clip(amplitude + 0.10 * roughness
                              + 0.08 * max(0.0, 0.65 - z_pos)
                              + 0.05 * min(1.0, abs(pitch)), 0.1, 1.0)
                freq = np.clip(frequency + 0.25 * roughness
                               + 0.10 * max(0.0, 0.2 - speed), 0.2, 2.0)

                if roughness and active_section != int(x_pos):
                    active_section = int(x_pos)
                    print(f"x={x_pos:5.2f} m: rough section; amplitude={amp:.2f}, "
                          f"frequency={freq:.2f} Hz, torso height={z_pos:.2f} m")
                elif not roughness:
                    active_section = -1

                x, y = state[:, 0], state[:, 1]
                radius_sq = x * x + y * y
                omega = 2.0 * np.pi * freq
                state += dt * np.column_stack((
                    alpha * (mu - radius_sq) * x - omega * y,
                    alpha * (mu - radius_sq) * y + omega * x,
                ))
                action = np.clip(amp * state[:, 0], env.action_space.low,
                                 env.action_space.high).reshape(env.action_space.shape)
                _obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                velocity_samples.append(float(env.unwrapped.data.qvel[0]))
                env.render()
                if terminated or truncated:
                    print(f"Episode ended at step {step + 1}: "
                          f"terminated={terminated}, truncated={truncated}")
                    break

            print(f"Rough-track result: x={env.unwrapped.data.qpos[0]:.2f} m, "
                  f"mean forward velocity={np.mean(velocity_samples):.3f} m/s, "
                  f"reward={total_reward:.2f}")
        finally:
            env.close()


def hopf_actions(state, amplitude):
    """Return oscillator x values scaled to actuator commands."""
    return np.asarray(amplitude * state[:, 0], dtype=np.float32)


def run_episode(env_name, seed, steps, amplitude, frequency, mu, alpha):
    env = gym.make(env_name)
    try:
        observation, _ = env.reset(seed=seed)
        n = int(np.prod(env.action_space.shape))
        dt = float(env.unwrapped.dt)

        # Start oscillators evenly around the limit cycle. The Hopf dynamics
        # converge toward radius sqrt(mu); omega=2*pi*frequency (rad/s).
        phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        radius = np.sqrt(mu)
        state = np.column_stack((radius * np.cos(phases), radius * np.sin(phases)))
        omega = 2.0 * np.pi * frequency
        total_reward = 0.0
        velocity_samples = []

        for _ in range(steps):
            x, y = state[:, 0], state[:, 1]
            radius_sq = x * x + y * y
            dx = alpha * (mu - radius_sq) * x - omega * y
            dy = alpha * (mu - radius_sq) * y + omega * x
            state += dt * np.column_stack((dx, dy))

            action = np.clip(hopf_actions(state, amplitude), env.action_space.low,
                             env.action_space.high).reshape(env.action_space.shape)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            velocity_samples.append(float(env.unwrapped.data.qvel[0]))
            if terminated or truncated:
                break

        return total_reward, float(np.mean(velocity_samples))
    finally:
        env.close()


def tune(env_name, seeds, steps, amplitudes, frequencies, mus, alphas):
    """Evaluate a small parameter grid and rank by mean forward velocity."""
    candidates = list(product(amplitudes, frequencies, mus, alphas))
    results = []
    for amplitude, frequency, mu, alpha in candidates:
        episodes = [
            run_episode(env_name, seed, steps, amplitude, frequency, mu, alpha)
            for seed in seeds
        ]
        mean_reward = float(np.mean([episode[0] for episode in episodes]))
        mean_velocity = float(np.mean([episode[1] for episode in episodes]))
        results.append((mean_velocity, mean_reward, amplitude, frequency, mu, alpha))

    results.sort(reverse=True)
    print(f"\n{env_name}: tested {len(candidates)} settings over {len(seeds)} seed(s)")
    print("rank  amplitude  frequency(Hz)  mu  alpha  mean_forward_velocity  mean_reward")
    for rank, row in enumerate(results[:5], start=1):
        velocity, reward, amplitude, frequency, mu, alpha = row
        print(f"{rank:>4}  {amplitude:>9.3f}  {frequency:>13.3f}  {mu:>3.1f}  "
              f"{alpha:>5.1f}  {velocity:>21.4f}  {reward:>11.2f}")
    best = results[0]
    print("Best setting (amplitude, frequency, mu, alpha): "
          f"({best[2]:.3f}, {best[3]:.3f}, {best[4]:.1f}, {best[5]:.1f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trials", type=int, default=2,
                        help="Number of fixed seeds used to compare each setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    parser.add_argument("--frequencies", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 5.0])
    parser.add_argument("--gui", action="store_true",
                        help="Show Ant adapting on a rough track in a live MuJoCo window")
    args = parser.parse_args()
    if args.steps < 1 or args.trials < 1:
        parser.error("--steps and --trials must be positive")

    if args.gui:
        run_adaptive_ant(args.seed, args.steps, args.amplitudes[0],
                         args.frequencies[0], args.mus[0], args.alphas[0])
        return

    seeds = range(args.seed, args.seed + args.trials)
    for env_name in ("Ant-v5", "Swimmer-v5"):
        tune(env_name, seeds, args.steps, args.amplitudes, args.frequencies,
             args.mus, args.alphas)


if __name__ == "__main__":
    main()
