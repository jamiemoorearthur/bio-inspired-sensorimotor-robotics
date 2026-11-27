"""Benchmark CPG and random-reservoir controllers in MuJoCo surface profiles.

Run from the repository root, for example:
    python -m src.simulations.benchmark_controller_surfaces --steps 500 --seeds 42 43 44

Each trial times controller inference and Gymnasium/MuJoCo env.step separately.
The reservoir controller is the repository's existing seeded random-readout
baseline, not a trained policy. Surface profiles are approximate model changes:
Ant supports contact terrain, while Swimmer is a planar fluid model, so its
"sand" condition is only a high-drag proxy and its uneven condition uses course
obstacles rather than ground elevation.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from src.controllers.cpg import CPGController
from src.controllers.reservoir import ReservoirController


ENVIRONMENTS = ("Ant-v5", "Swimmer-v5")
CONTROLLERS = ("cpg", "reservoir")
SURFACES = ("even", "uneven", "sand", "water")
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = REPO_ROOT / "results" / "csv" / "surface_runtime_benchmark.csv"

SURFACE_DESCRIPTIONS = {
    "even": "stock MuJoCo model and default contact/fluid parameters",
    "uneven": "Ant raised blocks; Swimmer course obstacles (not uneven ground)",
    "sand": "low-friction Ant floor; high-viscosity/high-joint-damping Swimmer proxy",
    "water": "Ant high-damping water-like proxy without buoyancy; Swimmer higher-viscosity fluid",
}


def source_xml(env_name: str) -> Path:
    """Locate the installed Gymnasium asset XML for the selected robot."""
    if env_name == "Ant-v5":
        import gymnasium.envs.mujoco.ant_v5 as module
    elif env_name == "Swimmer-v5":
        import gymnasium.envs.mujoco.swimmer_v5 as module
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    return Path(module.__file__).parent / "assets" / (
        "ant.xml" if env_name == "Ant-v5" else "swimmer.xml"
    )


def _set_joint_damping(root: ET.Element, damping: float) -> None:
    """Set default joint damping where the source model defines it."""
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    joint = default.find("joint")
    if joint is None:
        joint = ET.SubElement(default, "joint")
    joint.set("damping", str(damping))


def write_surface_xml(env_name: str, surface: str, directory: Path) -> Path | None:
    """Write a temporary model variant; return None for stock even terrain."""
    if surface == "even":
        return None
    tree = ET.parse(source_xml(env_name))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No worldbody found in {env_name} asset.")

    if surface == "uneven" and env_name == "Ant-v5":
        # Raised boxes form contact terrain in the Ant's forward travel path.
        # Existing Ant baseline moves toward negative x, so place terrain there.
        blocks = ((-0.65, 0.10), (-1.8, 0.18), (-3.0, 0.22), (-4.4, 0.08), (-5.8, 0.16))
        for index, (x_position, height) in enumerate(blocks):
            ET.SubElement(worldbody, "geom", {
                "name": f"benchmark_uneven_block_{index}", "type": "box",
                "pos": f"{x_position} 0 {height / 2:.4f}",
                "size": f"0.35 1.5 {height / 2:.4f}",
                "rgba": "0.45 0.34 0.24 1", "friction": "1.0 0.5 0.5",
                "condim": "3", "contype": "0", "conaffinity": "1",
            })
    elif surface == "uneven" and env_name == "Swimmer-v5":
        # Swimmer moves in the x-y plane and does not ride over heightfield ground.
        # These cross-course boxes test obstacle contacts instead of ground roughness.
        # Swimmer also drifts toward negative x under the current baselines.
        # Enable robot-to-obstacle contact while keeping robot self-collision off.
        for geom in worldbody.findall(".//geom"):
            if geom.get("name") != "floor":
                geom.set("contype", "1")
                geom.set("conaffinity", "0")
        for index, x_position in enumerate((-1.4, -1.9, -2.5)):
            ET.SubElement(worldbody, "geom", {
                "name": f"benchmark_course_obstacle_{index}", "type": "box",
                "pos": f"{x_position} 0 0", "size": "0.16 1.2 0.35",
                "rgba": "0.45 0.34 0.24 1", "contype": "0", "conaffinity": "1",
                "condim": "3",
            })
    elif surface == "sand" and env_name == "Ant-v5":
        # A low-friction floor is a repeatable proxy; MuJoCo does not model grains.
        floor = worldbody.find("geom[@name='floor']")
        if floor is None:
            raise RuntimeError("Ant floor geom not found.")
        floor.set("friction", "0.25 0.05 0.01")
    elif surface == "sand" and env_name == "Swimmer-v5":
        # Swimmer has no ground contact, so emulate a resistive medium instead.
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("viscosity", "5.0")
        _set_joint_damping(root, 2.0)
    elif surface == "water" and env_name == "Ant-v5":
        # Damping proxy only: Ant asset has no buoyancy or submerged-body model.
        _set_joint_damping(root, 8.0)
        floor = worldbody.find("geom[@name='floor']")
        if floor is not None:
            floor.set("friction", "0.6 0.2 0.1")
    elif surface == "water" and env_name == "Swimmer-v5":
        # The source is already a fluid simulation (viscosity 0.1); increase drag.
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("viscosity", "1.0")
    else:
        raise ValueError(f"Unsupported profile {surface} for {env_name}.")

    output = directory / f"{env_name.split('-')[0].lower()}_{surface}.xml"
    tree.write(output, encoding="unicode")
    return output


def make_controller(name: str, env, observation: np.ndarray, seed: int):
    """Create the selected controller with CPG time matched to MuJoCo dt."""
    if name == "cpg":
        controller = CPGController(action_space=env.action_space)
        controller.dt = float(env.unwrapped.dt)
        return controller
    if name == "reservoir":
        return ReservoirController(
            action_space=env.action_space,
            input_dim=int(np.asarray(observation).size),
            reservoir_size=100,
            spectral_radius=0.9,
            leak_rate=0.3,
            seed=seed,
        )
    raise ValueError(f"Unsupported controller: {name}")


def run_trial(
    env_name: str, surface: str, controller_name: str, seed: int, steps: int
) -> dict[str, Any]:
    """Run one headless episode and collect performance plus wall-clock timings."""
    with tempfile.TemporaryDirectory(prefix="robot_surface_benchmark_") as temp:
        xml_path = write_surface_xml(env_name, surface, Path(temp))
        kwargs = {"xml_file": str(xml_path)} if xml_path else {}
        env = gym.make(env_name, **kwargs)
        try:
            observation, _ = env.reset(seed=seed)
            controller = make_controller(controller_name, env, observation, seed)
            controller.reset()
            start_x = float(env.unwrapped.data.qpos[0])
            reward_total = 0.0
            controller_times: list[float] = []
            physics_times: list[float] = []
            terrain_contacts = 0
            lengths = 0
            terminated = truncated = False

            for _ in range(steps):
                tick = time.perf_counter_ns()
                action = controller.act(observation)
                action_ready = time.perf_counter_ns()
                observation, reward, terminated, truncated, _ = env.step(action)
                finished = time.perf_counter_ns()
                controller_times.append((action_ready - tick) / 1e6)
                physics_times.append((finished - action_ready) / 1e6)
                reward_total += float(reward)
                lengths += 1
                if surface == "uneven":
                    model, data = env.unwrapped.model, env.unwrapped.data
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        geom1 = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                        ) or ""
                        geom2 = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                        ) or ""
                        if geom1.startswith("benchmark_") or geom2.startswith("benchmark_"):
                            terrain_contacts += 1
                if terminated or truncated:
                    break

            elapsed_ms = sum(controller_times) + sum(physics_times)
            return {
                "environment": env_name,
                "surface": surface,
                "surface_description": SURFACE_DESCRIPTIONS[surface],
                "controller": controller_name,
                "controller_label": (
                    "sinusoidal CPG" if controller_name == "cpg"
                    else "untrained random-readout ESN"
                ),
                "seed": seed,
                "steps": lengths,
                "terrain_contact_count": terrain_contacts,
                "reward": reward_total,
                "displacement": float(env.unwrapped.data.qpos[0]) - start_x,
                "controller_mean_ms": mean(controller_times),
                "mujoco_step_mean_ms": mean(physics_times),
                "combined_step_mean_ms": elapsed_ms / lengths,
                "combined_steps_per_second": lengths / (elapsed_ms / 1000.0),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        finally:
            env.close()


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write one fresh CSV containing every trial and its model description."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print mean reward and timed step costs for each robot/profile/controller."""
    print("\nMean results across the selected seeds:")
    keys = sorted({(r["environment"], r["surface"], r["controller"]) for r in rows})
    for environment, surface, controller in keys:
        group = [
            r for r in rows
            if (r["environment"], r["surface"], r["controller"])
            == (environment, surface, controller)
        ]
        print(
            f"{environment:10s} {surface:7s} {controller:9s} "
            f"reward={mean(r['reward'] for r in group):9.2f} "
            f"terrain_contacts={mean(r['terrain_contact_count'] for r in group):6.1f} "
            f"controller={mean(r['controller_mean_ms'] for r in group):7.3f} ms "
            f"MuJoCo={mean(r['mujoco_step_mean_ms'] for r in group):7.3f} ms "
            f"combined={mean(r['combined_step_mean_ms'] for r in group):7.3f} ms/step "
            f"({mean(r['combined_steps_per_second'] for r in group):7.1f} steps/s)"
        )


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    """Run the full robot × surface × controller comparison and save CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args(argv)
    if args.steps < 1 or not args.seeds:
        parser.error("--steps and --seeds must contain positive values.")

    rows = []
    trials = len(ENVIRONMENTS) * len(SURFACES) * len(CONTROLLERS) * len(args.seeds)
    trial_number = 0
    for env_name in ENVIRONMENTS:
        for surface in SURFACES:
            for seed in args.seeds:
                for controller in CONTROLLERS:
                    trial_number += 1
                    print(
                        f"[{trial_number}/{trials}] {env_name} / {surface} / "
                        f"{controller} / seed {seed}", flush=True
                    )
                    rows.append(
                        run_trial(env_name, surface, controller, seed, args.steps)
                    )
    write_results(rows, args.output)
    print_summary(rows)
    print(f"\nDetailed trial results: {args.output}")
    print(
        "Interpretation: controller time is inference overhead; MuJoCo time is "
        "physics stepping; combined time is the practical control-loop cost. "
        "Reservoir results use an untrained random readout."
    )
    return rows


if __name__ == "__main__":
    main()
