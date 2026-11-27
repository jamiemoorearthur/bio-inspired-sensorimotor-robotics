"""Interactive four-leg Hopf CPG visualizer and tuning sandbox.

Run ``python cpg_realtime.py`` for the interactive window, or
``python cpg_realtime.py --check`` to run the headless numerical checks.
Install the app dependencies with ``pip install numpy scipy matplotlib
gymnasium[mujoco]``. Gymnasium Ant-v5 is optional until Connect Ant-v5 is clicked.

The displayed stance interval is an illustrative assumption: cos(theta) < 0.
It is not a MuJoCo contact measurement and this application does not simulate
a physical robot. The sliders and equations let you study oscillator locking.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from collections import deque
from typing import Any
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize


# Leg order is fixed throughout the app: front-left, front-right, hind-left,
# hind-right. These colors keep a leg identifiable in every panel.
LEGS = ("FL", "FR", "HL", "HR")
LEG_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
PRESET_PHASES = {
    "trot": (0.0, 0.5, 0.5, 0.0),
    "pace": (0.0, 0.5, 0.0, 0.5),
    "bound": (0.0, 0.0, 0.5, 0.5),
    "walk": (0.25, 0.75, 0.0, 0.5),
    "custom": (0.0, 0.5, 0.5, 0.0),
}
DT = 0.005  # Fixed integration step, in seconds, as specified for the demo.
HISTORY_SECONDS = 5.0  # Output and radius plots retain this much simulated time.
OPTIMISE_HORIZON = 3.0  # Fixed rollout duration used by the button's objective.
LOCK_COST_LIMIT = 0.25  # Heuristic total J_phase threshold for the explanation text.
WEAK_COUPLING_LIMIT = 0.2  # Heuristic k threshold for explaining phase drift.


def cpg_rhs(
    time_s: float,
    state: np.ndarray,
    omega: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    coupling: float,
    phi: np.ndarray,
) -> np.ndarray:
    """Evaluate the four-oscillator Hopf CPG derivative.

    ``state`` has shape (2, 4), with radii in row 0 and phases in row 1.
    ``omega`` is angular frequency in radians per second; ``mu`` is amplitude
    squared. The ``time_s`` parameter is included for conventional ODE shape.
    """
    del time_s  # The autonomous CPG equations do not depend explicitly on time.
    radii, theta = state

    # Equation 1: r_dot_i = alpha * (mu_i - r_i**2) * r_i.
    radius_dot = alpha * (mu - radii**2) * radii

    # Equation 2: theta_dot_i = omega_i + k * sum_j r_j*sin(theta_j-theta_i-phi_ij).
    phase_difference = theta[None, :] - theta[:, None] - phi
    phase_dot = omega + coupling * np.sum(
        radii[None, :] * np.sin(phase_difference), axis=1
    )
    return np.stack((radius_dot, phase_dot))


def rk4_step(
    state: np.ndarray,
    dt: float,
    omega: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    coupling: float,
    phi: np.ndarray,
) -> np.ndarray:
    """Advance the CPG state by one fixed-step fourth-order Runge-Kutta step."""
    # Classical RK4: four slope estimates combine to an O(dt**5) local step.
    k1 = cpg_rhs(0.0, state, omega, mu, alpha, coupling, phi)
    k2 = cpg_rhs(dt / 2.0, state + dt * k1 / 2.0, omega, mu, alpha, coupling, phi)
    k3 = cpg_rhs(dt / 2.0, state + dt * k2 / 2.0, omega, mu, alpha, coupling, phi)
    k4 = cpg_rhs(dt, state + dt * k3, omega, mu, alpha, coupling, phi)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gait_psi(name: str) -> np.ndarray:
    """Return a gait preset's four target phases as cycle fractions."""
    key = name.lower()
    if key not in PRESET_PHASES:
        raise ValueError(f"Unknown gait {name!r}; choose from {tuple(PRESET_PHASES)}")
    return np.asarray(PRESET_PHASES[key], dtype=float)


def target_phi(psi: np.ndarray) -> np.ndarray:
    """Convert target cycle fractions to the antisymmetric phase-lag matrix."""
    phases = np.asarray(psi, dtype=float)
    # phi_ij = 2*pi*(psi_j - psi_i), which is antisymmetric by construction.
    return 2.0 * np.pi * (phases[None, :] - phases[:, None])


def wrapped_lag(theta: np.ndarray) -> np.ndarray:
    """Return measured pairwise lags theta_j-theta_i wrapped to [-pi, pi]."""
    phases = np.asarray(theta, dtype=float)
    # Complex argument wrapping handles both sides of the -pi/pi boundary.
    return np.angle(np.exp(1j * (phases[None, :] - phases[:, None])))


def cost(
    radii: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray | float,
    phi: np.ndarray,
) -> tuple[float, float, float]:
    """Return (total J, phase cost, amplitude cost) for a CPG state."""
    radius_values = np.asarray(radii, dtype=float)
    target_radius = np.sqrt(np.broadcast_to(np.asarray(mu, dtype=float), radius_values.shape))
    phase_error = np.angle(np.exp(1j * (wrapped_lag(theta) - phi)))
    # J_phase sums squared wrapped lag errors for all ordered pairs i != j.
    off_diagonal = ~np.eye(radius_values.size, dtype=bool)
    phase_cost = float(np.sum(phase_error[off_diagonal] ** 2))
    # J_amp = sum_i (r_i - sqrt(mu_i))**2.
    amplitude_cost = float(np.sum((radius_values - target_radius) ** 2))
    return phase_cost + amplitude_cost, phase_cost, amplitude_cost


def _run_steps(
    state: np.ndarray,
    count: int,
    omega_hz: float,
    mu_value: float,
    alpha: float,
    coupling: float,
    phi: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """Integrate a copied state for a number of fixed RK4 steps."""
    result = np.asarray(state, dtype=float).copy()
    omega = np.full(4, 2.0 * np.pi * omega_hz)
    mu = np.full(4, mu_value)
    for _ in range(count):
        result = rk4_step(result, dt, omega, mu, alpha, coupling, phi)
    return result


def _initial_test_state(psi: np.ndarray, mu_value: float = 1.0) -> np.ndarray:
    """Build reproducible radii and phase offsets for headless checks."""
    # The same nonzero offsets seed the coupled and uncoupled trot checks.
    phase_offsets = np.array([0.9, -0.8, 0.7, -0.6])
    return np.stack((np.full(4, np.sqrt(mu_value)), 2.0 * np.pi * psi + phase_offsets))


def run_checks() -> bool:
    """Run the four requested numerical checks without importing pyplot."""
    checks: list[tuple[str, bool, str, np.ndarray | None]] = []

    # 1) One uncoupled oscillator must converge from r=0.1 to sqrt(mu).
    mu_one, alpha_one = 1.2, 5.0
    one_state = np.array([[0.1], [0.4]])
    one_phi = np.zeros((1, 1))
    for _ in range(round(10.0 / DT)):
        one_state = rk4_step(one_state, DT, np.array([2.0 * np.pi]),
                             np.array([mu_one]), alpha_one, 0.0, one_phi)
    one_error = abs(one_state[0, 0] - np.sqrt(mu_one)) / np.sqrt(mu_one)
    checks.append(("uncoupled radius converges within 1% after 10 s",
                   one_error <= 0.01, f"relative radius error={one_error:.6f}", None))

    # 2) Coupled trot should remove phase offsets within 20 seconds.
    psi_trot = gait_psi("trot")
    phi_trot = target_phi(psi_trot)
    initial = _initial_test_state(psi_trot)
    locked = _run_steps(initial, round(20.0 / DT), 1.0, 1.0, 5.0, 1.0, phi_trot)
    lag_locked = wrapped_lag(locked[1])
    error_locked = np.angle(np.exp(1j * (lag_locked - phi_trot)))
    off_diagonal = ~np.eye(4, dtype=bool)
    max_error = float(np.max(np.abs(error_locked[off_diagonal])))
    checks.append(("trot locks below 0.05*pi with k=1 after 20 s",
                   max_error < 0.05 * np.pi,
                   f"maximum pairwise phase error={max_error / np.pi:.6f} pi",
                   lag_locked))

    # 3) With k=0, common frequency preserves the initial phase error.
    drifting = _run_steps(initial, round(20.0 / DT), 1.0, 1.0, 5.0, 0.0, phi_trot)
    lag_drifting = wrapped_lag(drifting[1])
    error_drifting = np.angle(np.exp(1j * (lag_drifting - phi_trot)))
    max_uncoupled_error = float(np.max(np.abs(error_drifting[off_diagonal])))
    checks.append(("uncoupled trot retains a large phase error with k=0",
                   max_uncoupled_error > 0.5,
                   f"maximum pairwise phase error={max_uncoupled_error / np.pi:.6f} pi",
                   lag_drifting))

    # 4) Optimise k (omega is mathematically neutral for relative phase cost).
    # The objective is the stated J after a fixed short rollout from offset phases.
    before_state = _run_steps(initial, round(OPTIMISE_HORIZON / DT), 1.0,
                              1.0, 5.0, 0.0, phi_trot)
    before_cost = cost(before_state[0], before_state[1], 1.0, phi_trot)[0]

    def objective(parameters: np.ndarray) -> float:
        omega_hz = float(np.clip(parameters[0], 0.2, 2.0))
        coupling = float(np.clip(parameters[1], 0.0, 5.0))
        trial_state = _run_steps(initial, round(OPTIMISE_HORIZON / DT),
                                 omega_hz, 1.0, 5.0, coupling, phi_trot)
        return cost(trial_state[0], trial_state[1], 1.0, phi_trot)[0]

    result = minimize(objective, np.array([1.0, 0.0]), method="Nelder-Mead",
                      bounds=((0.2, 2.0), (0.0, 5.0)),
                      options={"maxiter": 60, "xatol": 0.01, "fatol": 1e-5})
    after_cost = float(result.fun)
    checks.append(("Nelder-Mead lowers J from the starting setting",
                   after_cost < before_cost - 1e-6,
                   f"J before={before_cost:.6f}, after={after_cost:.6f}",
                   wrapped_lag(before_state[1])))

    all_passed = True
    for label, passed, detail, lag_matrix in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {label} ({detail})")
        if not passed:
            all_passed = False
            if lag_matrix is not None:
                print("Measured lag matrix (radians):")
                print(np.array2string(lag_matrix, precision=4, suppress_small=True))
    return all_passed


def build_figure() -> tuple[Any, dict[str, Any]]:
    """Create the Matplotlib dashboard and all interactive controls."""
    import matplotlib

    # Prefer QtAgg and fall back to TkAgg, both of which provide a real window.
    active_backend = matplotlib.get_backend().lower()
    if "agg" == active_backend or "inline" in active_backend or "notebook" in active_backend:
        # These optional GUI bindings select the backend; they are not required
        # for --check, and a standard-library Tk install is a valid fallback.
        qt_bindings = ("PyQt6", "PySide6", "PyQt5", "PySide2")
        if any(importlib.util.find_spec(binding) is not None for binding in qt_bindings):
            matplotlib.use("QtAgg", force=True)
        else:
            try:
                import tkinter  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Install a Qt binding or Tk to use the interactive window"
                ) from exc
            matplotlib.use("TkAgg", force=True)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, RadioButtons, Slider

    plt.ion()
    fig = plt.figure(figsize=(16, 12), constrained_layout=False)
    fig.suptitle("Four-leg Hopf CPG — real-time learning dashboard", fontsize=16)
    gs = fig.add_gridspec(4, 3, left=0.05, right=0.78, top=0.92, bottom=0.06,
                          width_ratios=(1.0, 1.2, 1.2),
                          height_ratios=(1.1, 1.0, 0.65, 0.7))

    # Panel 1: polar leg clock; open markers show target phases at sqrt(mu).
    ax_clock = fig.add_subplot(gs[0, 0], projection="polar")
    ax_clock.set_title("1. Leg clock", pad=14)
    ax_clock.set_ylim(0.0, 1.9)
    ax_clock.set_yticks((0.5, 1.0, 1.5))
    ax_clock.text(0.5, -0.18, "stance = cos θ < 0 (assumed contact)",
                  transform=ax_clock.transAxes, ha="center", va="top", fontsize=8)

    # Panel 2: output x_i = r_i * (1 + cos(theta_i)) with stance shading.
    output_grid = gs[0, 1:3].subgridspec(4, 1, hspace=0.08)
    output_axes = [fig.add_subplot(output_grid[0, 0])]
    for index in range(1, 4):
        output_axes.append(fig.add_subplot(output_grid[index, 0],
                                           sharex=output_axes[0]))
    output_axes[0].set_title("2. Output traces: xᵢ = rᵢ(1 + cos θᵢ)", fontsize=10)
    output_lines = []
    for index, axis in enumerate(output_axes):
        axis.set_ylim(-0.1, 3.8)
        axis.set_ylabel(LEGS[index], rotation=0, labelpad=16,
                        color=LEG_COLORS[index], fontsize=8)
        axis.tick_params(axis="y", labelsize=7)
        if index < 3:
            axis.tick_params(axis="x", labelbottom=False)
        output_lines.append(axis.plot([], [], color=LEG_COLORS[index], linewidth=1.1)[0])
    output_axes[-1].set_xlabel("simulated time (s)")

    # Panel 3: each radius should approach the dashed sqrt(mu) target.
    ax_radius = fig.add_subplot(gs[1, 0])
    ax_radius.set_title("3. Radius convergence")
    ax_radius.set_xlabel("simulated time (s)")
    ax_radius.set_ylabel("rᵢ")
    radius_lines = [ax_radius.plot([], [], color=LEG_COLORS[i], label=LEGS[i])[0]
                    for i in range(4)]
    radius_target_line = ax_radius.axhline(1.0, color="black", linestyle="--",
                                          linewidth=1.0, label="√μ")

    # Panels 4 and 5: pairwise measured and target-relative lag matrices.
    ax_lag = fig.add_subplot(gs[1, 1])
    ax_lag.set_title("4. Measured lag / π")
    ax_error = fig.add_subplot(gs[1, 2])
    ax_error.set_title("5. |lag error| / π")
    matrix_ticks = np.arange(4)
    lag_image = ax_lag.imshow(np.zeros((4, 4)), vmin=-1.0, vmax=1.0,
                              cmap="coolwarm", interpolation="nearest")
    error_image = ax_error.imshow(np.zeros((4, 4)), vmin=0.0, vmax=2.0,
                                  cmap="magma", interpolation="nearest")
    annotations: dict[str, list[Any]] = {"lag": [], "error": []}
    for axis in (ax_lag, ax_error):
        axis.set_xticks(matrix_ticks, labels=LEGS)
        axis.set_yticks(matrix_ticks, labels=LEGS)
        axis.set_xlabel("j")
        axis.set_ylabel("i")
    fig.colorbar(lag_image, ax=ax_lag, fraction=0.046, pad=0.04)
    fig.colorbar(error_image, ax=ax_error, fraction=0.046, pad=0.04)

    # Panel 6: J terms and optimizer comparison.
    ax_cost = fig.add_subplot(gs[2, 0])
    ax_cost.axis("off")
    ax_cost.set_title("6. Cost readout", loc="left")
    cost_text = ax_cost.text(0.0, 0.85, "", va="top", family="monospace", fontsize=9)

    # Panel 7: concise plain-English interpretation of the live state.
    ax_explain = fig.add_subplot(gs[2, 1:3])
    ax_explain.axis("off")
    ax_explain.set_title("7. What the equations are doing", loc="left")
    explanation_text = ax_explain.text(0.02, 0.80, "", va="top", wrap=True,
                                       fontsize=11, transform=ax_explain.transAxes)

    # Panel 8 explains each runtime stage and shows Gymnasium feedback when
    # the optional Ant-v5 viewer is connected.
    ax_stages = fig.add_subplot(gs[3, :])
    ax_stages.axis("off")
    ax_stages.set_title("8. Critical code stages and live Ant feedback", loc="left")
    stage_text = ax_stages.text(0.01, 0.82, "", va="top", family="monospace",
                                fontsize=9, transform=ax_stages.transAxes)

    controls: dict[str, Any] = {}
    slider_definitions = (
        ("omega", "ω (Hz)", 0.2, 2.0, 1.0),
        ("mu", "μ (amplitude²)", 0.1, 3.0, 1.0),
        ("alpha", "α", 0.5, 20.0, 5.0),
        ("coupling", "k", 0.0, 5.0, 1.0),
        ("speed", "speed (×)", 0.1, 3.0, 1.0),
        ("psi_fr", "custom ψ FR", 0.0, 1.0, 0.5),
        ("psi_hl", "custom ψ HL", 0.0, 1.0, 0.5),
        ("psi_hr", "custom ψ HR", 0.0, 1.0, 0.0),
    )
    slider_y = (0.91, 0.83, 0.75, 0.67, 0.59, 0.49, 0.41, 0.33)
    for (key, label, low, high, initial_value), y in zip(slider_definitions, slider_y):
        slider_ax = fig.add_axes((0.82, y, 0.15, 0.035))
        controls[key] = Slider(slider_ax, label, low, high, valinit=initial_value,
                               valstep=0.01 if key.startswith("psi_") else None)

    radio_ax = fig.add_axes((0.82, 0.12, 0.15, 0.18))
    radio_ax.set_title("Gait preset", fontsize=9)
    controls["preset"] = RadioButtons(radio_ax, ("trot", "pace", "bound", "walk", "custom"),
                                       active=0)
    ant_ax = fig.add_axes((0.82, 0.105, 0.15, 0.04))
    perturb_ax = fig.add_axes((0.82, 0.060, 0.07, 0.035))
    reset_ax = fig.add_axes((0.90, 0.060, 0.07, 0.035))
    optimise_ax = fig.add_axes((0.82, 0.015, 0.15, 0.035))
    controls["ant_button"] = Button(ant_ax, "Connect Ant-v5")
    controls["perturb_button"] = Button(perturb_ax, "Perturb FR")
    controls["reset_button"] = Button(reset_ax, "Reset")
    controls["optimise_button"] = Button(optimise_ax, "Optimise (ω, k)")

    history_size = int(round(HISTORY_SECONDS / DT))
    runtime: dict[str, Any] = {
        "fig": fig,
        "axes": (ax_clock, output_axes, ax_radius, ax_lag, ax_error),
        "clock_points": [],
        "target_points": [],
        "history_t": deque(maxlen=history_size),
        "history_x": [deque(maxlen=history_size) for _ in range(4)],
        "history_r": [deque(maxlen=history_size) for _ in range(4)],
        "state": None,
        "sim_time": 0.0,
        "last_wall_time": time.perf_counter(),
        "accumulator": 0.0,
        "last_optimise": "Optimiser not run yet.",
        "rng": np.random.default_rng(),
        "target_phi": target_phi(gait_psi("trot")),
        "psi": gait_psi("trot"),
        "ani": None,
    }
    runtime.update({
        "output_lines": output_lines,
        "radius_lines": radius_lines,
        "radius_target_line": radius_target_line,
        "lag_image": lag_image,
        "error_image": error_image,
        "annotations": annotations,
        "cost_text": cost_text,
        "explanation_text": explanation_text,
        "stage_text": stage_text,
        "controls": controls,
        "robot_env": None,
        "robot_viewer": None,
        "robot_accumulator": 0.0,
        "robot_dt": 0.0,
        "robot_total_reward": 0.0,
        "robot_steps": 0,
        "robot_observation": None,
        "robot_feedback": "Ant-v5 is disconnected. Click Connect Ant-v5 to open MuJoCo.",
    })
    _wire_callbacks(runtime)
    return fig, runtime


def _current_psi(runtime: dict[str, Any]) -> np.ndarray:
    """Read the selected preset or current custom phase sliders."""
    controls = runtime["controls"]
    if controls["preset"].value_selected == "custom":
        return np.array((0.0, controls["psi_fr"].val,
                         controls["psi_hl"].val, controls["psi_hr"].val))
    return gait_psi(controls["preset"].value_selected)


def _update_targets(runtime: dict[str, Any], reset_gait: bool = False) -> None:
    """Refresh target phases and optionally reset the state around the gait."""
    psi = _current_psi(runtime)
    runtime["psi"] = psi
    runtime["target_phi"] = target_phi(psi)
    if reset_gait:
        mu_value = runtime["controls"]["mu"].val
        target_theta = 2.0 * np.pi * psi
        # Reset randomises every oscillator's radius and phase so convergence
        # can be observed again from a visibly different initial condition.
        initial_radii = runtime["rng"].uniform(0.1, 1.5 * np.sqrt(mu_value), size=4)
        random_phases = runtime["rng"].uniform(-np.pi, np.pi, size=4)
        runtime["state"] = np.stack((
            initial_radii, target_theta + random_phases,
        ))


def _disconnect_ant(runtime: dict[str, Any]) -> None:
    """Close the passive MuJoCo viewer and its Gymnasium environment safely."""
    viewer = runtime.get("robot_viewer")
    env = runtime.get("robot_env")
    runtime["robot_viewer"] = None
    runtime["robot_env"] = None
    if viewer is not None:
        viewer.close()
    if env is not None:
        env.close()
    runtime["robot_feedback"] = "Ant-v5 is disconnected. Click Connect Ant-v5 to open MuJoCo."
    runtime["controls"]["ant_button"].label.set_text("Connect Ant-v5")


def _toggle_ant(runtime: dict[str, Any]) -> None:
    """Connect or disconnect a Gymnasium Ant-v5 controlled by the four CPG legs."""
    if runtime["robot_env"] is not None:
        _disconnect_ant(runtime)
        return
    try:
        import gymnasium as gym
        import mujoco.viewer

        # Stage A: construct/reset the environment; Ant's action space has 8 motors.
        env = gym.make("Ant-v5")
        observation, _info = env.reset(seed=42)
        # Stage B: launch MuJoCo's passive viewer alongside the Matplotlib window.
        viewer = mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data)
        runtime["robot_env"] = env
        runtime["robot_viewer"] = viewer
        runtime["robot_observation"] = observation
        runtime["robot_dt"] = float(env.unwrapped.dt)
        runtime["robot_accumulator"] = 0.0
        runtime["robot_total_reward"] = 0.0
        runtime["robot_steps"] = 0
        runtime["robot_feedback"] = "Ant-v5 connected; CPG motor commands are being sent."
        runtime["controls"]["ant_button"].label.set_text("Stop Ant")
    except Exception as exc:
        if "env" in locals():
            env.close()
        runtime["robot_feedback"] = (
            f"Could not open Ant-v5: {exc}. Install gymnasium[mujoco] and use a GUI session."
        )
        print(runtime["robot_feedback"])


def _ant_action(state: np.ndarray) -> np.ndarray:
    """Map four leg oscillator outputs to Ant's eight motors in actuator order."""
    radii, theta = state
    # Recover signed drive from x_i = r_i(1+cos(theta_i)): x_i-r_i=r_i*cos(theta_i).
    leg_drive = radii * np.cos(theta)
    # Gymnasium Ant XML motor order is HR, FL, FR, HL; each hip/ankle pair
    # receives opposite signs to make a simple flex/extend teaching mapping.
    fl, fr, hl, hr = np.clip(leg_drive, -1.0, 1.0)
    return np.asarray((hr, -hr, fl, -fl, fr, -fr, hl, -hl), dtype=np.float32)


def _step_ant(runtime: dict[str, Any], state: np.ndarray) -> None:
    """Advance Ant when the CPG clock reaches one Gymnasium action interval."""
    env = runtime.get("robot_env")
    viewer = runtime.get("robot_viewer")
    if env is None or viewer is None:
        return
    runtime["robot_accumulator"] += DT
    # One Gymnasium action advances several MuJoCo integration steps. Hold the
    # latest CPG output until Ant's own control interval has elapsed.
    while runtime["robot_accumulator"] >= runtime["robot_dt"]:
        runtime["robot_accumulator"] -= runtime["robot_dt"]
        observation, reward, terminated, truncated, _info = env.step(_ant_action(state))
        runtime["robot_observation"] = observation
        runtime["robot_total_reward"] += float(reward)
        runtime["robot_steps"] += 1
        if terminated or truncated:
            observation, _info = env.reset()
            runtime["robot_observation"] = observation
            runtime["robot_feedback"] = (
                "Ant episode ended and reset; cumulative reward is retained for this view."
            )
    if viewer.is_running():
        viewer.sync()
    else:
        _disconnect_ant(runtime)


def _stage_breakdown(runtime: dict[str, Any], state: np.ndarray) -> str:
    """Describe the live compute/control stages in learner-facing language."""
    omega_hz = float(runtime["controls"]["omega"].val)
    target_radius = float(np.sqrt(runtime["controls"]["mu"].val))
    coupling = float(runtime["controls"]["coupling"].val)
    env = runtime.get("robot_env")
    if env is None:
        robot_line = runtime["robot_feedback"]
    else:
        data = env.unwrapped.data
        robot_line = (
            f"Gymnasium feedback: x={data.qpos[0]:+.2f} m, z={data.qpos[2]:.2f} m, "
            f"vx={data.qvel[0]:+.2f} m/s, steps={runtime['robot_steps']}, "
            f"reward sum={runtime['robot_total_reward']:.1f}. "
            "These observations are displayed; they do not yet alter CPG parameters."
        )
    return (
        f"1 Parameters: ω={omega_hz:.2f} Hz → 2πf={2*np.pi*omega_hz:.2f} rad/s; "
        f"μ={runtime['controls']['mu'].val:.2f} → target r=√μ={target_radius:.2f}; "
        f"α={runtime['controls']['alpha'].val:.2f}, k={coupling:.2f}.\n"
        "2 Hopf update: RK4 integrates ṙ=α(μ−r²)r and θ̇=ω+kΣrⱼsin(θⱼ−θᵢ−φᵢⱼ) at dt=0.005 s.\n"
        "3 Target comparison: wrapped pairwise lags are compared with φᵢⱼ; J_phase and J_amp report the error.\n"
        "4 Motor mapping: each signed leg drive r cosθ maps to two opposite hip/ankle commands in Ant's 8-action order.\n"
        f"5 Environment: env.step(action) advances Ant every {runtime['robot_dt'] or 0.05:.3f} s; viewer.sync() redraws it.\n"
        f"{robot_line}"
    )


def _wire_callbacks(runtime: dict[str, Any]) -> None:
    """Connect sliders, buttons, and gait selection to simulation state."""
    controls = runtime["controls"]

    def preset_changed(_label: str) -> None:
        custom = controls["preset"].value_selected == "custom"
        for key in ("psi_fr", "psi_hl", "psi_hr"):
            controls[key].set_active(custom)
        _update_targets(runtime)

    controls["preset"].on_clicked(preset_changed)

    def custom_phase_changed(_value: float) -> None:
        if controls["preset"].value_selected == "custom":
            _update_targets(runtime)

    for key in ("psi_fr", "psi_hl", "psi_hr"):
        controls[key].on_changed(custom_phase_changed)
        controls[key].set_active(False)

    def perturb(_event: Any) -> None:
        if runtime["state"] is None:
            return
        # Perturb button: add +0.3 cycle = +0.6*pi radians to FR phase.
        runtime["state"][1, 1] += 2.0 * np.pi * 0.3

    def reset(_event: Any) -> None:
        _update_targets(runtime, reset_gait=True)
        runtime["history_t"].clear()
        for history in runtime["history_x"] + runtime["history_r"]:
            history.clear()
        runtime["sim_time"] = 0.0
        if runtime["robot_env"] is not None:
            observation, _info = runtime["robot_env"].reset(seed=42)
            runtime["robot_observation"] = observation
            runtime["robot_accumulator"] = 0.0
            runtime["robot_total_reward"] = 0.0
            runtime["robot_steps"] = 0

    controls["perturb_button"].on_clicked(perturb)
    controls["reset_button"].on_clicked(reset)
    controls["optimise_button"].on_clicked(lambda _event: _optimise(runtime))
    controls["ant_button"].on_clicked(lambda _event: _toggle_ant(runtime))
    runtime["fig"].canvas.mpl_connect("close_event", lambda _event: _disconnect_ant(runtime))
    for key in ("omega", "mu", "alpha", "coupling", "speed"):
        controls[key].on_changed(lambda _value: None)
    _update_targets(runtime, reset_gait=True)


def _optimise(runtime: dict[str, Any]) -> None:
    """Use Nelder-Mead on (frequency, coupling) for the current phase target."""
    controls = runtime["controls"]
    if runtime["state"] is None:
        return
    initial_state = runtime["state"].copy()
    mu_value = float(controls["mu"].val)
    alpha = float(controls["alpha"].val)
    phi = runtime["target_phi"].copy()
    steps = int(round(OPTIMISE_HORIZON / DT))

    def objective(parameters: np.ndarray) -> float:
        # Bounds protect Slider limits; omega is Hz and converted to rad/s inside _run_steps.
        omega_hz = float(np.clip(parameters[0], 0.2, 2.0))
        coupling = float(np.clip(parameters[1], 0.0, 5.0))
        final_state = _run_steps(initial_state, steps, omega_hz, mu_value,
                                 alpha, coupling, phi)
        return cost(final_state[0], final_state[1], mu_value, phi)[0]

    start_parameters = np.array((controls["omega"].val, controls["coupling"].val))
    before = objective(start_parameters)
    result = minimize(objective, start_parameters, method="Nelder-Mead",
                      bounds=((0.2, 2.0), (0.0, 5.0)),
                      options={"maxiter": 80, "xatol": 0.01, "fatol": 1e-5})
    controls["omega"].set_val(float(np.clip(result.x[0], 0.2, 2.0)))
    controls["coupling"].set_val(float(np.clip(result.x[1], 0.0, 5.0)))
    after = objective(result.x)
    runtime["last_optimise"] = (
        f"Nelder-Mead: J {before:.4f} → {after:.4f}; "
        f"ω={controls['omega'].val:.3f} Hz, k={controls['coupling'].val:.3f}.\n"
        "ω is neutral to relative phase locking when every oscillator has the same frequency."
    )


def _draw_matrix_annotations(axis: Any, annotations: list[Any], values: np.ndarray,
                             signed: bool) -> list[Any]:
    """Replace a heatmap's four-by-four numeric labels."""
    for item in annotations:
        item.remove()
    new_items = []
    for row in range(4):
        for col in range(4):
            text_value = f"{values[row, col]:+.2f}" if signed else f"{values[row, col]:.2f}"
            new_items.append(axis.text(col, row, text_value, ha="center", va="center",
                                       fontsize=8, color="black"))
    return new_items


def _animation_frame(_frame: int, runtime: dict[str, Any]) -> list[Any]:
    """Advance simulation by elapsed wall time and redraw every dashboard panel."""
    controls = runtime["controls"]
    now = time.perf_counter()
    elapsed = max(0.0, now - runtime["last_wall_time"])
    runtime["last_wall_time"] = now
    runtime["accumulator"] += elapsed * float(controls["speed"].val)
    steps = int(runtime["accumulator"] / DT)
    runtime["accumulator"] -= steps * DT

    if runtime["state"] is None:
        _update_targets(runtime, reset_gait=True)
    state = runtime["state"]
    omega = np.full(4, 2.0 * np.pi * float(controls["omega"].val))
    mu = np.full(4, float(controls["mu"].val))
    alpha = float(controls["alpha"].val)
    coupling = float(controls["coupling"].val)
    phi = runtime["target_phi"]

    for _ in range(steps):
        state = rk4_step(state, DT, omega, mu, alpha, coupling, phi)
        runtime["sim_time"] += DT
        # Equation 3: x_i = r_i * (1 + cos(theta_i)).
        outputs = state[0] * (1.0 + np.cos(state[1]))
        runtime["history_t"].append(runtime["sim_time"])
        for leg in range(4):
            runtime["history_x"][leg].append(float(outputs[leg]))
            runtime["history_r"][leg].append(float(state[0, leg]))
        # Stage 4/5: send the current CPG output to Ant at its control rate.
        _step_ant(runtime, state)
    runtime["state"] = state
    if steps == 0 and runtime.get("robot_viewer") is not None:
        viewer = runtime["robot_viewer"]
        if viewer.is_running():
            viewer.sync()
        else:
            _disconnect_ant(runtime)

    radii, theta = state
    target_radius = float(np.sqrt(float(controls["mu"].val)))
    target_theta = 2.0 * np.pi * runtime["psi"]
    measured_lag = wrapped_lag(theta)
    phase_error = np.angle(np.exp(1j * (measured_lag - phi)))
    error_units_pi = np.abs(phase_error) / np.pi
    _total_cost, phase_cost, amplitude_cost = cost(radii, theta, mu, phi)

    ax_clock, output_axes, ax_radius, ax_lag, ax_error = runtime["axes"]
    if not runtime["clock_points"]:
        for index, color in enumerate(LEG_COLORS):
            runtime["target_points"].append(ax_clock.plot(
                [target_theta[index]], [target_radius], marker="o", markersize=10,
                markerfacecolor="none", markeredgecolor=color, linestyle="none",
                markeredgewidth=1.5,
            )[0])
            runtime["clock_points"].append(ax_clock.plot(
                [theta[index]], [radii[index]], marker="o", markersize=7,
                color=color, linestyle="none",
            )[0])
    for index in range(4):
        runtime["clock_points"][index].set_data([theta[index]], [radii[index]])
        runtime["target_points"][index].set_data([target_theta[index]], [target_radius])

    # Output panel: four stacked traces, with the assumed stance portions shaded.
    t_values = np.asarray(runtime["history_t"])
    for leg in range(4):
        x_values = np.asarray(runtime["history_x"][leg])
        runtime["output_lines"][leg].set_data(t_values, x_values)
    for axis in output_axes:
        for collection in list(axis.collections):
            collection.remove()
    for leg, axis in enumerate(output_axes):
        # Since x/r = 1 + cos(theta), x < r exactly marks cos(theta) < 0.
        if len(t_values):
            x_values = np.asarray(runtime["history_x"][leg])
            r_values = np.asarray(runtime["history_r"][leg])
            stance_mask = x_values < r_values
            axis.fill_between(t_values, 0.0, 3.8, where=stance_mask,
                              step="mid", color=LEG_COLORS[leg], alpha=0.08)
    if len(t_values):
        left = max(0.0, runtime["sim_time"] - HISTORY_SECONDS)
        for axis in output_axes:
            axis.set_xlim(left, max(left + 0.1, runtime["sim_time"]))
        ax_radius.set_xlim(left, max(left + 0.1, runtime["sim_time"]))
    ax_radius.set_ylim(0.0, max(1.8, float(np.max(radii)) * 1.25, target_radius * 1.25))
    radius_times = np.asarray(runtime["history_t"])
    for leg in range(4):
        runtime["radius_lines"][leg].set_data(radius_times,
                                               np.asarray(runtime["history_r"][leg]))
    runtime["radius_target_line"].set_ydata([target_radius, target_radius])

    lag_units_pi = measured_lag / np.pi
    runtime["lag_image"].set_data(lag_units_pi)
    runtime["error_image"].set_data(error_units_pi)
    runtime["annotations"]["lag"] = _draw_matrix_annotations(
        ax_lag, runtime["annotations"]["lag"], lag_units_pi, signed=True)
    runtime["annotations"]["error"] = _draw_matrix_annotations(
        ax_error, runtime["annotations"]["error"], error_units_pi, signed=False)

    total_cost, phase_cost, amplitude_cost = cost(radii, theta, mu, phi)
    cost_text = (
        f"J = J_phase + J_amp\nJ       = {total_cost:.4f}\n"
        f"J_phase = {phase_cost:.4f}\nJ_amp   = {amplitude_cost:.4f}\n"
        f"√μ      = {target_radius:.3f}\n{runtime['last_optimise']}"
    )
    runtime["cost_text"].set_text(cost_text)

    max_radius_error = float(np.max(np.abs(radii - target_radius)))
    if phase_cost > LOCK_COST_LIMIT and coupling < WEAK_COUPLING_LIMIT:
        explanation = "Coupling too weak to lock phases. Oscillators are drifting."
    elif max_radius_error > 0.05:
        explanation = "Amplitude still converging."
    elif phase_cost < LOCK_COST_LIMIT:
        explanation = "Gait locked to target."
    else:
        explanation = "Phase error remains; increase coupling or allow more settling time."
    runtime["explanation_text"].set_text(explanation)
    runtime["stage_text"].set_text(_stage_breakdown(runtime, state))

    return [*runtime["clock_points"], *runtime["target_points"],
            *runtime["output_lines"], *runtime["radius_lines"],
            runtime["radius_target_line"], runtime["lag_image"],
            runtime["error_image"], runtime["cost_text"],
            runtime["explanation_text"], runtime["stage_text"],
            *runtime["annotations"]["lag"],
            *runtime["annotations"]["error"]]


def main() -> None:
    """Run requested headless checks or start the interactive dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Run numerical checks without opening a window")
    arguments = parser.parse_args()
    if arguments.check:
        raise SystemExit(0 if run_checks() else 1)

    fig, runtime = build_figure()
    from matplotlib.animation import FuncAnimation

    # A short timer interval allows the elapsed-time accumulator to use actual
    # wall time while all oscillator integration still uses the fixed RK4 dt.
    runtime["ani"] = FuncAnimation(fig, _animation_frame, fargs=(runtime,),
                                   interval=30, blit=False, cache_frame_data=False)
    fig._cpg_animation = runtime["ani"]  # Keep a strong reference for pyplot.
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()
