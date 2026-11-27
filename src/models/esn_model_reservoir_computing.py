"""Train an ESN readout on Hopf CPG states and sweep spectral radius.

Run from the repository root:
    python src/models/esn_model_reservoir_computing.py

The Hopf oscillator generates the training inputs and target actuator signals.
For each reservoir spectral radius rho, the reservoir weights stay fixed and
scikit-learn Ridge fits the readout. The best readout and sweep table are saved
under results/models and results. This is a controller-learning demonstration;
the learned actions imitate the Hopf signal and are not MuJoCo-trained policy.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[2]
HOPF_DT = 0.01  # Seconds per generated training step.
HOPF_STEPS = 1800  # 18 seconds gives multiple cycles for training and testing.
HOPF_WARMUP = 200  # Discard the initial radial convergence transient.
HOPF_FREQUENCY_HZ = 1.0
HOPF_ALPHA = 10.0
HOPF_MU = 1.0
RESERVOIR_SIZE = 64
LEAK_RATE = 0.3
RHO_VALUES = np.round(np.arange(0.70, 1.1001, 0.05), 2)


def hopf_rhs(state: np.ndarray, alpha: float, mu: float, omega: float) -> np.ndarray:
    """Return Cartesian Hopf dynamics for each independent oscillator.

    xdot = alpha (mu - x² - y²)x - omega*y
    ydot = alpha (mu - x² - y²)y + omega*x
    """
    x, y = state
    radial_error = mu - x * x - y * y
    return np.stack(
        (alpha * radial_error * x - omega * y,
         alpha * radial_error * y + omega * x)
    )


def rk4_hopf_step(
    state: np.ndarray, dt: float, alpha: float, mu: float, omega: float
) -> np.ndarray:
    """Advance all oscillators one fixed RK4 integration step."""
    k1 = hopf_rhs(state, alpha, mu, omega)
    k2 = hopf_rhs(state + 0.5 * dt * k1, alpha, mu, omega)
    k3 = hopf_rhs(state + 0.5 * dt * k2, alpha, mu, omega)
    k4 = hopf_rhs(state + dt * k3, alpha, mu, omega)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_hopf_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate four phase-offset Hopf states and their target actions.

    Inputs contain only the four observed x coordinates. Targets are the
    next-step x coordinates, so reservoir memory helps infer phase direction
    where the same x value occurs on both sides of a cycle. Phase offsets form
    a trot-like diagonal pair pattern.
    """
    phases = np.array([0.0, np.pi, np.pi, 0.0])
    radius = np.sqrt(HOPF_MU)
    state = np.stack((radius * np.cos(phases), radius * np.sin(phases)))
    omega = 2.0 * np.pi * HOPF_FREQUENCY_HZ
    positions = []
    for step in range(HOPF_STEPS + HOPF_WARMUP):
        state = rk4_hopf_step(state, HOPF_DT, HOPF_ALPHA, HOPF_MU, omega)
        if step >= HOPF_WARMUP:
            positions.append((state[0] / radius).copy())
    # The one-step target is the next Hopf position; no CSV metrics are used.
    positions = np.asarray(positions)
    return positions[:-1], positions[1:]


@dataclass
class EchoStateNetwork:
    """Fixed random reservoir whose action readout is learned with sklearn Ridge."""

    input_dim: int
    output_dim: int
    reservoir_size: int = RESERVOIR_SIZE
    spectral_radius: float = 0.9
    leak_rate: float = LEAK_RATE
    seed: int = 7
    ridge_alpha: float = 1e-3

    def __post_init__(self) -> None:
        """Initialize input/recurrent weights and scale recurrence to rho."""
        if self.input_dim < 1 or self.output_dim < 1 or self.reservoir_size < 1:
            raise ValueError("Input, output, and reservoir dimensions must be positive.")
        if self.spectral_radius <= 0.0 or not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("rho and leak_rate must be positive; leak_rate must be <= 1.")
        rng = np.random.default_rng(self.seed)
        self.input_weights = rng.uniform(
            -0.5, 0.5, (self.reservoir_size, self.input_dim)
        )
        raw_recurrent = rng.normal(
            0.0, 1.0 / np.sqrt(self.reservoir_size),
            (self.reservoir_size, self.reservoir_size),
        )
        raw_radius = np.max(np.abs(np.linalg.eigvals(raw_recurrent)))
        self.reservoir_weights = raw_recurrent * (self.spectral_radius / raw_radius)
        # Echo-state update: h[t] = (1-l)h[t-1] + l*tanh(W_in u[t] + W_res h[t-1]).
        self.readout = Ridge(alpha=self.ridge_alpha)
        self.state = np.zeros(self.reservoir_size)
        self.fitted = False

    def reset(self) -> None:
        """Clear recurrent memory before a new sequence or evaluation split."""
        self.state.fill(0.0)

    def collect_states(self, inputs: np.ndarray) -> np.ndarray:
        """Run an input sequence through the reservoir and return each hidden state."""
        states = np.empty((len(inputs), self.reservoir_size), dtype=np.float64)
        for index, input_row in enumerate(inputs):
            candidate = np.tanh(
                self.input_weights @ input_row
                + self.reservoir_weights @ self.state
            )
            self.state = (
                (1.0 - self.leak_rate) * self.state
                + self.leak_rate * candidate
            )
            states[index] = self.state
        return states

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> "EchoStateNetwork":
        """Fit sklearn's linear readout from reservoir states to target actions."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("inputs must have shape (time, input_dim).")
        if targets.shape != (len(inputs), self.output_dim):
            raise ValueError("targets must have shape (time, output_dim).")
        self.reset()
        states = self.collect_states(inputs)
        # sklearn Ridge computes the trained output weights: action = W_out @ h + b.
        self.readout.fit(states, targets)
        self.fitted = True
        self.reset()
        return self

    def predict_sequence(self, inputs: np.ndarray) -> np.ndarray:
        """Predict actuator signals for an ordered input sequence."""
        if not self.fitted:
            raise RuntimeError("Fit the Ridge readout before prediction.")
        self.reset()
        states = self.collect_states(inputs)
        return np.clip(self.readout.predict(states), -1.0, 1.0)


def sweep_spectral_radius(
    inputs: np.ndarray, targets: np.ndarray
) -> tuple[list[dict[str, float]], EchoStateNetwork]:
    """Train/evaluate one ESN per rho and return the best held-out model."""
    split = int(0.70 * len(inputs))  # Keep time order to avoid random-frame leakage.
    train_x, test_x = inputs[:split], inputs[split:]
    train_y, test_y = targets[:split], targets[split:]
    results: list[dict[str, float]] = []
    best_model = None
    best_loss = float("inf")
    for rho in RHO_VALUES:
        model = EchoStateNetwork(
            input_dim=inputs.shape[1], output_dim=targets.shape[1],
            spectral_radius=float(rho), seed=7,
        )
        model.fit(train_x, train_y)
        prediction = model.predict_sequence(test_x)
        # Ignore the first second of the test segment while recurrent memory warms up.
        scored = slice(round(1.0 / HOPF_DT), None)
        mse = float(np.mean((prediction[scored] - test_y[scored]) ** 2))
        results.append({"rho": float(rho), "test_mse": mse})
        print(f"  rho={rho:.2f}  held-out MSE={mse:.7f}")
        if mse < best_loss:
            best_loss, best_model = mse, model
    assert best_model is not None
    return results, best_model


def save_sweep(
    results: list[dict[str, float]], best: EchoStateNetwork
) -> tuple[Path, Path]:
    """Save rho/error results and the selected trained ESN weights."""
    sweep_path = REPO_ROOT / "results" / "esn_hopf_rho_sweep.csv"
    weights_path = REPO_ROOT / "results" / "models" / "esn_hopf_best_weights.npz"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with sweep_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("rho", "test_mse"))
        writer.writeheader()
        writer.writerows(results)
    np.savez_compressed(
        weights_path,
        rho=best.spectral_radius,
        input_weights=best.input_weights,
        reservoir_weights=best.reservoir_weights,
        readout_weights=best.readout.coef_,
        readout_intercept=best.readout.intercept_,
        ridge_alpha=best.ridge_alpha,
    )
    return sweep_path, weights_path


def three_oscillator_cpg(time_s: float, action_dim: int = 8) -> np.ndarray:
    """Generate a three-group sinusoidal action for a simple speed comparison."""
    phases = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    groups = np.arange(action_dim) % 3
    return 0.5 * np.sin(2.0 * np.pi * HOPF_FREQUENCY_HZ * time_s + phases[groups])


def benchmark_inference(model: EchoStateNetwork, iterations: int = 3000) -> None:
    """Time three CPG groups against the trained ESN on Hopf-state inputs."""
    action_dim = model.output_dim
    observation = np.zeros(model.input_dim)
    start = time.perf_counter()
    for step in range(iterations):
        three_oscillator_cpg(step * HOPF_DT, action_dim)
    cpg_us = (time.perf_counter() - start) * 1e6 / iterations
    start = time.perf_counter()
    for _ in range(iterations):
        # One ESN inference step includes recurrent matrix-vector products and Ridge readout.
        model.reset() if _ == 0 else None
        hidden = np.tanh(model.input_weights @ observation + model.reservoir_weights @ model.state)
        model.state = (1.0 - model.leak_rate) * model.state + model.leak_rate * hidden
        model.readout.predict(model.state.reshape(1, -1))
    esn_us = (time.perf_counter() - start) * 1e6 / iterations
    print("\nInference timing (local CPU microbenchmark; not MuJoCo step time):")
    print(f"  3-group CPG: {cpg_us:.2f} microseconds/step")
    print(f"  {model.reservoir_size}-unit ESN: {esn_us:.2f} microseconds/step")
    print(f"  CPG / ESN time ratio: {cpg_us / esn_us:.3f}x")


def main() -> None:
    """Generate Hopf training sequences, sweep rho, and save learned weights."""
    inputs, targets = make_hopf_dataset()
    print(
        f"Hopf training set: {inputs.shape[0]} timesteps, "
        f"{inputs.shape[1]} Hopf position inputs -> "
        f"{targets.shape[1]} next-step targets"
    )
    print("Edge-of-chaos spectral-radius sweep (rho = 0.70 to 1.10):")
    results, best = sweep_spectral_radius(inputs, targets)
    best_result = min(results, key=lambda row: row["test_mse"])
    sweep_path, weights_path = save_sweep(results, best)
    print(
        f"Best held-out rho={best_result['rho']:.2f}, "
        f"MSE={best_result['test_mse']:.7f}"
    )
    print(f"Readout weight matrix shape: {best.readout.coef_.shape}")
    print(f"Sweep saved to: {sweep_path}")
    print(f"Trained ESN weights saved to: {weights_path}")
    print(
        "The best rho is best only for this generated Hopf task and seed. "
        "The edge of chaos is task- and reservoir-dependent, not a universal rho."
    )
    benchmark_inference(best)


if __name__ == "__main__":
    main()
