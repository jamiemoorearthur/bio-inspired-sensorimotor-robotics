"""A small echo-state-network style controller with a fixed reservoir."""

import numpy as np

from .base_controller import BaseController


class ReservoirController(BaseController):
    """Map observations to actions through a recurrent reservoir.

    The reservoir weights stay fixed; ``readout_weights`` can be supplied
    from a trained model. If they are omitted, a seeded random readout is
    used as a simple runnable baseline, rather than a trained policy.
    """

    def __init__(
        self,
        action_space,
        reservoir_size=100,
        input_dim=None,
        spectral_radius=0.9,
        input_scale=1.0,
        leak_rate=1.0,
        readout_weights=None,
        seed=None,
    ):
        # Step 1: reuse the base class to learn the action shape and bounds.
        super().__init__(action_space)

        # Step 2: validate the settings that control the recurrent state.
        if reservoir_size < 1:
            raise ValueError("reservoir_size must be at least 1.")
        if not 0.0 < spectral_radius:
            raise ValueError("spectral_radius must be greater than 0.")
        if not 0.0 < leak_rate <= 1.0:
            raise ValueError("leak_rate must be in the interval (0, 1].")

        self.reservoir_size = int(reservoir_size)
        self.input_dim = input_dim
        self.spectral_radius = float(spectral_radius)
        self.input_scale = float(input_scale)
        self.leak_rate = float(leak_rate)

        # Step 3: use a local random generator so seeded controllers are
        # repeatable without changing NumPy's global random state.
        self.rng = np.random.default_rng(seed)

        # Step 4: create fixed input-to-reservoir and reservoir-to-reservoir
        # connections. The recurrent matrix is scaled to the requested
        # spectral radius, which controls how long activity tends to persist.
        self.input_weights = None
        self.reservoir_weights = self.rng.normal(
            0.0, 1.0 / np.sqrt(self.reservoir_size),
            (self.reservoir_size, self.reservoir_size),
        ).astype(np.float32)
        largest_eigenvalue = np.max(np.abs(np.linalg.eigvals(self.reservoir_weights)))
        if largest_eigenvalue > 0:
            self.reservoir_weights *= self.spectral_radius / largest_eigenvalue

        # Step 5: load a trained observation-to-action readout, or create a
        # seeded random one so this class can be exercised before training.
        if readout_weights is None:
            self.readout_weights = self.rng.normal(
                0.0, 1.0 / np.sqrt(self.reservoir_size),
                (self.action_dim, self.reservoir_size),
            ).astype(np.float32)
        else:
            self.readout_weights = np.asarray(readout_weights, dtype=np.float32)
            if self.readout_weights.shape != (self.action_dim, self.reservoir_size):
                raise ValueError(
                    "readout_weights must have shape "
                    f"({self.action_dim}, {self.reservoir_size})."
                )

        # Step 6: start with no activity in the reservoir's memory.
        self.state = np.zeros(self.reservoir_size, dtype=np.float32)

        # If the observation size is known now, prepare its input connections.
        if self.input_dim is not None:
            self._configure_input_weights(self.input_dim)

    def _configure_input_weights(self, input_dim):
        """Create input connections once the observation size is known."""
        input_dim = int(input_dim)
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1.")

        # Step 7: create weights with one column per observed sensor value.
        self.input_dim = input_dim
        self.input_weights = self.rng.uniform(
            -self.input_scale,
            self.input_scale,
            (self.reservoir_size, self.input_dim),
        ).astype(np.float32)

    def act(self, observation):
        # Step 8: flatten the sensor reading to a single vector.
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)

        # If the observation size was not supplied, learn it on the first call.
        if self.input_weights is None:
            self._configure_input_weights(observation.size)
        elif observation.size != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} observation values, "
                f"received {observation.size}."
            )

        # Step 9: combine current sensor input and previous reservoir activity,
        # then squash it through tanh so each internal unit stays bounded.
        new_state = np.tanh(
            self.input_weights @ observation
            + self.reservoir_weights @ self.state
        )

        # Step 10: blend old and new activity. A smaller leak rate makes the
        # reservoir change more gradually and retain more of its past state.
        self.state = (
            (1.0 - self.leak_rate) * self.state
            + self.leak_rate * new_state
        ).astype(np.float32)

        # Step 11: the readout turns reservoir activity into actuator commands.
        # Tanh keeps these baseline commands within [-1, 1]; the base class
        # then clips or reshapes them to match the environment's action space.
        action = np.tanh(self.readout_weights @ self.state)
        return self.clip_action(action)

    def reset(self):
        # Step 12: forget the previous episode while keeping the learned
        # connections and readout intact.
        self.state.fill(0.0)
