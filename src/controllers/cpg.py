# src/controllers/cpg.py

import numpy as np
from .base_controller import BaseController


class CPGController(BaseController):
    """
    Central Pattern Generator controller.

    Generates rhythmic actuator commands using sinusoidal oscillators.
    """

    def __init__(
        self,
        action_space,
        amplitude=0.5,
        frequency=1.0,
        phase_offsets=None,
        dt=0.01
    ):

        super().__init__(action_space)

        self.dt = dt

        # Configure oscillator amplitudes
        self.amplitude = np.broadcast_to(
            np.asarray(amplitude, dtype=np.float32),
            (self.action_dim,)
        ).copy()

        # Configure oscillator frequencies
        self.frequency = np.broadcast_to(
            np.asarray(frequency, dtype=np.float32),
            (self.action_dim,)
        ).copy()

        # Configure phase offsets
        if phase_offsets is None:
            phase_offsets = np.linspace(
                0,
                2 * np.pi,
                self.action_dim,
                endpoint=False
            )

        self.phase_offsets = np.asarray(
            phase_offsets, dtype=np.float32
        )

        if self.phase_offsets.shape != (self.action_dim,):
            raise ValueError(
                "phase_offsets must contain one value per actuator."
            )

        self.time = 0.0

    def act(self, observation):

        # Generate one rhythmic signal per actuator
        phases = (
            2 * np.pi * self.frequency * self.time
            + self.phase_offsets
        )

        action = self.amplitude * np.sin(phases)

        self.time += self.dt

        return self.clip_action(action)

    def reset(self):

        self.time = 0.0