"""Shared interface and utilities for robot action controllers."""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """Base class that gives every controller the same action interface."""

    def __init__(self, action_space):
        # Step 1: keep the environment's action space so subclasses know the
        # shape and legal range of the commands they must produce.
        self.action_space = action_space
        self.action_low = np.asarray(action_space.low, dtype=np.float32)
        self.action_high = np.asarray(action_space.high, dtype=np.float32)

        # Step 2: flatten the action shape to count individual actuators.
        self.action_dim = int(np.prod(action_space.shape))

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Convert the current observation into an action."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Clear any internal state between episodes."""
        raise NotImplementedError

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Reshape and clamp an action to the environment's legal limits."""
        # Step 3: make sure the generated command has the expected shape.
        action = np.asarray(action, dtype=np.float32).reshape(
            self.action_space.shape
        )

        # Step 4: clamp each actuator independently, then return float32 data.
        return np.clip(action, self.action_low, self.action_high).astype(
            np.float32
        )
