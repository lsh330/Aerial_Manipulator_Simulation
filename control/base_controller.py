"""Abstract base controller with Template Method pattern."""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """Template Method pattern for controllers.

    Defines the update flow: compute_error → compute_control → apply_saturation.
    Subclasses implement _compute_control() with their specific control law.
    """

    def __init__(self, saturation_limits: np.ndarray | None = None):
        self._sat_limits = saturation_limits
        self._integral_error = None

    def update(self, state: np.ndarray, reference: dict, dt: float) -> np.ndarray:
        """Main control update (template method)."""
        error = self._compute_error(state, reference)
        raw_output = self._compute_control(error, dt)
        return self._apply_saturation(raw_output)

    @abstractmethod
    def _compute_error(self, state: np.ndarray, reference: dict) -> dict:
        """Compute error signal from state and reference."""
        ...

    @abstractmethod
    def _compute_control(self, error: dict, dt: float) -> np.ndarray:
        """Compute raw control output from error."""
        ...

    def _apply_saturation(self, output: np.ndarray) -> np.ndarray:
        if self._sat_limits is not None:
            return np.clip(output, -self._sat_limits, self._sat_limits)
        return output

    def reset(self):
        """Reset controller internal state (integrators, etc.)."""
        self._integral_error = None
