"""Memory of the temperature-tracking error, shared by the PID and the observations."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_ERROR_WINDOW = 12  # steps; the PID gains of results/expert were tuned with it


@dataclass(frozen=True)
class ErrorState:
    """Tracking error of every house at one step, ``(N,)`` arrays.

    Attributes
    ----------
    error : NDArray
        ``setpoint + offset - temperature``.
    integral : NDArray
        Sum of the errors of the last ``window`` steps, the current one included.
    derivative : NDArray
        Change of the error without offset since the previous step; at the first
        step the previous error is taken as 0.
    """

    error: NDArray
    integral: NDArray
    derivative: NDArray


class TrackingErrorFilter:
    """Windowed integral and derivative of the tracking error of ``N`` houses.

    Call ``update`` once per step with the error ``setpoint - temperature``. A
    repeated step returns the cached state, and step 0 (or any step before the
    last one) starts a new episode, so ``reset`` is only needed to forget the past
    explicitly. An edge in continuous operation passes a step counter that never
    goes back and is never reset.

    The arithmetic is float32 and follows the order the environment used, so
    tuned gains give the same actions.

    Parameters
    ----------
    window : int, default=DEFAULT_ERROR_WINDOW
        Number of steps in the integral.
    """

    def __init__(self, window: int = DEFAULT_ERROR_WINDOW) -> None:
        if window <= 0:
            raise ValueError("window must be positive.")
        self.window = window
        self.reset()

    def reset(self) -> None:
        """Forget the errors and offsets of the past steps."""
        self._errors: deque[NDArray] = deque(maxlen=self.window)
        self._offsets: deque[NDArray] = deque(maxlen=self.window)
        self._step: int | None = None
        self._state: ErrorState | None = None

    @property
    def state(self) -> ErrorState:
        """Return the state of the last ``update``."""
        if self._state is None:
            raise RuntimeError("update must be called before reading the state.")
        return self._state

    def update(
        self, step: int, error: ArrayLike, offset: ArrayLike = 0.0
    ) -> ErrorState:
        """Add the error of ``step`` and return the state.

        Parameters
        ----------
        step : int
            Decision step; a value below the previous one starts a new episode, and
            the same value returns the cached state.
        error : ArrayLike
            ``setpoint - temperature`` of every house, shape ``(N,)``.
        offset : ArrayLike, default=0.0
            Setpoint offset of every house at this step (°C), added to the error
            and the integral but not the derivative.
        """
        if step == self._step:
            return self.state
        if self._step is None or step < self._step:
            self.reset()
        error = np.asarray(error, dtype=np.float32)
        previous = self._errors[-1] if self._errors else np.zeros_like(error)
        self._errors.append(error)
        self._offsets.append(
            np.broadcast_to(np.asarray(offset, np.float64), error.shape)
        )
        offsets = self._offsets[0]
        for past in list(self._offsets)[1:]:
            offsets = offsets + past  # oldest first
        integral = np.sum(np.stack(self._errors, axis=1), axis=1)
        self._step = step
        self._state = ErrorState(
            error=error + self._offsets[-1].astype(np.float32),
            integral=integral + offsets.astype(np.float32),
            derivative=error - previous,
        )
        return self._state
