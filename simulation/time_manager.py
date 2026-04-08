"""Time step management and event scheduling."""


class TimeManager:
    """Manages simulation time progression and periodic events."""

    def __init__(self, dt: float, duration: float, log_interval: int = 1):
        self._dt = dt
        self._duration = duration
        self._log_interval = log_interval
        self._step = 0
        self._t = 0.0

    @property
    def t(self) -> float:
        return self._t

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def total_steps(self) -> int:
        return int(self._duration / self._dt)

    def is_finished(self) -> bool:
        return self._t >= self._duration - 1e-12

    def should_log(self) -> bool:
        return self._step % self._log_interval == 0

    def advance(self):
        self._step += 1
        self._t = self._step * self._dt

    def progress(self) -> float:
        """Progress fraction [0, 1]."""
        return min(self._t / self._duration, 1.0)
