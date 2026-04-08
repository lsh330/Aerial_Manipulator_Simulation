"""Real-time data logging during simulation (Observer pattern)."""

import numpy as np
from pathlib import Path
from models.state import STATE_DIM


class DataLogger:
    """Records simulation data at each logged time step.

    Stores time, state, input, and reference trajectories
    in pre-allocated numpy arrays for efficiency.
    """

    INPUT_DIM = 6

    def __init__(self, capacity: int = 100000):
        self._capacity = capacity
        self._count = 0
        self._time = np.zeros(capacity)
        self._states = np.zeros((capacity, STATE_DIM))
        self._inputs = np.zeros((capacity, self.INPUT_DIM))
        self._references = {}

    def on_step(self, t: float, state: np.ndarray, input_vec: np.ndarray,
                reference: dict | None = None):
        """Record one time step of data."""
        if self._count >= self._capacity:
            self._expand()

        self._time[self._count] = t
        self._states[self._count] = state
        self._inputs[self._count] = input_vec

        if reference:
            for key, val in reference.items():
                if key not in self._references:
                    self._references[key] = np.zeros((self._capacity, np.asarray(val).size))
                arr = self._references[key]
                if self._count >= arr.shape[0]:
                    self._references[key] = np.vstack([arr, np.zeros_like(arr)])
                    arr = self._references[key]
                arr[self._count] = np.asarray(val).flatten()

        self._count += 1

    def _expand(self):
        new_cap = self._capacity * 2
        self._time = np.concatenate([self._time, np.zeros(self._capacity)])
        self._states = np.vstack([self._states, np.zeros((self._capacity, STATE_DIM))])
        self._inputs = np.vstack([self._inputs, np.zeros((self._capacity, self.INPUT_DIM))])
        self._capacity = new_cap

    @property
    def count(self) -> int:
        return self._count

    def get_time(self) -> np.ndarray:
        return self._time[:self._count]

    def get_states(self) -> np.ndarray:
        return self._states[:self._count]

    def get_inputs(self) -> np.ndarray:
        return self._inputs[:self._count]

    def get_reference(self, key: str) -> np.ndarray | None:
        if key in self._references:
            return self._references[key][:self._count]
        return None

    def to_dataframe(self):
        """Convert logged data to a pandas DataFrame."""
        import pandas as pd
        cols = ["t"]
        data = [self.get_time()]

        state_names = [
            "x", "y", "z", "vx", "vy", "vz",
            "qw", "qx", "qy", "qz",
            "wx", "wy", "wz",
            "q1", "q2", "q1_dot", "q2_dot",
        ]
        for i, name in enumerate(state_names):
            cols.append(name)
            data.append(self.get_states()[:, i])

        input_names = ["f1", "f2", "f3", "f4", "tau_q1", "tau_q2"]
        for i, name in enumerate(input_names):
            cols.append(name)
            data.append(self.get_inputs()[:, i])

        return pd.DataFrame(dict(zip(cols, data)))

    def save_hdf5(self, path: str | Path):
        import h5py
        path = Path(path)
        with h5py.File(path, "w") as f:
            f.create_dataset("time", data=self.get_time())
            f.create_dataset("states", data=self.get_states())
            f.create_dataset("inputs", data=self.get_inputs())
            for key, val in self._references.items():
                f.create_dataset(f"ref/{key}", data=val[:self._count])

    def save_csv(self, path: str | Path):
        self.to_dataframe().to_csv(path, index=False)
