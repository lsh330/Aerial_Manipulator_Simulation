"""Unit tests for DataLogger."""

import numpy as np
import pytest
from analysis.data_logger import DataLogger


class TestDataLogger:
    def test_initial_empty(self):
        logger = DataLogger()
        assert logger.count == 0

    def test_on_step_increments(self):
        logger = DataLogger()
        logger.on_step(0.0, np.zeros(17), np.zeros(6))
        assert logger.count == 1

    def test_get_time(self):
        logger = DataLogger()
        for i in range(5):
            logger.on_step(i * 0.1, np.zeros(17), np.zeros(6))
        t = logger.get_time()
        np.testing.assert_allclose(t, [0, 0.1, 0.2, 0.3, 0.4])

    def test_state_recording(self):
        logger = DataLogger()
        state = np.arange(17, dtype=float)
        logger.on_step(0.0, state, np.zeros(6))
        np.testing.assert_array_equal(logger.get_states()[0], state)

    def test_to_dataframe_columns(self):
        logger = DataLogger()
        logger.on_step(0.0, np.zeros(17), np.zeros(6))
        df = logger.to_dataframe()
        assert "t" in df.columns
        assert "x" in df.columns
        assert "qw" in df.columns
        assert "f1" in df.columns
        assert "tau_q1" in df.columns

    def test_auto_expand(self):
        logger = DataLogger(capacity=2)
        for i in range(5):
            logger.on_step(float(i), np.zeros(17), np.zeros(6))
        assert logger.count == 5

    def test_save_csv(self, tmp_path):
        logger = DataLogger()
        logger.on_step(0.0, np.zeros(17), np.zeros(6))
        path = tmp_path / "test.csv"
        logger.save_csv(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_reference_recording(self):
        logger = DataLogger()
        ref = {"position": np.array([1.0, 2.0, 3.0])}
        logger.on_step(0.0, np.zeros(17), np.zeros(6), ref)
        pos_ref = logger.get_reference("position")
        np.testing.assert_array_equal(pos_ref[0], [1, 2, 3])
