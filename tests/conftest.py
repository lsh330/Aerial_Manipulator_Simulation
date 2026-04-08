"""Shared pytest fixtures for aerial manipulator tests."""

import pytest
import numpy as np
import yaml
from pathlib import Path


CONFIG_DIR = Path(__file__).parent.parent / "config"


@pytest.fixture
def default_params():
    """Load default physical parameters."""
    with open(CONFIG_DIR / "default_params.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def controller_params():
    """Load controller parameters."""
    with open(CONFIG_DIR / "controller_params.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def simulation_params():
    """Load simulation parameters."""
    with open(CONFIG_DIR / "simulation_params.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def hover_state():
    """State vector at hover equilibrium (z=1m, identity quaternion, arm down)."""
    state = np.zeros(17)
    state[2] = 1.0       # z = 1m
    state[6] = 1.0       # qw = 1 (identity quaternion)
    return state


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for test artifacts."""
    dirs = {
        "images": tmp_path / "images",
        "animations": tmp_path / "animations",
        "reports": tmp_path / "reports",
    }
    for d in dirs.values():
        d.mkdir()
    return dirs
