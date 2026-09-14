import numpy as np
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/simulated.toml")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
