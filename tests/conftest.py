"""Shared fixtures. Every test builds what it needs from scratch on the tiny
game Bluff(3,2,2,4) (10,738 nodes); nothing is read from results/ or a cache."""
import numpy as np
import pytest

from bluff.game import Game
from bluff import solvers as S
from bluff.agents import Context

TINY = (3, 2, 2, 4)


@pytest.fixture(scope="session")
def g():
    return Game(*TINY)


@pytest.fixture(scope="session")
def g_obs():
    """Same game, keeping the readable observation records."""
    return Game(*TINY, keep_obs=True)


@pytest.fixture(scope="session")
def eq(g):
    return S.cfr(g, 400, plus=True)[0]


@pytest.fixture(scope="session")
def ctx(g, eq):
    return Context(g, eq)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
