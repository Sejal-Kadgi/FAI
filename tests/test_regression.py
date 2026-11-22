"""Regression: the vectorised engine reproduces the original node-by-node
implementation (tests/reference/bluff_v0.py) exactly."""
import numpy as np
import pytest

import bluff_v0 as V0
from bluff import solvers as S
from conftest import TINY


@pytest.fixture(scope="module")
def pair(g):
    g0 = V0.Game(*TINY)
    info_map = {n.info: g.info[n.id] for n in g0.nodes if n.kind in ("play", "resp")}
    return g0, info_map


def to_flat(g, info_map, strat0):
    flat = np.zeros(g.nseq)
    for I0, I in info_map.items():
        flat[g.info_off[I]:g.info_off[I] + g.info_nacts[I]] = strat0[I0]
    return flat


def test_same_tree(g, pair):
    g0, _ = pair
    assert g.N == len(g0.nodes) and g.n_info == g0.n_info
    kinds = {"chance": 0, "play": 1, "resp": 2, "term": 3}
    assert all(g.kind[n.id] == kinds[n.kind] for n in g0.nodes)
    assert all(g.payoff[n.id] == n.payoff for n in g0.nodes if n.kind == "term")


def test_values_best_responses_exploitability(g, pair):
    g0, info_map = pair
    rng = np.random.default_rng(0)
    for _ in range(3):
        strat0 = [rng.dirichlet(np.ones(k)) for k in g0.info_actions]
        flat = to_flat(g, info_map, strat0)
        sig0 = g0.sigma_from_info(strat0)
        assert np.isclose(g0.expected_value(sig0), S.expected_value(g, flat), atol=1e-12)
        assert np.isclose(g0.best_response_value(0, sig0), S.br_value(g, 0, flat), atol=1e-12)
        assert np.isclose(g0.best_response_value(1, sig0), S.br_value(g, 1, flat), atol=1e-12)
        assert np.isclose(g0.exploitability(sig0), S.exploitability(g, flat), atol=1e-12)


def test_realization_plan_mixing(g, pair):
    g0, info_map = pair
    rng = np.random.default_rng(1)
    a = [rng.dirichlet(np.ones(k)) for k in g0.info_actions]
    b = [rng.dirichlet(np.ones(k)) for k in g0.info_actions]
    m0 = g0.mix(a, b, 0.3)
    m = S.mix(g, to_flat(g, info_map, a), to_flat(g, info_map, b), 0.3)
    assert np.allclose(to_flat(g, info_map, m0), m, atol=1e-12)


def test_plain_cfr_matches(g, pair):
    g0, info_map = pair
    avg0, _ = g0.cfr(20)
    avg, _ = S.cfr(g, 20, plus=False)
    assert np.allclose(to_flat(g, info_map, avg0), avg, atol=1e-10)


def test_cfr_plus_converges(g):
    _, hist = S.cfr(g, 300, plus=True, log_every=100)
    eps = [e for _, e in hist]
    assert eps[-1] < eps[0] and eps[-1] < 1e-3


def test_best_response_dominates(g, eq):
    """No strategy beats the exact best response against a fixed opponent."""
    rng = np.random.default_rng(2)
    for i in (0, 1):
        q = S.opp_reach(g, eq, i)
        br, val = S.best_response(g, i, q)
        assert np.isclose(S.value_vs(g, br, eq, i), val, atol=1e-12)
        for _ in range(5):
            x = g.normalise(rng.random(g.nseq))
            assert S.value_vs(g, x, eq, i) <= val + 1e-12
