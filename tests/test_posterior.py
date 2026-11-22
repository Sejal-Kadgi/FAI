"""Bayesian filter: normalisation, agreement with the game engine (simulation),
exact Bayes rule over the type grid, and concentration on the true type."""
import numpy as np

from bluff import solvers as S
from bluff.agents import Bayes, run_match, sample_game, path_reveals, reveal_cut
from bluff.opponents import type_policy


def test_likelihood_matches_simulation(g, eq, ctx, rng):
    """P(agent's final observation class | type k) predicted by the filter
    (L[k, c] times the agent's own reach, constant within c) equals the
    empirical frequency when games are SAMPLED from the engine."""
    k, seat = 7, 0
    opp = ctx.pols[k]
    full = g.combine(eq, opp, seat)
    n = 20000
    counts = {}
    for _ in range(n):
        c = g.oc[seat][sample_game(g, full, rng)[-1]]
        counts[c] = counts.get(c, 0) + 1
    own = S.reach(g, eq, (seat,), chance=False)
    own_cls = {g.oc[seat][z]: own[z] for z in g.terms}
    for c in sorted(counts, key=lambda c: -counts[c])[:8]:
        pred = ctx.L[seat][k, c] * own_cls[c]
        se = np.sqrt(pred * (1 - pred) / n)
        assert abs(counts[c] / n - pred) < 5 * se + 1e-4


def test_predicted_class_probabilities_sum_to_one(g, eq, ctx):
    own = S.reach(g, eq, (0,), chance=False)
    for k in (0, 12, 24):
        p = {}
        for z in g.terms:
            c = g.oc[0][z]
            p[c] = ctx.L[0][k, c] * own[z]
        assert np.isclose(sum(p.values()), 1.0)


def test_update_is_bayes_rule_and_normalised(g, eq, ctx, rng):
    L = Bayes(ctx, "obs", alpha=0)
    L.reset(rng)
    manual = np.full(ctx.K, 1.0 / ctx.K)
    opp = type_policy(g, 0.7, 0.3)
    for k in range(12):
        seat = k % 2
        path = sample_game(g, g.combine(eq, opp, seat), rng)
        L.observe(seat, 0.0, path, None)
        c = g.oc[seat][path[-1]]
        manual = manual * ctx.L[seat][:, c]
        manual /= manual.sum()
        b = L.b
        assert np.isfinite(b).all() and np.isclose(b.sum(), 1.0) and (b >= 0).all()
        assert np.allclose(b, manual, atol=1e-10)


def test_budget_freezes_at_the_reveal(g, eq, ctx, rng):
    """Measurement device (E2/E3): with budget B the belief uses the
    observation class at the node of the B-th reveal and then stops."""
    opp = type_policy(g, 0.5, 0.5)
    while True:
        path = sample_game(g, g.combine(eq, opp, 0), rng)
        if path_reveals(g, path, 0) >= 1:
            break
    L = Bayes(ctx, "obs", alpha=0, budget=1)
    L.reset(rng)
    L.observe(0, 0.0, path, None)
    cut = reveal_cut(g, path, 0, 1)
    expect = ctx.L[0][:, g.oc[0][cut]] / ctx.L[0][:, g.oc[0][cut]].sum()
    assert np.allclose(L.b, expect)
    before = L.b.copy()
    L.observe(1, 0.0, sample_game(g, g.combine(eq, opp, 1), rng), None)
    assert np.allclose(L.b, before)                # frozen


def test_posterior_concentrates(g, ctx):
    for th in [(0.1, 0.1), (0.9, 0.9)]:
        k = ctx.thetas.index(th)
        recs = run_match(ctx, Bayes(ctx, "obs", alpha=0), ctx.pols[k], 200, np.random.default_rng(3))
        b = recs[-1]["b"]
        assert np.argmax(b) == k and b[k] > 0.5
