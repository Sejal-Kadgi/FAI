"""Safety constraint: every seat (not the seat average) stays within eps_bar;
chord / bisection / per-seat chord are feasible and mutually consistent;
the v* bracket is valid; an infeasible eps_bar raises."""
import numpy as np
import pytest

from bluff import solvers as S
from bluff.agents import alpha_cap, Bayes, InfeasibleSafetyLevel, run_match

EPS_LEVELS = (0.02, 0.05, 0.1, 0.3)


def test_vstar_bracket(g, eq, ctx):
    lo, hi = ctx.vb
    v = S.expected_value(g, eq)
    assert lo - 1e-12 <= v <= hi + 1e-12
    assert np.isclose(hi - lo, 2 * S.exploitability(g, eq))           # width = 2 eps_avg(x*)


def test_seat_exploitability_bounds_are_consistent(g, eq, ctx):
    """The certified per-seat bounds average to eps_avg plus half the bracket
    width, and dominate the seat average."""
    rng = np.random.default_rng(4)
    lo, hi = ctx.vb
    for _ in range(3):
        x = g.normalise(rng.random(g.nseq))
        e0, e1 = S.seat_exploitability(g, x, ctx.vb)
        avg = S.exploitability(g, x)
        assert np.isclose(0.5 * (e0 + e1), avg + 0.5 * (hi - lo))
        assert max(e0, e1) >= avg


@pytest.mark.parametrize("k", range(25))
def test_all_methods_keep_both_seats_within_cap(g, ctx, k):
    for b in (np.eye(25)[k], 0.5 * np.eye(25)[k] + 0.5 / 25):
        xbr = ctx.posterior_br(b)
        for eb in EPS_LEVELS:
            a_c, _ = alpha_cap(ctx, xbr, eb, "chord")
            a_b, _ = alpha_cap(ctx, xbr, eb, "bisect")
            a_s, _ = alpha_cap(ctx, xbr, eb, "chord-seat")
            # consistency: chord <= bisection; per-seat weights >= common chord
            assert a_c <= a_b + 1e-12 and min(a_s) >= a_c - 1e-12
            for a in (a_c, a_b, a_s):
                e0, e1 = S.seat_exploitability(g, S.mix(g, ctx.eq, xbr, a, ctx.xeq), ctx.vb)
                assert e0 <= eb + 1e-9 and e1 <= eb + 1e-9


def test_average_metric_would_not_be_safe(g, ctx):
    """Regression for the corrected bug: a weight chosen on the seat AVERAGE
    can leave one seat above eps_bar; the eps_max chord never does."""
    found = False
    for k in range(25):
        xbr = ctx.posterior_br(np.eye(25)[k])
        e_star, e_br = S.exploitability(g, ctx.eq), S.exploitability(g, xbr)
        eb = 0.05
        if e_br <= eb:
            continue
        a_avg = (eb - e_star) / (e_br - e_star)
        e0, e1 = S.seat_exploitability(g, S.mix(g, ctx.eq, xbr, a_avg, ctx.xeq), ctx.vb)
        if max(e0, e1) > eb + 1e-6:
            found = True
            a, _ = alpha_cap(ctx, xbr, eb, "chord")
            assert max(S.seat_exploitability(g, S.mix(g, ctx.eq, xbr, a, ctx.xeq), ctx.vb)) <= eb + 1e-9
    assert found


def test_infeasible_level_raises(ctx):
    xbr = ctx.posterior_br(np.full(25, 1 / 25))
    tiny = ctx.eps_eq / 2
    with pytest.raises(InfeasibleSafetyLevel):
        alpha_cap(ctx, xbr, tiny)
    with pytest.raises(InfeasibleSafetyLevel):
        Bayes(ctx, "safe", eps_bar=tiny)


def test_capped_learners_in_a_match(g, ctx):
    k = ctx.thetas.index((0.9, 0.1))
    for cap in ("chord", "chord-seat"):
        L = Bayes(ctx, "safe", alpha=1.0, eps_bar=0.05, cap=cap)
        recs = run_match(ctx, L, ctx.pols[k], 20, np.random.default_rng(5), track_eps=True)
        assert max(r["eps0"] for r in recs) <= 0.05 + 1e-9
        assert max(r["eps1"] for r in recs) <= 0.05 + 1e-9
        assert all(r["eps"] == max(r["eps0"], r["eps1"]) for r in recs)
