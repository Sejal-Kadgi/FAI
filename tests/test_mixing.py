"""Realization-plan mixing (eq. 5) and Lemma 1(a)."""
import numpy as np

from bluff import solvers as S


def _br_profile(g, opp):
    return g.combine(S.best_response(g, 0, S.opp_reach(g, opp, 0))[0],
                     S.best_response(g, 1, S.opp_reach(g, opp, 1))[0], 0)


def test_mixture_realization_plan_is_convex_combination(g, eq, ctx):
    xbr = _br_profile(g, ctx.pols[3])
    xa, xb = S.realization_info(g, eq), S.realization_info(g, xbr)
    for a in (0.0, 0.25, 0.6, 1.0):
        xm = S.realization_info(g, S.mix(g, eq, xbr, a, xa, xb))
        assert np.allclose(xm, (1 - a) * xa + a * xb, atol=1e-12)


def test_endpoints_and_valid_strategy(g, eq, ctx):
    xbr = _br_profile(g, ctx.pols[3])
    m0, m1 = S.mix(g, eq, xbr, 0.0), S.mix(g, eq, xbr, 1.0)
    x0, x1 = S.realization_info(g, eq), S.realization_info(g, xbr)
    reach0, reach1 = (x0 > 0)[g.seq_info], (x1 > 0)[g.seq_info]
    assert np.allclose(m0[reach0], eq[reach0]) and np.allclose(m1[reach1], xbr[reach1])
    m = S.mix(g, eq, xbr, 0.4)
    assert np.allclose(np.bincount(g.seq_info, weights=m, minlength=g.n_info), 1.0)


def test_payoff_linear_in_alpha(g, eq, ctx):
    for k in (0, 18):
        opp = ctx.pols[k]
        xbr = _br_profile(g, opp)
        for seat in (0, 1):
            u = [S.value_vs(g, S.mix(g, eq, xbr, a), opp, seat) for a in (0.0, 0.3, 1.0)]
            assert np.isclose(u[1], 0.7 * u[0] + 0.3 * u[2], atol=1e-12)


def test_exploitability_convex_in_alpha(g, eq, ctx):
    xbr = _br_profile(g, ctx.pols[18])
    al = np.linspace(0, 1, 11)
    E = np.array([S.seat_exploitability(g, S.mix(g, eq, xbr, a), ctx.vb) for a in al])
    assert (np.diff(E, 2, axis=0) >= -1e-9).all()      # each seat convex


def test_per_seat_alpha_mixes_seats_independently(g, eq, ctx):
    xbr = _br_profile(g, ctx.pols[3])
    m = S.mix(g, eq, xbr, (0.2, 0.7))
    s0, s1 = g.seq_player == 0, g.seq_player == 1
    assert np.allclose(m[s0], S.mix(g, eq, xbr, 0.2)[s0])
    assert np.allclose(m[s1], S.mix(g, eq, xbr, 0.7)[s1])
