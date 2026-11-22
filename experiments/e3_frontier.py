"""E3 -- exploitation/exploitability frontier as a function of information (RQ2, H1, H3).

Information level n = number of reveals the belief is built from. Beliefs are
sampled by letting a passive observer (plays x*) watch the opponent until the
n-th reveal and freezing the posterior there -- a *measurement device* that
holds the amount of information fixed (feedback #2); the deployed learner in E4
never discards information. n = inf means the true type is known.

For each (archetype, n, belief sample) and alpha on a grid we compute exactly
  G(alpha)   = u(x_alpha, theta*) - u(x*, theta*)          (seat-averaged)
  eps(alpha) = exploitability of x_alpha
plus the price-of-safety / price-of-ignorance terms of Lemma 2, and the
restricted-Nash-response frontier with the true model as a reference.
"""
import sys
import numpy as np
from multiprocessing import Pool
from common import *
from bluff.agents import Bayes, run_match
from bluff.opponents import ARCHETYPES

ALPHAS = np.round(np.linspace(0, 1, 21), 3)
NLIST = [0, 1, 2, 5, 10, 20, 50, "inf"]
P_RNR = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

_ctx = None
_cfg = None


def init(cfg):
    global _ctx, _cfg
    _cfg = cfg
    _ctx = context(cfg)


def u_avg(g, prof, pol):
    return 0.5 * (S.value_vs(g, prof, pol, 0) + S.value_vs(g, prof, pol, 1))


def sample_belief(ctx, k, n, seed):
    if n == 0:
        return ctx.prior if hasattr(ctx, "prior") else np.full(ctx.K, 1 / ctx.K), 0
    if n == "inf":
        b = np.zeros(ctx.K); b[k] = 1.0
        return b, 0
    L = Bayes(ctx, "observer", alpha=0, budget=n)
    rng = np.random.default_rng([seed, k, n, 11])
    L.reset(rng)
    games = 0
    g = ctx.g
    from bluff.agents import sample_game
    while L.reveals < n and games < 2000:
        seat = games % 2
        path = sample_game(g, g.combine(ctx.eq, ctx.pols[k], seat), rng)
        pay = g.payoff[path[-1]] * (1 if seat == 0 else -1)
        L.observe(seat, pay, path, None)
        games += 1
    return L.b, games


def job(args):
    name, n, seed = args
    ctx, g = _ctx, _ctx.g
    k = ctx.thetas.index(ARCHETYPES[name])
    pol = ctx.pols[k]
    b, games = sample_belief(ctx, k, n, seed)
    xbr = ctx.posterior_br(b)
    xb = S.realization_info(g, xbr)
    u_eq = u_avg(g, ctx.eq, pol)
    u_br = u_avg(g, xbr, pol)
    u_eq_s = [S.value_vs(g, ctx.eq, pol, i) for i in (0, 1)]
    u_br_s = [S.value_vs(g, xbr, pol, i) for i in (0, 1)]
    eps = []                                   # rows: alpha; cols: (eps_seat0, eps_seat1)
    for a in ALPHAS:
        prof = ctx.eq if a == 0 else (xbr if a == 1 else S.mix(g, ctx.eq, xbr, a, ctx.xeq, xb))
        eps.append(S.seat_exploitability(g, prof, ctx.vb))
    # Lemma 1(a) check: payoff at alpha = 0.5 equals the average
    lin_err = abs(u_avg(g, S.mix(g, ctx.eq, xbr, 0.5, ctx.xeq, xb), pol) - 0.5 * (u_eq + u_br))
    return dict(name=name, n=n, seed=seed, b=b, games=games, u_eq=u_eq, u_br=u_br,
                u_eq_s=u_eq_s, u_br_s=u_br_s, eps_s=np.array(eps), lin_err=lin_err)


def rnr_job(args):
    name, p = args
    ctx, g = _ctx, _ctx.g
    k = ctx.thetas.index(ARCHETYPES[name])
    pol = ctx.pols[k]
    iters = 300 if g.N < 3e5 else 150
    prof = S.rnr_profile(g, pol, p, iters)
    return dict(name=name, p=p, u=u_avg(g, prof, pol), eps=S.exploitability_max(g, prof, ctx.vb))


def oracle_job(name):
    ctx, g = _ctx, _ctx.g
    k = ctx.thetas.index(ARCHETYPES[name])
    pol = ctx.pols[k]
    xo = ctx.br_to(pol)
    return dict(name=name, u_or=u_avg(g, xo, pol), u_eq=u_avg(g, ctx.eq, pol),
                eps_or=S.exploitability_max(g, xo, ctx.vb))


def main(which):
    cfg = SMALL if which == "small" else LARGE
    seeds = 20 if which == "small" else 8
    ctx = context(cfg)                  # also caches the equilibrium
    eps_eq = ctx.eps_eq
    del ctx
    jobs = [(nm, n, s) for nm in ARCHETYPES for n in NLIST for s in (range(seeds) if n not in (0, "inf") else [0])]
    with Pool(2, initializer=init, initargs=(cfg,)) as pool:
        orc = pool.map(oracle_job, list(ARCHETYPES))
        res = pool.map(job, jobs, chunksize=2)
        print("beliefs done", flush=True)
        rn = pool.map(rnr_job, [(nm, p) for nm in ARCHETYPES for p in P_RNR], chunksize=1)
    save_pickle(dict(cfg=cfg, alphas=ALPHAS, nlist=NLIST, res=res, rnr=rn, oracle=orc, eps_eq=eps_eq),
                f"e3_{which}.pkl")
    print("E3", which, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
