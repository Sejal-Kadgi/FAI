"""E2 -- identification (RQ1, H2).

For each archetype theta*, play matches in which the agent observes the
opponent and updates the exact posterior over the 25-type grid (hidden cards
marginalised). Two observation policies:
  passive : agent plays the equilibrium x* (pure observer, alpha = 0)
  active  : agent plays the safe learner with information-seeking calls
            (lambda > 0, alpha capped by eps_bar), which buys extra showdowns.
Logged per game: b(theta*), b(E_eta(theta*)), entropy, |E[beta]-beta*|,
|E[gamma]-gamma*|, cumulative reveals.
Also: the payoff-equivalence matrix loss[theta, theta*] =
u(BR_theta*, theta*) - u(BR_theta, theta*), which defines E_eta.
"""
import sys
import numpy as np
from multiprocessing import Pool
from common import *
from bluff.agents import Bayes, run_match, entropy
from bluff.opponents import ARCHETYPES

G_GAMES = 150
SEEDS = 30
ETA = 0.01
LAM = 0.2

_ctx = None


def init():
    global _ctx
    _ctx = context(SMALL)


def loss_matrix(ctx):
    g = ctx.g
    brs = [ctx.br_to(p) for p in ctx.pols]
    U = np.zeros((ctx.K, ctx.K))          # U[a, t] = u(BR_a vs type t), seat-averaged
    for t, pol in enumerate(ctx.pols):
        for a in range(ctx.K):
            U[a, t] = 0.5 * (S.value_vs(g, brs[a], pol, 0) + S.value_vs(g, brs[a], pol, 1))
    return U.diagonal()[None, :] - U      # loss[a, t] >= 0


def job(args):
    name, mode, seed, loss = args
    ctx = _ctx
    th = ARCHETYPES[name]
    k = ctx.thetas.index(th)
    opp = ctx.pols[k]
    E = loss[:, k] <= ETA
    if mode == "passive":
        L = Bayes(ctx, "passive", alpha=0)
    else:
        L = Bayes(ctx, "active", alpha=1.0, eps_bar=EPS_BAR, lam=LAM)
    rng = np.random.default_rng([seed, k, 7])
    rngd = np.random.default_rng([seed, k, 8])
    recs = run_match(ctx, L, opp, G_GAMES, rng, rng_deal=rngd)
    B = np.array([r["b"] for r in recs])
    beta = np.array([t[0] for t in ctx.thetas]); gam = np.array([t[1] for t in ctx.thetas])
    return dict(name=name, mode=mode, seed=seed,
                b_true=B[:, k], b_eff=B[:, E].sum(1), H=entropy(B.T, 0),
                err_beta=np.abs(B @ beta - th[0]), err_gamma=np.abs(B @ gam - th[1]),
                reveals=np.cumsum([r["reveals"] for r in recs]),
                map_correct=(B.argmax(1) == k).astype(float))


def main():
    ctx = context(SMALL)
    loss = loss_matrix(ctx)
    jobs = [(n, m, s, loss) for n in ARCHETYPES for m in ("passive", "active") for s in range(SEEDS)]
    with Pool(2, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=4)
    save_pickle(dict(results=res, loss=loss, thetas=ctx.thetas, eta=ETA), "e2.pkl")
    print("E2 done", len(res))


if __name__ == "__main__":
    main()
