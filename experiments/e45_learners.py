"""E4 -- learner comparison (RQ3, H3, H4) and E5 -- misspecification (RQ4).

Deployed setting (feedback #2): the agent learns from every observation; the
scarce resource is the length of the encounter (G games), and the cost of
information is endogenous -- a call that fails costs the pile. Each learner
plays G games against each opponent for SEEDS seeds; deals are paired across
learners. Per game we log the exact expected payoff, the oracle and
equilibrium payoffs in the same seat, the exploitability of the profile the
learner committed to, alpha, and the number of reveals.
"""
import sys, os, time
import numpy as np
from multiprocessing import Pool
from common import *
from bluff.agents import Bayes, Bandit, Fixed, RNRBayes, run_match
from bluff.opponents import ARCHETYPES, type_policy, noisy_policy, size_dependent, max_dump

G_GAMES = 100
SEEDS = 30
LAM = 0.2
P_RNR = 0.5

E4_LEARNERS = ["Equilibrium", "Oracle", "Bayes-greedy", "Bayes-MAP", "Bayes-gated",
               "Bayes-safe", "Bayes-safe-seat", "Bayes-safe-gated", "Bayes-safe-active",
               "UCB1", "Exp3", "SW-greedy", "RNR-MAP"]
E5_LEARNERS = ["Equilibrium", "Oracle", "Bayes-greedy", "Bayes-gated", "Bayes-safe", "Bayes-safe-seat",
               "Bayes-safe-gated", "UCB1", "RNR-MAP"]

_W = {}


def rnr_library(ctx):
    f = os.path.join(CACHE, f"rnr_{tag(SMALL)}_p{P_RNR}.npy")
    if os.path.exists(f):
        return list(np.load(f))
    t = time.time()
    with Pool(2, initializer=init) as pool:
        lib = pool.map(_rnr_one, range(ctx.K))
    np.save(f, np.stack(lib))
    print(f"[cache] RNR library in {time.time()-t:.0f}s", flush=True)
    return lib


def _rnr_one(k):
    ctx = _W["ctx"]
    return S.rnr_profile(ctx.g, ctx.pols[k], P_RNR, 300)


def opponents(which, ctx):
    if which == "e4":
        return {n: type_policy(ctx.g, *th) for n, th in ARCHETYPES.items()}
    g = ctx.g
    return {
        "Off-grid (0.2,0.8)": type_policy(g, 0.2, 0.8),
        "Off-grid (0.6,0.2)": type_policy(g, 0.6, 0.2),
        "Off-grid (0.8,0.6)": type_policy(g, 0.8, 0.6),
        "Size-dependent": size_dependent(g),
        "Max-dump": max_dump(g),
        "Noisy equilibrium": noisy_policy(g, ctx.eq, 0.3),
        "Equilibrium": ctx.eq,
    }


def init(which=None):
    ctx = context(SMALL)
    _W["ctx"] = ctx
    if which is None:
        return
    arch = [ctx.thetas.index(th) for th in ARCHETYPES.values()]
    _W["arms"] = [ctx.eq] + [ctx.br_to(ctx.pols[k]) for k in arch]
    _W["rnr"] = list(np.load(os.path.join(CACHE, f"rnr_{tag(SMALL)}_p{P_RNR}.npy")))
    _W["opps"] = opponents(which, ctx)
    _W["oracle"] = {n: ctx.br_to(p) for n, p in _W["opps"].items()}


def make(name, opp_name):
    ctx = _W["ctx"]
    if name == "Equilibrium": return Fixed(name, ctx.eq)
    if name == "Oracle": return Fixed(name, _W["oracle"][opp_name])
    if name == "Bayes-greedy": return Bayes(ctx, name, alpha=1.0)
    if name == "Bayes-MAP": return Bayes(ctx, name, alpha=1.0, map_only=True)
    if name == "Bayes-gated": return Bayes(ctx, name, alpha="gated")
    if name == "Bayes-safe": return Bayes(ctx, name, alpha=1.0, eps_bar=EPS_BAR)
    if name == "Bayes-safe-seat": return Bayes(ctx, name, alpha=1.0, eps_bar=EPS_BAR, cap="chord-seat")
    if name == "Bayes-safe-gated": return Bayes(ctx, name, alpha="gated", eps_bar=EPS_BAR)
    if name == "Bayes-safe-active": return Bayes(ctx, name, alpha=1.0, eps_bar=EPS_BAR, lam=LAM)
    if name == "UCB1": return Bandit(name, _W["arms"], "ucb", G_GAMES)
    if name == "Exp3": return Bandit(name, _W["arms"], "exp3", G_GAMES)
    if name == "SW-greedy": return Bandit(name, _W["arms"], "sw", G_GAMES)
    if name == "RNR-MAP": return RNRBayes(ctx, name, _W["rnr"])
    raise ValueError(name)


def job(args):
    lname, oname, seed, oi = args
    ctx = _W["ctx"]
    opp = _W["opps"][oname]
    L = make(lname, oname)
    rng = np.random.default_rng([seed, oi, 101])
    rngd = np.random.default_rng([seed, oi, 202])      # paired deals
    fixed = tuple(_W["arms"]) + tuple(_W["rnr"]) + (_W["oracle"][oname], ctx.eq)
    recs = run_match(ctx, L, opp, G_GAMES, rng, oracle=_W["oracle"][oname], track_eps=True,
                     rng_deal=rngd, fixed_ids=fixed)
    keys = ["u", "u_oracle", "u_eq", "payoff", "reveals", "eps", "eps0", "eps1"]
    out = {k: np.array([r[k] for r in recs]) for k in keys}
    out["alpha"] = np.array([r.get("alpha", np.nan) for r in recs])
    out.update(learner=lname, opp=oname, seed=seed)
    return out


def main(which):
    ctx = context(SMALL)
    rnr_library(ctx)
    learners = E4_LEARNERS if which == "e4" else E5_LEARNERS
    onames = list(opponents(which, ctx))
    jobs = [(l, o, s, oi) for oi, o in enumerate(onames) for l in learners for s in range(SEEDS)]
    t = time.time()
    with Pool(2, initializer=init, initargs=(which,)) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=3)):
            res.append(r)
            if (i + 1) % 150 == 0:
                print(f"{which}: {i+1}/{len(jobs)} jobs, {time.time()-t:.0f}s", flush=True)
    # deterministic order (imap_unordered returns jobs in completion order)
    res.sort(key=lambda r: (onames.index(r["opp"]), learners.index(r["learner"]), r["seed"]))
    save_pickle(dict(results=res, learners=learners, opps=onames, G=G_GAMES, eps_bar=EPS_BAR,
                     eps_eq=ctx.eps_eq), f"{which}.pkl")
    print(which, "done", time.time() - t, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "e4")
