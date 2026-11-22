"""
agents.py -- Bayesian filter, safe-exploitation learners, model-free and
robust baselines, and the match simulator.

A match is G games against a fixed opponent policy; the agent alternates seats
(game g is played in seat g mod 2). Before each game the learner commits to a
full two-seat profile; its exact expected payoff and exploitability for that
game are computed on the tree (no sampling noise), while the realised game is
sampled and is the only thing the learner observes.
"""
import math
import numpy as np
from . import solvers as S
from .game import RESP, CALL
from .opponents import type_grid, type_policy, ARCHETYPES


def entropy(p, axis=0):
    p = np.clip(p, 1e-300, None)
    return -(p * np.log(p)).sum(axis=axis)


class Context:
    """Everything that is precomputed once per game size."""

    def __init__(self, g, eq, verbose=False):
        self.g = g
        self.eq = eq
        self.vb = S.vstar_bounds(g, eq)                 # certified bounds on v_0*
        self.eps_seat_eq = S.seat_exploitability(g, eq, self.vb)
        self.eps_eq = max(self.eps_seat_eq)             # eps_max of the reference
        self.xeq = S.realization_info(g, eq)
        self.thetas = type_grid()
        self.pols = [type_policy(g, b, c) for b, c in self.thetas]
        self.K = len(self.thetas)
        self.R = []            # R[i][k, n]: chance * type-k opponent reach, agent in seat i
        self.L = []            # L[i][k, cls]: likelihood of agent observation class
        self.eig = []          # structures for expected information gain
        for i in (0, 1):
            Ri = np.stack([S.opp_reach(g, p, i) for p in self.pols])
            self.R.append(Ri)
            Li = np.stack([np.bincount(g.oc[i], weights=Ri[k], minlength=g.n_cls[i]) for k in range(self.K)])
            self.L.append(Li)
            resp = np.nonzero((g.kind == RESP) & (g.player == i))[0]
            kids_call = np.array([g.kids(n)[CALL] for n in resp])
            Jcls = g.oc[i][resp]
            Ccls = g.oc[i][kids_call]
            pairs = np.unique(np.stack([Ccls, Jcls], 1), axis=0)
            Juniq, Jidx = np.unique(pairs[:, 1], return_inverse=True)
            self.eig.append(dict(resp=resp, kids_call=kids_call, Jcls=Jcls,
                                 cc=pairs[:, 0], cJ=Jidx, Juniq=Juniq))
        if verbose:
            print("context ready: eps(eq) =", self.eps_eq)

    # ------------------------------------------------------------- beliefs
    def update(self, logb, seat, cls):
        lb = logb + np.log(np.clip(self.L[seat][:, cls], 1e-300, None))
        return lb - lb.max()

    def eig_bonus(self, b, seat, q, lam):
        """bonus on every CALL edge of the agent: lam * EIG(J) * q(parent)."""
        E = self.eig[seat]
        L = self.L[seat]
        Lc = b[:, None] * L[:, E["cc"]]
        Pc = Lc.sum(0)
        Hc = entropy(Lc / np.maximum(Pc, 1e-300), 0)
        nJ = len(E["Juniq"])
        num = np.bincount(E["cJ"], weights=Pc * Hc, minlength=nJ)
        den = np.bincount(E["cJ"], weights=Pc, minlength=nJ)
        LJ = b[:, None] * L[:, E["Juniq"]]
        HJ = entropy(LJ / np.maximum(LJ.sum(0), 1e-300), 0)
        eig = np.maximum(HJ - num / np.maximum(den, 1e-300), 0.0)
        # map back to nodes
        pos = np.searchsorted(E["Juniq"], E["Jcls"])
        bonus = np.zeros(self.g.N)
        bonus[E["kids_call"]] = lam * eig[pos] * q[E["resp"]]
        return bonus

    def posterior_br(self, b, lam=0.0, map_only=False):
        g = self.g
        out = []
        for i in (0, 1):
            if map_only:
                q = self.R[i][int(np.argmax(b))]
            else:
                q = b @ self.R[i]
            bonus = self.eig_bonus(b, i, q, lam) if lam > 0 else None
            out.append(S.best_response(g, i, q, bonus)[0])
        return g.combine(out[0], out[1], 0)

    def br_to(self, pol):
        g = self.g
        return g.combine(S.best_response(g, 0, S.opp_reach(g, pol, 0))[0],
                         S.best_response(g, 1, S.opp_reach(g, pol, 1))[0], 0)


# ------------------------------------------------------------- safety cap
class InfeasibleSafetyLevel(ValueError):
    """eps_bar is below the per-seat exploitability of the equilibrium itself."""


def check_feasible(ctx, eps_bar):
    if eps_bar < ctx.eps_eq:
        raise InfeasibleSafetyLevel(
            f"eps_bar={eps_bar:.3g} < eps_max(x*)={ctx.eps_eq:.3g}: the constraint is infeasible "
            f"(per seat: {ctx.eps_seat_eq[0]:.3g}, {ctx.eps_seat_eq[1]:.3g}). Solve x* more accurately "
            f"or raise eps_bar.")


def alpha_cap(ctx, xbr, eps_bar, method="chord", tol=1 / 64):
    """Mixing weight(s) that keep EVERY seat within eps_bar.

    Safety metric: eps_max(x) = max_i eps_i(x_i), with eps_i certified upper
    bounds (solvers.seat_exploitability). Each eps_i(x_alpha) is convex in
    alpha, so eps_max is convex too (Lemma 1b, Prop. 1).

    chord      : common alpha = (eps_bar - eps_max(x*)) / (eps_max(x_BR) - eps_max(x*)),
                 feasible by convexity; 2 BR passes.
    bisect     : tight common alpha_max on exact eps_max (interval sublevel set).
    chord-seat : a separate chord weight per seat, alpha_i from eps_i only.
                 Seat i's exploitability depends only on seat i's strategy,
                 so the per-seat constraints decouple; alpha_i >= common alpha.
    Returns (alpha, n_evals); alpha is a float, or a pair for chord-seat.
    Raises InfeasibleSafetyLevel if eps_bar < eps_max(x*).
    """
    g = ctx.g
    check_feasible(ctx, eps_bar)
    e1s = S.seat_exploitability(g, xbr, ctx.vb)
    if method == "chord-seat":
        al = []
        for e0, e1 in zip(ctx.eps_seat_eq, e1s):
            al.append(1.0 if e1 <= eps_bar else (eps_bar - e0) / (e1 - e0))
        return (al[0], al[1]), 1
    e0, e1 = ctx.eps_eq, max(e1s)
    if e1 <= eps_bar:
        return 1.0, 1
    a_chord = (eps_bar - e0) / (e1 - e0)
    if method == "chord":
        return a_chord, 1
    lo, hi, n = a_chord, 1.0, 1
    xb = S.realization_info(g, xbr)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        n += 1
        if S.exploitability_max(g, S.mix(g, ctx.eq, xbr, mid, ctx.xeq, xb), ctx.vb) <= eps_bar:
            lo = mid
        else:
            hi = mid
    return lo, n


# ---------------------------------------------------------------- learners
class Learner:
    name = "base"
    uses_belief = False

    def reset(self, rng): self.rng = rng
    def policy(self, g_idx, seat): raise NotImplementedError
    def observe(self, seat, payoff, path, cut): pass
    def info(self): return {}


class Fixed(Learner):
    def __init__(self, name, prof): self.name, self.prof = name, prof
    def policy(self, g_idx, seat): return self.prof


class Bayes(Learner):
    """Bayesian learner over the type grid.

    alpha     : float -> fixed weight; 'gated' -> amax * (1 - H(b)/H(b0))
    eps_bar   : None or safety level; enforced exactly by alpha_cap
    lam       : weight of the information-seeking bonus on calls (0 = passive)
    map_only  : plug-in best response to the MAP type (ablation)
    budget    : number of reveals after which the belief is frozen. Used ONLY
                as a measurement device (E2/E3); deployed learners use inf.
    """
    uses_belief = True

    def __init__(self, ctx, name, alpha="gated", amax=1.0, eps_bar=None, cap="chord",
                 lam=0.0, map_only=False, budget=math.inf, prior=None):
        self.ctx, self.name, self.alpha, self.amax = ctx, name, alpha, amax
        self.eps_bar, self.cap, self.lam, self.map_only = eps_bar, cap, lam, map_only
        self.budget = budget
        self.prior = np.full(ctx.K, 1.0 / ctx.K) if prior is None else np.asarray(prior)
        if eps_bar is not None:
            check_feasible(ctx, eps_bar)        # fail loudly, never silently play alpha = 0

    def reset(self, rng):
        self.rng = rng
        self.logb = np.log(self.prior)
        self.H0 = entropy(self.prior)
        self.reveals = 0
        self.last = {}

    @property
    def b(self):
        p = np.exp(self.logb - self.logb.max())
        return p / p.sum()

    def policy(self, g_idx, seat):
        ctx, b = self.ctx, self.b
        if self.alpha == 0 or (self.alpha == "gated" and self.amax == 0):
            self.last = dict(alpha=0.0, eps=ctx.eps_eq)
            return ctx.eq
        xbr = ctx.posterior_br(b, self.lam, self.map_only)
        if self.alpha == "gated":
            a = self.amax * max(0.0, 1 - entropy(b) / self.H0)
        else:
            a = float(self.alpha)
        a_raw = a
        if self.eps_bar is not None:
            acap, _ = alpha_cap(ctx, xbr, self.eps_bar, self.cap)
            a = tuple(min(a, c) for c in acap) if isinstance(acap, tuple) else min(a, acap)
        self.last = dict(alpha=float(np.mean(a)), alpha_raw=a_raw)
        if np.all(np.asarray(a) >= 1.0):
            return xbr
        if np.all(np.asarray(a) <= 0.0):
            return ctx.eq
        return S.mix(ctx.g, ctx.eq, xbr, a, ctx.xeq)

    def observe(self, seat, payoff, path, cut):
        if self.reveals >= self.budget:
            return
        g = self.ctx.g
        node, nrev = path[-1], path_reveals(g, path, seat)
        if self.reveals + nrev >= self.budget:          # freeze at the B-th reveal
            node = reveal_cut(g, path, seat, self.budget - self.reveals)
        self.logb = self.ctx.update(self.logb, seat, g.oc[seat][node])
        self.reveals += nrev


class Bandit(Learner):
    """UCB1 / Exp3 / sliding-window epsilon-greedy over a portfolio of profiles."""

    def __init__(self, name, arms, kind="ucb", G=100, window=10, eps=0.1):
        self.name, self.arms, self.kind, self.G = name, arms, kind, G
        self.window, self.eps = window, eps

    def reset(self, rng):
        self.rng = rng
        n = len(self.arms)
        self.cnt = np.zeros(n); self.sum = np.zeros(n)
        self.w = np.zeros(n)       # Exp3 log-weights
        self.hist = []
        self.t = 0
        self.gam = min(1.0, math.sqrt(n * math.log(n) / ((math.e - 1) * self.G)))

    def _probs(self):
        n = len(self.arms)
        p = np.exp(self.w - self.w.max()); p /= p.sum()
        return (1 - self.gam) * p + self.gam / n

    def policy(self, g_idx, seat):
        n = len(self.arms)
        self.t += 1
        if self.kind == "ucb":
            if (self.cnt == 0).any():
                a = int(np.argmax(self.cnt == 0))
            else:
                a = int(np.argmax(self.sum / self.cnt + np.sqrt(2 * math.log(self.t) / self.cnt)))
        elif self.kind == "exp3":
            self.p = self._probs()
            a = int(self.rng.choice(n, p=self.p))
        else:   # sliding-window epsilon-greedy
            means = np.zeros(n)
            for k in range(n):
                r = [x for (j, x) in self.hist if j == k][-self.window:]
                means[k] = np.mean(r) if r else np.inf
            a = int(self.rng.integers(n)) if self.rng.random() < self.eps else int(np.argmax(means))
        self.a = a
        return self.arms[a]

    def observe(self, seat, payoff, path, cut):
        r = (payoff + 1) / 2
        a = self.a
        self.cnt[a] += 1; self.sum[a] += r
        self.hist.append((a, r))
        if self.kind == "exp3":
            n = len(self.arms)
            self.w[a] += self.gam * (r / self.p[a]) / n


class RNRBayes(Learner):
    """Restricted Nash response to the MAP type (precomputed per grid type)."""
    uses_belief = True

    def __init__(self, ctx, name, rnr_profiles):
        self.ctx, self.name, self.prof = ctx, name, rnr_profiles

    def reset(self, rng):
        self.rng = rng
        self.logb = np.zeros(self.ctx.K)

    @property
    def b(self):
        p = np.exp(self.logb - self.logb.max()); return p / p.sum()

    def policy(self, g_idx, seat):
        if g_idx == 0:
            return self.ctx.eq
        return self.prof[int(np.argmax(self.logb))]

    def observe(self, seat, payoff, path, cut):
        self.logb = self.ctx.update(self.logb, seat, self.ctx.g.oc[seat][path[-1]])


# -------------------------------------------------------------- simulation
def sample_game(g, prof, rng, rng_deal=None):
    """Sample a root-to-leaf path under the full profile `prof`. The deal is
    drawn from rng_deal (if given) so that compared learners share deals."""
    n = 0
    path = [0]
    while g.kind[n] != 3:
        ks = g.kids(n)
        if n == 0:
            p = g.cprob[ks]
            u = (rng_deal or rng).random()
        else:
            p = prof[g.esq[ks]]
            u = rng.random()
        p = p / p.sum()
        n = int(ks[min(np.searchsorted(np.cumsum(p), u * (1 - 1e-12)), len(ks) - 1)])
        path.append(n)
    return path


def path_reveals(g, path, seat):
    return int(g.rev[seat][path].sum())


def reveal_cut(g, path, seat, k):
    """Node at which the agent's k-th reveal on this path happens."""
    c = np.cumsum(g.rev[seat][path])
    return path[int(np.argmax(c >= k))]


_EPS_CACHE = {}


def _eps_cached(ctx, prof):
    k = id(prof)
    if k in _EPS_CACHE and _EPS_CACHE[k][0] is prof:
        return _EPS_CACHE[k][1]
    e = S.seat_exploitability(ctx.g, prof, ctx.vb)
    _EPS_CACHE[k] = (prof, e)
    return e


def run_match(ctx, learner, opp, G, rng, oracle=None, track_eps=False, rng_deal=None,
              fixed_ids=()):
    """Play G games. Returns per-game records (exact expected values)."""
    g = ctx.g
    learner.reset(rng)
    recs = []
    u_or, u_eq = {}, {}
    for k in range(G):
        seat = k % 2
        prof = learner.policy(k, seat)
        full = g.combine(prof, opp, seat)
        path = sample_game(g, full, rng, rng_deal)
        pay = g.payoff[path[-1]] * (1 if seat == 0 else -1)
        u = S.value_vs(g, prof, opp, seat)
        rec = dict(game=k, seat=seat, payoff=float(pay), u=u,
                   reveals=path_reveals(g, path, seat))
        if oracle is not None:
            if seat not in u_or:
                u_or[seat] = S.value_vs(g, oracle, opp, seat)
                u_eq[seat] = S.value_vs(g, ctx.eq, opp, seat)
            rec["u_oracle"], rec["u_eq"] = u_or[seat], u_eq[seat]
        if track_eps:
            e0, e1 = (_eps_cached(ctx, prof) if any(prof is f for f in fixed_ids)
                      else S.seat_exploitability(g, prof, ctx.vb))
            rec["eps0"], rec["eps1"], rec["eps"] = e0, e1, max(e0, e1)
        if getattr(learner, "last", None):
            rec.update({k_: v for k_, v in learner.last.items() if k_ in ("alpha", "alpha_raw")})
        learner.observe(seat, pay, path, None)
        if learner.uses_belief:
            rec["b"] = learner.b.copy()
        recs.append(rec)
    return recs
