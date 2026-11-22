"""
solvers.py -- vectorised exact computations on a flattened Bluff tree.

All strategies are flat vectors over sequences (see game.py). Every routine is
a forward (root -> leaves) or backward (leaves -> root) pass over tree levels.

* reach(...)            per-node reach probabilities
* expected_value(...)   value to player 0 of a full profile
* best_response(...)    exact best response to *opponent reach weights*; the
                        weights may be a single policy or a Bayesian mixture of
                        policies (type at a hidden chance root)
* exploitability(...)   seat-averaged exploitability (Definition 3)
* cfr(...)              CFR / CFR+ self-play
* rnr(...)              restricted Nash response (Johanson et al. 2007)
* realization / mix     realization-plan mixing, eq. (5)
"""
import numpy as np
from .game import CHANCE, PLAY, RESP, TERM


# --------------------------------------------------------------------- passes
def edge_prob(g, sigma):
    """Probability of the edge into every node under profile sigma (chance incl.)."""
    p = g.cprob.copy()
    e = g.dec_edges
    p[e] = sigma[g.esq[e]]
    return p


def reach(g, sigma, players=(0, 1), chance=True):
    """Product along the path of the edge probabilities of the listed players
    (and of chance if chance=True). Other players' edges count as 1."""
    f = np.ones(g.N)
    e = g.dec_edges
    for i in players:
        sel = e[g.edge_player[e] == i]
        f[sel] = sigma[g.esq[sel]]
    if chance:
        root_kids = g.kids(0)
        f[root_kids] = g.cprob[root_kids]
    r = np.ones(g.N)
    for d in range(1, g.max_depth + 1):
        c = g.level[d]
        r[c] = r[g.parent[c]] * f[c]
    return r


def opp_reach(g, sigma_opp, agent):
    """chance * opponent action probabilities (agent's own edges count as 1)."""
    return reach(g, sigma_opp, players=(1 - agent,), chance=True)


def node_values(g, sigma):
    """EV[n] = expected payoff to player 0 conditional on reaching node n."""
    ep = edge_prob(g, sigma)
    V = np.zeros(g.N)
    V[g.terms] = g.payoff[g.terms]
    for d in range(g.max_depth, 0, -1):
        c = g.level[d]
        V += np.bincount(g.parent[c], weights=ep[c] * V[c], minlength=g.N) * (g.kind != TERM)
    return V


def expected_value(g, sigma):
    return float(node_values(g, sigma)[0])


def best_response(g, i, q, bonus_node=None, return_value=True):
    """Exact best response of player i.

    q[n]  : opponent-and-chance reach weight of node n (any non-negative
            weights; for a Bayesian mixture use sum_k b_k * R_k).
    bonus_node : optional extra counterfactual value added on the edge into a
            node (used for information-seeking calls).
    Returns (pure behavioural strategy for player i as flat vector over all
    sequences -- other player's entries are uniform, value of i at the root
    = sum_z q(z) u_i(z) under the best response).
    """
    ui = g.payoff if i == 0 else -g.payoff
    CV = np.zeros(g.N)
    CV[g.terms] = q[g.terms] * ui[g.terms]
    if bonus_node is not None:
        CV += bonus_node
    strat = g.uniform().copy()
    chosen = np.zeros(g.nseq)
    for d in range(g.max_depth - 1, -1, -1):
        c = g.level[d + 1]
        par = g.parent[c]
        own = g.edge_player[c] == i
        # opponent / chance parents: sum of children
        co = c[~own]
        add = np.zeros(g.N)
        add += np.bincount(g.parent[co], weights=CV[co], minlength=g.N)
        # own parents: choose best action per infoset
        infos = g.info_by.get((d, i))
        if infos is not None and len(infos):
            cs = c[own]
            ssum = np.bincount(g.esq[cs], weights=CV[cs], minlength=g.nseq)
            pad = g.info_pad[infos]
            vals = np.where(pad >= 0, ssum[np.maximum(pad, 0)], -np.inf)
            best = pad[np.arange(len(infos)), np.argmax(vals, axis=1)]
            seqs = pad[pad >= 0]
            strat[seqs] = 0.0
            strat[best] = 1.0
            chosen[best] = 1.0
            add += np.bincount(g.parent[cs], weights=CV[cs] * chosen[g.esq[cs]], minlength=g.N)
        nonterm = g.level[d][g.kind[g.level[d]] != TERM]
        CV[nonterm] += add[nonterm]
    return strat, float(CV[0])


def br_value(g, i, sigma_opp):
    """Payoff to i when best-responding to sigma_opp (opponent = 1-i)."""
    return best_response(g, i, opp_reach(g, sigma_opp, i))[1]


def vstar_bounds(g, eq):
    """Certified bounds lo <= v_0* <= hi from an approximate equilibrium:
    v_0* <= max_pi0 u_0(pi0, eq_1) = BR_0 value, v_0* >= -BR_1 value.
    The width hi - lo equals 2 * eps_avg(eq)."""
    return -br_value(g, 1, eq), br_value(g, 0, eq)


def seat_exploitability(g, sigma, vb):
    """Upper bounds on the per-seat exploitabilities (Definition 3):
        eps_0(pi_0) = v_0* - min u_0 = v_0* + BR_1-value(pi_0) <= hi + b1
        eps_1(pi_1) = v_1* - min u_1 = -v_0* + BR_0-value(pi_1) <= b0 - lo
    vb = vstar_bounds(g, eq). Being upper bounds, a cap enforced on them is
    a certified cap on the true per-seat exploitabilities."""
    lo, hi = vb
    b0 = br_value(g, 0, sigma)
    b1 = br_value(g, 1, sigma)
    return hi + b1, b0 - lo


def exploitability_max(g, sigma, vb):
    """Safety metric: eps_max(pi) = max(eps_0(pi_0), eps_1(pi_1))."""
    return max(seat_exploitability(g, sigma, vb))


def exploitability(g, sigma, per_seat=False):
    """Seat-AVERAGED exploitability eps(pi) = 1/2 (BR_0(pi_1) + BR_1(pi_0)).
    Used for reporting CFR convergence only; it is NOT a safety metric (an
    average does not bound each seat). Safety uses exploitability_max."""
    b0 = br_value(g, 0, sigma)
    b1 = br_value(g, 1, sigma)
    if per_seat:
        return 0.5 * (b0 + b1), b0, b1
    return 0.5 * (b0 + b1)


def value_vs(g, sigma_agent, sigma_opp, agent):
    """Exact expected payoff to `agent` (seat) when agent plays sigma_agent and
    the opponent plays sigma_opp."""
    v = expected_value(g, g.combine(sigma_agent, sigma_opp, agent))
    return v if agent == 0 else -v


# -------------------------------------------------------------- regret match
def regret_matching(g, R):
    return g.normalise(np.maximum(R, 0.0))


# ----------------------------------------------------------------------- CFR
def cfr(g, iters, plus=True, log_every=0, log=None):
    """Simultaneous-update CFR (plus=False) or CFR+ (plus=True, regret
    floor at 0 and linear averaging). Returns (average profile, history)."""
    R = np.zeros(g.nseq)
    S = np.zeros(g.nseq)
    e = g.dec_edges
    ep_e = g.edge_player[e]
    par = g.parent[e]
    sq = g.esq[e]
    hist = []
    for it in range(1, iters + 1):
        sigma = regret_matching(g, R)
        r0 = reach(g, sigma, (0,), chance=False)
        r1 = reach(g, sigma, (1,), chance=False)
        rc = reach(g, sigma, (), chance=True)
        V = node_values(g, sigma)
        adv = V[e] - V[par]
        w = np.where(ep_e == 0, r1[par] * rc[par] * adv, -r0[par] * rc[par] * adv)
        inst = np.bincount(sq, weights=w, minlength=g.nseq)
        own = np.where(ep_e == 0, r0[par], r1[par])
        contrib = np.bincount(sq, weights=own * sigma[sq], minlength=g.nseq)
        if plus:
            R = np.maximum(R + inst, 0.0)
            S += it * contrib
        else:
            R += inst
            S += contrib
        if log_every and (it % log_every == 0 or it == iters):
            avg = g.normalise(S)
            eps = exploitability(g, avg)
            hist.append((it, eps))
            if log:
                log(it, eps)
    return g.normalise(S), hist


def rnr(g, agent, model, p, iters=300, warm=None):
    """Restricted Nash response for seat `agent`.

    The opponent (seat 1-agent) is, with probability p, forced to play the
    fixed `model` policy and, with probability 1-p, free; the agent does not
    know which. Solved with CFR+. Returns the agent's average strategy
    (flat vector; only the agent's sequences are meaningful).
    p = 0 gives an equilibrium strategy, p = 1 a best response to `model`.
    """
    o = 1 - agent
    s_a = 1.0 if agent == 0 else -1.0          # sign of agent payoff relative to u0
    RA = np.zeros(g.nseq); SA = np.zeros(g.nseq)
    RF = np.zeros(g.nseq)
    e = g.dec_edges
    ep_e = g.edge_player[e]
    par = g.parent[e]
    sq = g.esq[e]
    rM = reach(g, model, (o,), chance=False)
    rc = reach(g, model, (), chance=True)
    isA = ep_e == agent
    isF = ep_e == o
    for it in range(1, iters + 1):
        sA = regret_matching(g, RA)
        sF = regret_matching(g, RF)
        prof_F = g.combine(sA, sF, agent)
        prof_M = g.combine(sA, model, agent)
        rA = reach(g, prof_F, (agent,), chance=False)
        rF = reach(g, prof_F, (o,), chance=False)
        VF = node_values(g, prof_F) * s_a
        VM = node_values(g, prof_M) * s_a
        # agent regrets: mixture over opponent modes
        wA = rc[par] * (p * rM[par] * (VM[e] - VM[par]) + (1 - p) * rF[par] * (VF[e] - VF[par]))
        instA = np.bincount(sq[isA], weights=wA[isA], minlength=g.nseq)
        # free-opponent regrets (its payoff is -agent payoff)
        wF = -rc[par] * rA[par] * (VF[e] - VF[par])
        instF = np.bincount(sq[isF], weights=wF[isF], minlength=g.nseq)
        RA = np.maximum(RA + instA, 0.0)
        RF = np.maximum(RF + instF, 0.0)
        SA += it * np.bincount(sq[isA], weights=rA[par[isA]] * sA[sq[isA]], minlength=g.nseq)
    return g.normalise(SA)


def rnr_profile(g, model, p, iters=300):
    """Agent profile for both seats from two RNR solves."""
    s0 = rnr(g, 0, model, p, iters)
    s1 = rnr(g, 1, model, p, iters)
    return g.combine(s0, s1, 0)


# ------------------------------------------------- realization-plan mixing
def realization_info(g, sigma):
    """x(I): probability that the owner's own choices lead to infoset I."""
    r0 = reach(g, sigma, (0,), chance=False)
    r1 = reach(g, sigma, (1,), chance=False)
    x = np.zeros(g.n_info)
    dec = np.nonzero(g.info >= 0)[0]
    own = np.where(g.player[dec] == 0, r0[dec], r1[dec])
    x[g.info[dec]] = own
    return x


def mix(g, sig_a, sig_b, alpha, xa=None, xb=None):
    """Behavioural strategy of x = (1-alpha) x_a + alpha x_b, both seats.
    alpha may be a scalar or a pair (alpha_seat0, alpha_seat1)."""
    if xa is None: xa = realization_info(g, sig_a)
    if xb is None: xb = realization_info(g, sig_b)
    a = np.asarray(alpha, float)
    aI = a[g.info_player] if a.ndim else np.full(g.n_info, float(a))
    wa = ((1 - aI) * xa)[g.seq_info]
    wb = (aI * xb)[g.seq_info]
    tot = wa + wb
    return np.where(tot > 0, (wa * sig_a + wb * sig_b) / np.where(tot > 0, tot, 1), sig_a)
