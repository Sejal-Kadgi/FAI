"""
bluff.py -- reduced two-player Bluff: game tree, CFR, exact best response,
exploitability, opponent types, Bayesian posterior, realization-plan mixing.

Design notes
* Cards matter only by rank, so a hand is a count vector of length K.
* Some cards are left UNDEALT (2*h < K*m), so a player never knows the
  opponent's hand exactly (TA feedback #1).
* The whole game tree is built once (explicit nodes), so CFR, best response,
  exploitability and the Bayesian posterior are all exact tree computations.
* Payoffs are stored for player 0. Player 1 receives the negative.
"""
import itertools
from math import comb
import numpy as np

ACCEPT, CALL = 0, 1


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))


class Node:
    __slots__ = ("id", "kind", "player", "depth", "t", "hand", "r", "claim",
                 "acts", "info", "children", "probs", "payoff")

    def __init__(self, nid, kind, player, depth):
        self.id, self.kind, self.player, self.depth = nid, kind, player, depth
        self.t = self.hand = self.r = self.claim = self.acts = self.info = None
        self.children, self.probs, self.payoff = [], None, 0.0


class Game:
    """Bluff(K, m, h, T): K ranks, m copies each, h cards dealt to each player,
    at most T plays. Requires 2h < K*m so that some cards stay undealt."""

    def __init__(self, K=3, m=2, h=2, T=4):
        assert 2 * h < K * m, "leave some cards undealt (TA feedback #1)"
        self.K, self.m, self.h, self.T = K, m, h, T
        self.nodes, self.info_key, self.info_actions = [], {}, []
        self.info_player, self.info_nodes = [], []
        self._build()
        self.by_depth = {}
        for n in self.nodes:
            self.by_depth.setdefault(n.depth, []).append(n)
        self.max_depth = max(self.by_depth)

    # ---------------------------------------------------------------- build
    def _new(self, kind, player, depth):
        n = Node(len(self.nodes), kind, player, depth)
        self.nodes.append(n)
        return n

    def _reg(self, node, priv, nacts):
        key = (node.player, priv)
        if key not in self.info_key:
            self.info_key[key] = len(self.info_actions)
            self.info_actions.append(nacts)
            self.info_player.append(node.player)
            self.info_nodes.append([])
        node.info = self.info_key[key]
        self.info_nodes[node.info].append(node.id)

    def _term(self, payoff0, depth):
        n = self._new("term", -1, depth)
        n.payoff = payoff0
        return n

    def _timeout(self, hands, depth):
        d = sum(hands[1]) - sum(hands[0])          # fewer cards is better
        return self._term(float((d > 0) - (d < 0)), depth)

    def _hands(self):
        K, m, h = self.K, self.m, self.h
        return [v for v in itertools.product(range(m + 1), repeat=K) if sum(v) == h]

    def _build(self):
        K, m, h = self.K, self.m, self.h
        root = self._new("chance", -1, 0)
        N = K * m
        total = comb(N, h) * comb(N - h, h)
        for a in self._hands():
            for b in self._hands():
                if any(x + y > m for x, y in zip(a, b)):
                    continue
                ways = 1
                for x, y in zip(a, b):
                    ways *= comb(m, x) * comb(m - x, y)
                priv = ((("hand", a),), (("hand", b),))
                child = self._play((a, b), (0,) * K, 0, priv, 1)
                root.children.append(child.id)
        root.probs = []
        for a in self._hands():
            for b in self._hands():
                if any(x + y > m for x, y in zip(a, b)):
                    continue
                ways = 1
                for x, y in zip(a, b):
                    ways *= comb(m, x) * comb(m - x, y)
                root.probs.append(ways / total)
        self.root = root

    def _subsets(self, hand):
        cmax = min(self.m, sum(hand))
        out = []
        for s in itertools.product(*[range(x + 1) for x in hand]):
            if 1 <= sum(s) <= cmax:
                out.append(s)
        return out

    def _play(self, hands, pile, t, priv, depth):
        p, r = t % 2, t % self.K
        node = self._new("play", p, depth)
        node.t, node.hand, node.r = t, hands[p], r
        node.acts = self._subsets(hands[p])
        self._reg(node, priv[p], len(node.acts))
        for S in node.acts:
            nh = list(hands); nh[p] = _sub(hands[p], S)
            npriv = list(priv)
            npriv[p] = priv[p] + (("play", S),)
            npriv[1 - p] = priv[1 - p] + (("claim", sum(S)),)
            node.children.append(self._resp(tuple(nh), pile, S, t, tuple(npriv), depth + 1).id)
        return node

    def _resp(self, hands, pile, S, t, priv, depth):
        p = t % 2; q = 1 - p; r = t % self.K; c = sum(S)
        node = self._new("resp", q, depth)
        node.t, node.hand, node.r, node.claim = t, hands[q], r, c
        self._reg(node, priv[q], 2)
        pile2 = _add(pile, S)
        # ACCEPT
        pa = (priv[0] + (("acc",),), priv[1] + (("acc",),))
        if sum(hands[p]) == 0:
            ch = self._term(1.0 if p == 0 else -1.0, depth + 1)
        elif t + 1 == self.T:
            ch = self._timeout(hands, depth + 1)
        else:
            ch = self._play(hands, pile2, t + 1, pa, depth + 1)
        node.children.append(ch.id)
        # CALL (showdown): S is revealed
        truthful = S[r] == c
        taker = q if truthful else p
        nh = list(hands); nh[taker] = _add(hands[taker], pile2)
        pc = [priv[0] + (("call", S),), priv[1] + (("call", S),)]
        pc[taker] = pc[taker] + (("take", pile2),)
        pc[1 - taker] = pc[1 - taker] + (("took", sum(pile2)),)
        if truthful and sum(hands[p]) == 0:
            ch = self._term(1.0 if p == 0 else -1.0, depth + 1)
        elif t + 1 == self.T:
            ch = self._timeout(tuple(nh), depth + 1)
        else:
            ch = self._play(tuple(nh), (0,) * self.K, t + 1, tuple(pc), depth + 1)
        node.children.append(ch.id)
        return node

    # ------------------------------------------------------------ utilities
    @property
    def n_info(self): return len(self.info_actions)

    def sigma_from_info(self, strat):
        """strat[info] -> per-node list of action-probability arrays."""
        return [strat[n.info] if n.kind in ("play", "resp") else None for n in self.nodes]

    def nprobs(self, n, sig):
        if n.kind == "chance": return n.probs
        return sig[n.id]

    def expected_value(self, sig):
        """Expected payoff to player 0 when every decision node follows sig."""
        V = [0.0] * len(self.nodes)
        for d in range(self.max_depth, -1, -1):
            for n in self.by_depth[d]:
                if n.kind == "term": V[n.id] = n.payoff
                else:
                    pr = self.nprobs(n, sig)
                    V[n.id] = sum(p * V[c] for p, c in zip(pr, n.children))
        return V[self.root.id]

    def best_response_value(self, i, sig_opp):
        """Exact payoff to player i when i best-responds to sig_opp (nodes of
        player 1-i). Aggregates counterfactual values per information set."""
        nn = len(self.nodes)
        q = [0.0] * nn; q[self.root.id] = 1.0
        for n in self.nodes:                       # parents precede children
            if n.kind in ("term",): continue
            if n.kind == "chance": pr = n.probs
            elif n.player == i: pr = [1.0] * len(n.children)
            else: pr = sig_opp[n.id]
            for p, c in zip(pr, n.children):
                q[c] = q[n.id] * p
        V = [0.0] * nn
        for d in range(self.max_depth, -1, -1):
            seen = set()
            for n in self.by_depth[d]:
                if n.kind in ("play", "resp") and n.player == i and n.info not in seen:
                    seen.add(n.info)
                    nodes = [self.nodes[x] for x in self.info_nodes[n.info]]
                    A = np.zeros(self.info_actions[n.info])
                    for m_ in nodes:
                        for a, c in enumerate(m_.children):
                            A[a] += q[m_.id] * V[c]
                    best = int(np.argmax(A))
                    for m_ in nodes:
                        V[m_.id] = V[m_.children[best]]
            for n in self.by_depth[d]:
                if n.kind == "term":
                    V[n.id] = n.payoff if i == 0 else -n.payoff
                elif n.kind == "chance" or n.player != i:
                    pr = self.nprobs(n, sig_opp)
                    V[n.id] = sum(p * V[c] for p, c in zip(pr, n.children))
        return V[self.root.id]

    def exploitability(self, sig):
        """eps(pi) = 1/2 (BR_0(pi_1) + BR_1(pi_0)); equals 0 at equilibrium."""
        return 0.5 * (self.best_response_value(0, sig) + self.best_response_value(1, sig))

    # ------------------------------------------------------------------ CFR
    def cfr(self, iters, log_every=0, log=None):
        regret = [np.zeros(k) for k in self.info_actions]
        ssum = [np.zeros(k) for k in self.info_actions]
        nodes = self.nodes

        def rm(r):
            p = np.maximum(r, 0.0); s = p.sum()
            return p / s if s > 0 else np.full(len(r), 1.0 / len(r))

        def walk(nid, r0, r1, rc, strat):
            n = nodes[nid]
            if n.kind == "term": return n.payoff
            if n.kind == "chance":
                return sum(p * walk(c, r0, r1, rc * p, strat) for p, c in zip(n.probs, n.children))
            s = strat[n.info]
            vals = np.empty(len(s))
            for a, c in enumerate(n.children):
                if n.player == 0: vals[a] = walk(c, r0 * s[a], r1, rc, strat)
                else: vals[a] = walk(c, r0, r1 * s[a], rc, strat)
            v = float(s @ vals)
            if n.player == 0:
                regret[n.info] += r1 * rc * (vals - v)
                ssum[n.info] += r0 * s
            else:
                regret[n.info] += r0 * rc * (-(vals - v))
                ssum[n.info] += r1 * s
            return v

        history = []
        for it in range(1, iters + 1):
            strat = [rm(r) for r in regret]
            walk(self.root.id, 1.0, 1.0, 1.0, strat)
            if log_every and (it % log_every == 0 or it == iters):
                avg = self._avg(ssum)
                eps = self.exploitability(self.sigma_from_info(avg))
                history.append((it, eps))
                if log: log(it, eps)
        return self._avg(ssum), history

    def _avg(self, ssum):
        out = []
        for s in ssum:
            t = s.sum()
            out.append(s / t if t > 0 else np.full(len(s), 1.0 / len(s)))
        return out

    # ------------------------------------------------------- opponent types
    def type_sigma(self, beta, gamma):
        """Node-level policy of opponent type theta=(beta,gamma) (Definition 6)."""
        sig = [None] * len(self.nodes)
        for n in self.nodes:
            if n.kind == "play":
                truth = [a for a, S in enumerate(n.acts) if S[n.r] == sum(S)]
                lie = [a for a, S in enumerate(n.acts) if S[n.r] != sum(S)]
                p = np.zeros(len(n.acts))
                if not truth: p[lie] = 1.0 / len(lie)
                elif not lie: p[truth] = 1.0 / len(truth)
                else:
                    p[lie] = beta / len(lie); p[truth] = (1 - beta) / len(truth)
                sig[n.id] = p
            elif n.kind == "resp":
                if n.hand[n.r] > self.m - n.claim: pc = 1.0       # provably false
                else: pc = gamma ** (1.0 / n.claim)
                sig[n.id] = np.array([1 - pc, pc])
        return sig

    def combine(self, sig_a, sig_b, a_player):
        """Per-node policy: player a_player follows sig_a, the other sig_b."""
        return [None if n.kind not in ("play", "resp") else
                (sig_a[n.id] if n.player == a_player else sig_b[n.id]) for n in self.nodes]

    # ----------------------------------------------------- Bayesian filter
    def opp_reach_table(self, agent, type_sigs):
        """R[k][node] = chance prob * opponent-type-k action probs on the path
        to the node (agent's own action probabilities are omitted: they cancel
        in the posterior)."""
        R = np.zeros((len(type_sigs), len(self.nodes)))
        for k, sig in enumerate(type_sigs):
            R[k, self.root.id] = 1.0
            for n in self.nodes:
                if n.kind == "term": continue
                if n.kind == "chance": pr = n.probs
                elif n.player == agent: pr = [1.0] * len(n.children)
                else: pr = sig[n.id]
                for p, c in zip(pr, n.children):
                    R[k, c] = R[k, n.id] * p
        return R

    def posterior(self, prior, R, info):
        """Exact b(theta | agent infoset): hidden cards are marginalised."""
        like = np.array([R[k, self.info_nodes[info]].sum() for k in range(R.shape[0])])
        post = prior * like
        s = post.sum()
        return post / s if s > 0 else prior.copy()

    # ----------------------------------------- realization-plan mixing (eq 5)
    def realization(self, strat, i):
        """x_i(I): probability that i's own choices lead to infoset I."""
        x = [0.0] * self.n_info
        reach = [0.0] * len(self.nodes); reach[self.root.id] = 1.0
        for n in self.nodes:
            if n.kind == "term": continue
            if n.kind == "chance": pr = [1.0] * len(n.children)
            elif n.player == i:
                pr = strat[n.info]
                x[n.info] = reach[n.id]
            else: pr = [1.0] * len(n.children)
            for p, c in zip(pr, n.children): reach[c] = reach[n.id] * p
        return x

    def mix(self, strat_a, strat_b, alpha):
        """Behavioural strategy of x = (1-alpha) x_a + alpha x_b (both players)."""
        out = [None] * self.n_info
        xa = [self.realization(strat_a, i) for i in (0, 1)]
        xb = [self.realization(strat_b, i) for i in (0, 1)]
        for I in range(self.n_info):
            i = self.info_player[I]
            wa, wb = (1 - alpha) * xa[i][I], alpha * xb[i][I]
            out[I] = (wa * strat_a[I] + wb * strat_b[I]) / (wa + wb) if wa + wb > 0 else strat_a[I]
        return out

    def pure_br_strategy(self, i, sig_opp):
        """Behavioural (pure) best-response strategy of player i to sig_opp."""
        nn = len(self.nodes)
        q = [0.0] * nn; q[self.root.id] = 1.0
        for n in self.nodes:
            if n.kind == "term": continue
            pr = n.probs if n.kind == "chance" else ([1.0] * len(n.children) if n.player == i else sig_opp[n.id])
            for p, c in zip(pr, n.children): q[c] = q[n.id] * p
        V = [0.0] * nn
        strat = [None] * self.n_info
        for I in range(self.n_info):
            strat[I] = np.full(self.info_actions[I], 1.0 / self.info_actions[I])  # unreached/other seat: uniform
        for d in range(self.max_depth, -1, -1):
            seen = set()
            for n in self.by_depth[d]:
                if n.kind in ("play", "resp") and n.player == i and n.info not in seen:
                    seen.add(n.info)
                    nodes = [self.nodes[x] for x in self.info_nodes[n.info]]
                    A = np.zeros(self.info_actions[n.info])
                    for m_ in nodes:
                        for a, c in enumerate(m_.children): A[a] += q[m_.id] * V[c]
                    best = int(np.argmax(A))
                    p = np.zeros(len(A)); p[best] = 1.0; strat[n.info] = p
                    for m_ in nodes: V[m_.id] = V[m_.children[best]]
            for n in self.by_depth[d]:
                if n.kind == "term": V[n.id] = n.payoff if i == 0 else -n.payoff
                elif n.kind == "chance" or n.player != i:
                    pr = self.nprobs(n, sig_opp)
                    V[n.id] = sum(p * V[c] for p, c in zip(pr, n.children))
        return strat
