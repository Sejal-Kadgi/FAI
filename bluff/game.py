"""
game.py -- reduced two-player Bluff(K, m, h, T) as an explicit, flattened game tree.

Rules (Definition 5 of the plan)
* K ranks, m copies each. Each player is dealt h cards; by default 2h < K*m so
  some cards stay UNDEALT and nobody can infer the opponent's hand exactly
  (TA feedback #1). `allow_full_deal=True` builds the old full-deal variant,
  used only to quantify feedback #1 in experiment E1.
* At play t the active player p = t mod 2 puts 1..min(m,|H_p|) cards face down
  and claims they all have rank r_t = t mod K.
* The responder ACCEPTs or CALLs. On a call the cards are shown; the liar (or
  the caller, if the claim was true) takes the whole pile.
* A player who empties their hand wins (+1) if the last play is accepted or
  called and found true. After T plays: sign of the hand-size difference.
* Payoffs are stored for player 0; player 1 gets the negative.

Representation
Every node is a row in a set of numpy arrays (parent, depth, kind, ...). Each
edge leaving a decision node is a *sequence* (= infoset, action) with a global
index, so a behavioural strategy for both seats is a single flat vector over
sequences. This makes CFR, best response, exploitability, Bayesian filtering
and realization-plan mixing vectorised passes over tree levels.

Information visibility (the Bayesian filter uses exactly these rules, because
its likelihoods are sums over the same observation classes built here):
* own hand: known; cards are tracked by RANK only (count vectors of length K);
* a play: the actor knows the cards S, the other player sees only the claim size;
* ACCEPT: both see the response; nothing is revealed;
* CALL: the played cards S are shown to BOTH players; the player who collects
  the pile sees its full contents (counts per rank), the other player learns
  only the number of cards collected;
* undealt cards are never revealed, and hands are not shown at the end of a
  game (terminal observations add nothing beyond the last public action).

Observation classes: oc[i][n] identifies what player i has observed when the
play reaches node n (its private history). Two nodes with the same class are
indistinguishable to i. Decision infosets are the classes of the acting player.
"""
import itertools
from math import comb
import numpy as np

CHANCE, PLAY, RESP, TERM = 0, 1, 2, 3
ACCEPT, CALL = 0, 1


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))


class Game:
    def __init__(self, K=3, m=2, h=2, T=6, allow_full_deal=False, keep_obs=False):
        if not allow_full_deal:
            assert 2 * h < K * m, "leave some cards undealt (TA feedback #1)"
        assert 2 * h <= K * m
        self.K, self.m, self.h, self.T = K, m, h, T
        self._keep_obs = keep_obs       # keep readable observation records (tests only)
        self._build()
        self._finalise()

    # ------------------------------------------------------------------ build
    def _build(self):
        K, m, h = self.K, self.m, self.h
        L = self._L = dict(kind=[], player=[], depth=[], parent=[], pa=[], payoff=[],
                           cprob=[], oc0=[], oc1=[], rev0=[], rev1=[],
                           t=[], r=[], claim=[], provfalse=[], truth=[], handsz0=[], handsz1=[], pilesz=[])
        self._ockey = [{}, {}]
        self._info_key = {}
        # sequence-level data
        self.info_player, self.info_kind, self.info_off, self.info_nacts = [], [], [], []
        self.info_depth, self.info_provfalse, self.info_r, self.info_t = [], [], [], []
        self.seq_truth, self.seq_csize, self.seq_act = [], [], []
        self.info_handsz = []
        self._dec_info = {}

        root = self._new(CHANCE, -1, 0, -1, -1, ((), ()))
        N = K * m
        total = comb(N, h) * comb(N - h, h)
        hands = [v for v in itertools.product(range(m + 1), repeat=K) if sum(v) == h]
        for a in hands:
            for b in hands:
                if any(x + y > m for x, y in zip(a, b)):
                    continue
                ways = 1
                for x, y in zip(a, b):
                    ways *= comb(m, x) * comb(m - x, y)
                priv = ((("hand", a),), (("hand", b),))
                c = self._play((a, b), (0,) * K, 0, priv, 1, root, len(self._children_of_root()),
                               contrib=(False, False))
                L["cprob"][c] = ways / total
        assert abs(sum(L["cprob"][c] for c in self._root_children) - 1) < 1e-9

    def _children_of_root(self):
        if not hasattr(self, "_root_children"):
            self._root_children = []
        return self._root_children

    def _cls(self, i, key):
        d = self._ockey[i]
        if key not in d:
            d[key] = len(d)
        return d[key]

    def _new(self, kind, player, depth, parent, pa, priv, hands=None):
        L = self._L
        nid = len(L["kind"])
        L["kind"].append(kind); L["player"].append(player); L["depth"].append(depth)
        L["parent"].append(parent); L["pa"].append(pa); L["payoff"].append(0.0)
        L["cprob"].append(1.0)
        L["oc0"].append(self._cls(0, (kind,) + priv[0]))
        L["oc1"].append(self._cls(1, (kind,) + priv[1]))
        L["rev0"].append(0); L["rev1"].append(0)
        L["t"].append(-1); L["r"].append(-1); L["claim"].append(0)
        L["provfalse"].append(False); L["truth"].append(False)
        L["handsz0"].append(sum(hands[0]) if hands else -1)
        L["handsz1"].append(sum(hands[1]) if hands else -1)
        L["pilesz"].append(-1)          # filled in by the caller (bookkeeping only)
        if parent == 0:
            self._children_of_root().append(nid)
        return nid

    def _reg(self, nid, player, kind, nacts, depth):
        L = self._L
        key = (player, L["oc0"][nid] if player == 0 else L["oc1"][nid])
        if key not in self._info_key:
            I = len(self.info_player)
            self._info_key[key] = I
            self.info_player.append(player); self.info_kind.append(kind)
            self.info_off.append(len(self.seq_truth)); self.info_nacts.append(nacts)
            self.info_depth.append(depth)
            self.info_provfalse.append(L["provfalse"][nid])
            self.info_r.append(L["r"][nid]); self.info_t.append(L["t"][nid])
            self.info_handsz.append(L["handsz0"][nid] if player == 0 else L["handsz1"][nid])
            for _ in range(nacts):
                self.seq_truth.append(False); self.seq_csize.append(0); self.seq_act.append(-1)
            fresh = True
        else:
            fresh = False
        I = self._info_key[key]
        self._dec_info[nid] = I
        return I, fresh

    def _term(self, payoff0, depth, parent, pa, priv, hands, pile):
        nid = self._new(TERM, -1, depth, parent, pa, priv, hands)
        self._L["payoff"][nid] = payoff0
        self._L["pilesz"][nid] = sum(pile)
        return nid

    def _timeout(self, hands, depth, parent, pa, priv, pile):
        d = sum(hands[1]) - sum(hands[0])          # fewer cards is better
        return self._term(float((d > 0) - (d < 0)), depth, parent, pa, priv, hands, pile)

    def _subsets(self, hand):
        cmax = min(self.m, sum(hand))
        return [s for s in itertools.product(*[range(x + 1) for x in hand]) if 1 <= sum(s) <= cmax]

    def _play(self, hands, pile, t, priv, depth, parent, pa, contrib):
        L = self._L
        p, r = t % 2, t % self.K
        nid = self._new(PLAY, p, depth, parent, pa, priv, hands)
        L["t"][nid], L["r"][nid] = t, r
        L["pilesz"][nid] = sum(pile)
        acts = self._subsets(hands[p])
        I, fresh = self._reg(nid, p, PLAY, len(acts), depth)
        if fresh:
            off = self.info_off[I]
            for a, S in enumerate(acts):
                self.seq_truth[off + a] = (S[r] == sum(S))
                self.seq_csize[off + a] = sum(S)
        for a, S in enumerate(acts):
            nh = list(hands); nh[p] = _sub(hands[p], S)
            npriv = list(priv)
            npriv[p] = priv[p] + (("play", S),)
            npriv[1 - p] = priv[1 - p] + (("claim", sum(S)),)
            self._resp(tuple(nh), pile, S, t, tuple(npriv), depth + 1, nid, a, contrib)
        return nid

    def _resp(self, hands, pile, S, t, priv, depth, parent, pa, contrib):
        L = self._L
        p = t % 2; q = 1 - p; r = t % self.K; c = sum(S)
        nid = self._new(RESP, q, depth, parent, pa, priv, hands)
        L["t"][nid], L["r"][nid], L["claim"][nid] = t, r, c
        L["pilesz"][nid] = sum(pile) + c                    # S is already face down
        pf = hands[q][r] > self.m - c                       # provably false for q
        L["provfalse"][nid] = pf
        truthful = S[r] == c
        L["truth"][nid] = truthful
        if pf:                                              # implicit reveal for q
            L["rev0" if q == 0 else "rev1"][nid] += 1
        I, fresh = self._reg(nid, q, RESP, 2, depth)
        if fresh:
            off = self.info_off[I]
            self.seq_act[off] = ACCEPT; self.seq_act[off + 1] = CALL
            self.seq_csize[off] = c; self.seq_csize[off + 1] = c
        pile2 = _add(pile, S)
        contrib2 = tuple(contrib[i] or i == p for i in (0, 1))
        # ---- ACCEPT
        pa_ = (priv[0] + (("acc",),), priv[1] + (("acc",),))
        if sum(hands[p]) == 0:
            self._term(1.0 if p == 0 else -1.0, depth + 1, nid, ACCEPT, pa_, hands, pile2)
        elif t + 1 == self.T:
            self._timeout(hands, depth + 1, nid, ACCEPT, pa_, pile2)
        else:
            self._play(hands, pile2, t + 1, pa_, depth + 1, nid, ACCEPT, contrib2)
        # ---- CALL (showdown): S is revealed to both
        taker = q if truthful else p
        nh = list(hands); nh[taker] = _add(hands[taker], pile2); nh = tuple(nh)
        pc = [priv[0] + (("call", S),), priv[1] + (("call", S),)]
        pc[taker] = pc[taker] + (("take", pile2),)
        pc[1 - taker] = pc[1 - taker] + (("took", sum(pile2)),)
        pc = tuple(pc)
        empty = (0,) * self.K                               # the pile was picked up
        if truthful and sum(hands[p]) == 0:
            ch = self._term(1.0 if p == 0 else -1.0, depth + 1, nid, CALL, pc, nh, empty)
        elif t + 1 == self.T:
            ch = self._timeout(nh, depth + 1, nid, CALL, pc, empty)
        else:
            ch = self._play(nh, (0,) * self.K, t + 1, pc, depth + 1, nid, CALL, (False, False))
        # reveal accounting on the CALL edge
        if not pf:                                          # q calls p's claim: reveal for q
            L["rev0" if q == 0 else "rev1"][ch] += 1
        if taker == p and contrib[q]:                       # p picks up a pile holding q's cards
            L["rev0" if p == 0 else "rev1"][ch] += 1
        return nid

    # --------------------------------------------------------------- finalise
    def _finalise(self):
        L = self._L
        A = lambda k, dt=None: np.asarray(L[k], dtype=dt)
        self.kind = A("kind", np.int8); self.player = A("player", np.int8)
        self.depth = A("depth", np.int16); self.parent = A("parent", np.int64)
        self.pa = A("pa", np.int64); self.payoff = A("payoff", float)
        self.cprob = A("cprob", float)
        self.oc = [A("oc0", np.int64), A("oc1", np.int64)]
        self.rev = [A("rev0", np.int8), A("rev1", np.int8)]
        self.t = A("t", np.int16); self.r = A("r", np.int16); self.claim = A("claim", np.int16)
        self.provfalse = A("provfalse", bool); self.truth = A("truth", bool)
        self.handsz = [A("handsz0", np.int16), A("handsz1", np.int16)]
        self.pilesz = A("pilesz", np.int16)     # cards face down in the pile (-1 at the root)
        del self._L
        N = self.N = len(self.kind)
        self.n_cls = [len(d) for d in self._ockey]
        if self._keep_obs:              # obs_key[i][c] = what player i has observed in class c
            self.obs_key = [[None] * n for n in self.n_cls]
            for i in (0, 1):
                for key, c in self._ockey[i].items():
                    self.obs_key[i][c] = key
        del self._ockey, self._info_key

        self.info_player = np.asarray(self.info_player, np.int8)
        self.info_kind = np.asarray(self.info_kind, np.int8)
        self.info_off = np.asarray(self.info_off, np.int64)
        self.info_nacts = np.asarray(self.info_nacts, np.int64)
        self.info_depth = np.asarray(self.info_depth, np.int64)
        self.info_provfalse = np.asarray(self.info_provfalse, bool)
        self.info_r = np.asarray(self.info_r, np.int64)
        self.info_t = np.asarray(self.info_t, np.int64)
        self.info_handsz = np.asarray(self.info_handsz, np.int64)
        self.n_info = len(self.info_player)
        self.seq_truth = np.asarray(self.seq_truth, bool)
        self.seq_csize = np.asarray(self.seq_csize, np.int64)
        self.seq_act = np.asarray(self.seq_act, np.int64)
        self.nseq = len(self.seq_truth)
        self.seq_info = np.repeat(np.arange(self.n_info), self.info_nacts)
        self.seq_player = self.info_player[self.seq_info]
        self.seq_kind = self.info_kind[self.seq_info]
        self.seq_a = np.arange(self.nseq) - self.info_off[self.seq_info]

        self.info = np.full(N, -1, np.int64)
        for nid, I in self._dec_info.items():
            self.info[nid] = I
        del self._dec_info

        # edges (every non-root node is the head of one edge)
        par = self.parent.copy(); par[0] = 0
        pk = self.kind[par]
        self.edge_player = np.where(np.arange(N) == 0, -1,
                                    np.where(pk == CHANCE, -1, self.player[par])).astype(np.int8)
        self.esq = np.full(N, -1, np.int64)
        dec = (np.arange(N) > 0) & (pk != CHANCE)
        self.esq[dec] = self.info_off[self.info[par[dec]]] + self.pa[dec]
        self.dec_edges = np.nonzero(dec)[0]

        self.max_depth = int(self.depth.max())
        self.level = [np.nonzero(self.depth == d)[0] for d in range(self.max_depth + 1)]
        self.terms = np.nonzero(self.kind == TERM)[0]
        # infosets by (depth, player) for best-response passes
        Amax = int(self.info_nacts.max())
        self.Amax = Amax
        pad = np.full((self.n_info, Amax), -1, np.int64)
        for a in range(Amax):
            ok = self.info_nacts > a
            pad[ok, a] = self.info_off[ok] + a
        self.info_pad = pad
        self.info_by = {(d, i): np.nonzero((self.info_depth == d) & (self.info_player == i))[0]
                        for d in range(self.max_depth + 1) for i in (0, 1)}
        # child lists for simulation (CSR)
        order = np.argsort(self.parent[1:], kind="stable") + 1
        self.child_ptr = np.zeros(N + 1, np.int64)
        np.add.at(self.child_ptr, self.parent[1:] + 1, 1)
        self.child_ptr = np.cumsum(self.child_ptr)
        self.children = order            # children of n: children[child_ptr[n]:child_ptr[n+1]] (ordered by pa)
        # within a parent, children were created in action order, so a stable sort keeps pa order

    # -------------------------------------------------------------- helpers
    def kids(self, n):
        return self.children[self.child_ptr[n]:self.child_ptr[n + 1]]

    def uniform(self):
        return 1.0 / self.info_nacts[self.seq_info]

    def normalise(self, x):
        """Normalise a non-negative per-sequence vector into a behavioural strategy."""
        s = np.bincount(self.seq_info, weights=x, minlength=self.n_info)[self.seq_info]
        u = self.uniform()
        return np.where(s > 0, x / np.where(s > 0, s, 1), u)

    def combine(self, sig_a, sig_b, a_player):
        """Flat profile: player a_player follows sig_a, the other player sig_b."""
        return np.where(self.seq_player == a_player, sig_a, sig_b)

    def describe(self):
        return dict(K=self.K, m=self.m, h=self.h, T=self.T, nodes=int(self.N),
                    infosets=int(self.n_info), sequences=int(self.nseq),
                    terminals=int(len(self.terms)), depth=self.max_depth)
