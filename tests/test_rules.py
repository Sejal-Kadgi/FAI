"""Game rules, card conservation and information visibility."""
import itertools
import numpy as np
import pytest

from bluff.game import Game, CHANCE, PLAY, RESP, TERM, ACCEPT, CALL
from conftest import TINY

TAGS = {"hand", "play", "claim", "acc", "call", "take", "took"}


# ------------------------------------------------------------------ dealing
def test_undealt_cards_required():
    with pytest.raises(AssertionError):
        Game(3, 2, 3, 4)                          # full deal is refused by default
    Game(3, 2, 3, 2, allow_full_deal=True)        # allowed only explicitly (E1 comparison)


def test_chance_probabilities(g):
    kids = g.kids(0)
    assert np.isclose(g.cprob[kids].sum(), 1.0)
    assert (g.cprob[kids] > 0).all()


# ------------------------------------------------------- card conservation
@pytest.mark.parametrize("cfg", [TINY, (3, 2, 2, 6)])
def test_card_conservation(cfg):
    """hand_0 + hand_1 + pile + undealt = K*m at every node."""
    gg = Game(*cfg)
    K, m, h, _ = cfg
    nr = np.arange(gg.N) > 0
    total = gg.handsz[0][nr] + gg.handsz[1][nr] + gg.pilesz[nr] + (K * m - 2 * h)
    assert np.all(total == K * m)
    assert gg.handsz[0][nr].min() >= 0 and gg.handsz[1][nr].min() >= 0 and gg.pilesz[nr].min() >= 0


# --------------------------------------------------------------- transitions
def test_turn_order_rank_and_claim_sizes(g):
    for n in np.nonzero(g.kind == PLAY)[0]:
        t = g.t[n]
        assert g.player[n] == t % 2 and g.r[n] == t % g.K
        hand = g.handsz[g.player[n]][n]
        for c in g.kids(n):                      # every play leads to a response by the other player
            assert g.kind[c] == RESP and g.player[c] == 1 - g.player[n]
            assert 1 <= g.claim[c] <= min(g.m, hand)
            assert g.handsz[g.player[n]][c] == hand - g.claim[c]


def test_accept_and_call_transitions(g):
    for n in np.nonzero(g.kind == RESP)[0]:
        q = g.player[n]; p = 1 - q
        acc, call = g.kids(n)
        # ACCEPT: hands unchanged, pile keeps the cards
        assert g.handsz[0][acc] == g.handsz[0][n] and g.handsz[1][acc] == g.handsz[1][n]
        assert g.pilesz[acc] == g.pilesz[n]
        # CALL: the liar (or the caller if the claim was true) takes the whole pile
        taker = q if g.truth[n] else p
        assert g.handsz[taker][call] == g.handsz[taker][n] + g.pilesz[n]
        assert g.handsz[1 - taker][call] == g.handsz[1 - taker][n]
        assert g.pilesz[call] == 0


def test_terminal_payoffs(g):
    """Emptying your hand wins; otherwise fewer cards after T plays wins.
    Both reduce to payoff_0 = sign(|H_1| - |H_0|) at every terminal."""
    z = g.terms
    assert np.array_equal(g.payoff[z], np.sign(g.handsz[1][z] - g.handsz[0][z]).astype(float))
    # games end exactly when a hand is empty after an accepted/true play, or at the horizon
    for n in z:
        par = g.parent[n]
        empty = min(g.handsz[0][n], g.handsz[1][n]) == 0
        assert empty or g.t[par] + 1 == g.T


def _deal_index(K, m, h, a, b):
    hands = [v for v in itertools.product(range(m + 1), repeat=K) if sum(v) == h]
    idx = 0
    for x in hands:
        for y in hands:
            if any(u + w > m for u, w in zip(x, y)):
                continue
            if x == a and y == b:
                return idx
            idx += 1
    raise ValueError


def test_hand_checked_game(g_obs):
    """Deal P0=(1,1,0), P1=(0,1,1); worked through by hand."""
    g = g_obs
    n = g.kids(0)[_deal_index(3, 2, 2, (1, 1, 0), (0, 1, 1))]
    # t=0: P0 must claim rank 0; legal plays (product order): (0,1,0), (1,0,0), (1,1,0)
    assert g.kind[n] == PLAY and g.player[n] == 0 and len(g.kids(n)) == 3
    n = g.kids(n)[1]                               # play (1,0,0): a true 1-card claim
    assert g.claim[n] == 1 and g.truth[n] and not g.provfalse[n]
    n = g.kids(n)[CALL]                            # P1 calls a true claim: P1 takes the pile
    assert (g.handsz[0][n], g.handsz[1][n], g.pilesz[n]) == (1, 3, 0)
    assert g.obs_key[0][g.oc[0][n]][-2:] == (("call", (1, 0, 0)), ("took", 1))
    assert g.obs_key[1][g.oc[1][n]][-2:] == (("call", (1, 0, 0)), ("take", (1, 0, 0)))
    assert g.rev[1][n] == 1                        # P1's call revealed P0's cards
    # t=1: P1 holds (1,1,1) and must claim rank 1; play (0,1,0) (index 1), P0 accepts
    n = g.kids(n)[1]
    assert g.truth[n] and g.obs_key[0][g.oc[0][n]][-1] == ("claim", 1)
    n = g.kids(n)[ACCEPT]
    # t=2: P0 holds (0,1,0) and must claim rank 2: the only play is a lie
    assert g.player[n] == 0 and len(g.kids(n)) == 1
    resp = g.kids(n)[0]
    assert not g.truth[resp] and not g.provfalse[resp]
    win = g.kids(resp)[ACCEPT]                     # accepted lie that empties the hand: P0 wins
    assert g.kind[win] == TERM and g.payoff[win] == 1.0
    caught = g.kids(resp)[CALL]                    # called: P0 picks up the 2-card pile
    assert g.kind[caught] == PLAY and (g.handsz[0][caught], g.handsz[1][caught]) == (2, 2)
    assert g.obs_key[0][g.oc[0][caught]][-1] == ("take", (0, 2, 0))   # sees P1's card too
    assert g.obs_key[1][g.oc[1][caught]][-1] == ("took", 2)
    assert g.rev[0][caught] == 1                   # pile held P1's card -> a reveal for P0


# ------------------------------------------------------ information visibility
def _events(key):
    return key[1:]                                 # key[0] is the node kind


def test_observations_are_rank_counts_and_known_tags(g_obs):
    g = g_obs
    for i in (0, 1):
        for key in g.obs_key[i]:
            for ev in _events(key):
                assert ev[0] in TAGS
                if ev[0] in ("hand", "play", "call", "take"):
                    v = ev[1]
                    assert len(v) == g.K and all(isinstance(x, int) and 0 <= x <= g.m for x in v)


def test_call_reveals_cards_and_pile_visibility(g_obs):
    g = g_obs
    for n in np.nonzero(g.kind == RESP)[0]:
        q = g.player[n]; p = 1 - q
        S = _events(g.obs_key[p][g.oc[p][n]])[-1]
        assert S[0] == "play"                       # the actor knows the cards...
        assert _events(g.obs_key[q][g.oc[q][n]])[-1] == ("claim", g.claim[n])   # ...the other only the size
        acc, call = g.kids(n)
        for i in (0, 1):                            # accept reveals nothing
            assert _events(g.obs_key[i][g.oc[i][acc]])[-1] == ("acc",)
        taker = q if g.truth[n] else p
        kt = _events(g.obs_key[taker][g.oc[taker][call]])
        ko = _events(g.obs_key[1 - taker][g.oc[1 - taker][call]])
        assert kt[-2] == ("call", S[1]) and ko[-2] == ("call", S[1])     # S shown to both
        assert kt[-1][0] == "take" and sum(kt[-1][1]) == g.pilesz[n]    # collector sees the pile
        assert ko[-1] == ("took", g.pilesz[n])                          # other: size only


def test_opponent_hand_and_undealt_never_revealed(g_obs):
    """Only the own starting hand is ever observed as a hand; terminal
    observations add nothing beyond the last public action."""
    g = g_obs
    for i in (0, 1):
        for key in g.obs_key[i]:
            ev = _events(key)
            if not ev:
                continue
            assert ev[0][0] == "hand" and all(e[0] != "hand" for e in ev[1:])
    # a terminal's observation equals its parent's plus only the final response events
    for z in g.terms:
        par = g.parent[z]
        for i in (0, 1):
            kz, kp = _events(g.obs_key[i][g.oc[i][z]]), _events(g.obs_key[i][g.oc[i][par]])
            assert kz[:len(kp)] == kp and {e[0] for e in kz[len(kp):]} <= {"acc", "call", "take", "took"}


def test_infoset_nodes_share_observations(g):
    """All nodes of an infoset carry the same acting-player observation class
    and the same number of actions (well-formed information sets)."""
    dec = np.nonzero(g.info >= 0)[0]
    for I in np.unique(g.info[dec])[:2000]:
        nodes = dec[g.info[dec] == I]
        pl = g.player[nodes[0]]
        assert len(set(g.oc[pl][nodes])) == 1
        assert len({len(g.kids(n)) for n in nodes}) == 1
