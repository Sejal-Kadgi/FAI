"""
opponents.py -- parametric opponent types (Definition 6) and the
off-model opponents used in the misspecification study (E5).

Every opponent is a flat behavioural policy over sequences of BOTH seats, so it
can sit in either seat. Policies are hand-conditioned but depend only on what
the acting player observes, hence are valid behavioural strategies.
"""
import numpy as np

GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
ARCHETYPES = {
    "Honest": (0.1, 0.1),
    "Aggressive bluffer": (0.9, 0.1),
    "Suspicious caller": (0.3, 0.9),
    "Random": (0.5, 0.5),
    "Paranoid liar": (0.9, 0.9),
}


def type_grid():
    return [(b, c) for b in GRID for c in GRID]


def type_policy(g, beta, gamma, beta_fn=None, call_fn=None):
    """sigma_theta as a flat vector.

    Playing: if both truthful and false plays exist, bluff w.p. beta (uniform
    within each category); otherwise uniform over what exists.
    Responding: call w.p. 1 if the claim is provably false from own hand,
    else w.p. gamma**(1/c).
    beta_fn(info) / call_fn(info) optionally override beta / call probability
    per infoset (used for off-model opponents).
    """
    sig = np.zeros(g.nseq)
    play = g.seq_kind == 1
    I = g.seq_info
    truth = g.seq_truth & play
    lie = (~g.seq_truth) & play
    nt = np.bincount(I, weights=truth.astype(float), minlength=g.n_info)
    nl = np.bincount(I, weights=lie.astype(float), minlength=g.n_info)
    b = np.full(g.n_info, beta, float) if beta_fn is None else beta_fn(g)
    ntI, nlI, bI = nt[I], nl[I], b[I]
    p_truth = np.where(nlI == 0, 1.0 / np.maximum(ntI, 1), (1 - bI) / np.maximum(ntI, 1))
    p_lie = np.where(ntI == 0, 1.0 / np.maximum(nlI, 1), bI / np.maximum(nlI, 1))
    sig[truth] = p_truth[truth]
    sig[lie] = p_lie[lie]
    resp = g.seq_kind == 2
    c = np.maximum(g.seq_csize, 1)
    pc = gamma ** (1.0 / c) if call_fn is None else call_fn(g)[I]
    pc = np.where(g.info_provfalse[I], 1.0, pc)
    sig[resp & (g.seq_act == 1)] = pc[resp & (g.seq_act == 1)]
    sig[resp & (g.seq_act == 0)] = 1 - pc[resp & (g.seq_act == 0)]
    return sig


def grid_policies(g):
    th = type_grid()
    return th, [type_policy(g, b, c) for b, c in th]


# ------------------------------------------------------- off-model opponents
def noisy_policy(g, base, eta):
    """(1-eta) * base + eta * uniform, per infoset."""
    return (1 - eta) * base + eta * g.uniform()


def size_dependent(g):
    """Bluffs when holding many cards (beta=0.9 if |H|>=2 else 0.1); calls
    2-card claims w.p. 0.9 and 1-card claims w.p. 0.1. Not representable by
    any (beta, gamma) on the grid."""
    bfn = lambda g: np.where(g.info_handsz >= 2, 0.9, 0.1)
    cfn = lambda g: np.where(g.seq_csize[g.info_off] >= 2, 0.9, 0.1)
    return type_policy(g, 0.5, 0.5, beta_fn=bfn, call_fn=cfn)


def max_dump(g, gamma=0.3):
    """Always plays the largest legal claim (truthful if possible among the
    largest), calls with gamma**(1/c). A greedy 'get rid of cards' player."""
    sig = type_policy(g, 0.5, gamma)
    play = g.seq_kind == 1
    I = g.seq_info
    cmax = np.zeros(g.n_info)
    np.maximum.at(cmax, I[play], g.seq_csize[play])
    big = play & (g.seq_csize == cmax[I])
    bt = big & g.seq_truth
    has_bt = np.bincount(I, weights=bt.astype(float), minlength=g.n_info) > 0
    sel = np.where(has_bt[I], bt, big)
    w = np.where(play, sel.astype(float), 0.0)
    pol = g.normalise(w)
    return np.where(play, pol, sig)
