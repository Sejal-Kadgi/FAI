"""E1 -- equilibrium reference and the effect of undealt cards (feedback #1).

(a) CFR vs CFR+ exploitability vs iterations on three game sizes.
(b) Tree sizes, solve time, game value.
(c) Feedback #1: under each game's equilibrium, what fraction of claims can the
    responder classify as true/false from its own information alone, and how
    many opponent hands are consistent with its own starting hand?
"""
import time
import numpy as np
from common import *
from bluff.game import RESP
from itertools import product


def decidability(g, eq):
    """Reach-weighted share of claims whose truth value is identical in every
    node of the responder's infoset, i.e. decidable without calling."""
    r = S.reach(g, eq)
    resp = np.nonzero(g.kind == RESP)[0]
    I = g.info[resp]
    tr = g.truth[resp].astype(float)
    n = np.bincount(I, minlength=g.n_info)
    st = np.bincount(I, weights=tr, minlength=g.n_info)
    dec_I = (st == 0) | (st == n)
    w = r[resp]
    lie_w = (w * (1 - tr)).sum() / w.sum()
    # call rate at equilibrium
    call = eq[g.info_off[I] + 1]
    return dict(decidable=float((w * dec_I[I]).sum() / w.sum()),
                provably_false=float((w * g.provfalse[resp]).sum() / w.sum()),
                lie_rate=float(lie_w), call_rate=float((w * call).sum() / w.sum()),
                claims_per_game=float(w.sum()))


def opp_hand_uncertainty(K, m, h):
    hands = [v for v in product(range(m + 1), repeat=K) if sum(v) == h]
    cnt = []
    for a in hands:
        cnt.append(sum(1 for b in hands if all(x + y <= m for x, y in zip(a, b))))
    return float(np.mean(cnt))


def main():
    out = dict(curves={}, sizes=[], feedback1=[])
    for cfg, iters, every in [(TINY, 2000, 20), (SMALL, 2000, 20), (LARGE, 600, 20)]:
        g = load_game(cfg)
        for plus in (False, True):
            t = time.time()
            avg, hist = S.cfr(g, iters, plus=plus, log_every=every)
            dt = time.time() - t
            out["curves"][f"{tag(cfg)}_{'cfr+' if plus else 'cfr'}"] = hist
            print(tag(cfg), "CFR+" if plus else "CFR", f"final eps={hist[-1][1]:.2e} ({dt:.0f}s)", flush=True)
            if plus:
                eq = avg
                np.save(os.path.join(CACHE, f"eq_{tag(cfg)}.npy"), eq) if not os.path.exists(
                    os.path.join(CACHE, f"eq_{tag(cfg)}.npy")) else None
                d = g.describe()
                d.update(value=S.expected_value(g, eq), eps=hist[-1][1], seconds=dt, iters=iters)
                out["sizes"].append(d)
    # feedback #1: full deal vs undealt
    for cfg, full in [((3, 2, 3, 4), True), ((3, 2, 2, 4), False),
                      ((3, 2, 3, 6), True), ((3, 2, 2, 6), False), ((4, 2, 3, 4), False)]:
        g = load_game(cfg, full_deal=full)
        eq = equilibrium(g, cfg, iters=1000 if g.N < 3e5 else 600, full_deal=full)
        d = decidability(g, eq)
        d.update(cfg=list(cfg), full_deal=full, undealt=cfg[0] * cfg[1] - 2 * cfg[2],
                 opp_hands=opp_hand_uncertainty(*cfg[:3]), nodes=int(g.N),
                 value=S.expected_value(g, eq))
        out["feedback1"].append(d)
        print("feedback1", cfg, "full" if full else "undealt", {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}, flush=True)
    save_json(out, "e1.json")


if __name__ == "__main__":
    main()
