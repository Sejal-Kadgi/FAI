"""Shared set-up for the experiments: game sizes, cached equilibria, contexts."""
import os, sys, json, pickle, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bluff.game import Game                      # noqa: E402
from bluff import solvers as S                   # noqa: E402
from bluff.agents import Context                 # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
CACHE = os.path.join(RES, "cache")
for d in (RES, FIG, CACHE):
    os.makedirs(d, exist_ok=True)

SMALL = (3, 2, 2, 6)      # main testbed for simulation experiments (E2, E4, E5)
LARGE = (4, 2, 3, 4)      # larger testbed for exact experiments (E1, E3)
TINY = (3, 2, 2, 4)       # sanity size
EQ_ITERS = {TINY: 2000, SMALL: 2000, LARGE: 600}
EPS_BAR = 0.05            # default safety level for the deployed learners


def tag(cfg):
    return "K{}m{}h{}T{}".format(*cfg)


def load_game(cfg, full_deal=False):
    return Game(*cfg, allow_full_deal=full_deal)


def equilibrium(g, cfg, iters=None, full_deal=False):
    f = os.path.join(CACHE, f"eq_{tag(cfg)}{'_full' if full_deal else ''}.npy")
    if os.path.exists(f):
        return np.load(f)
    t = time.time()
    eq, _ = S.cfr(g, iters or EQ_ITERS.get(cfg, 1000), plus=True)
    np.save(f, eq)
    print(f"[cache] equilibrium {tag(cfg)} in {time.time()-t:.1f}s, eps={S.exploitability(g, eq):.2e}")
    return eq


def context(cfg):
    g = load_game(cfg)
    eq = equilibrium(g, cfg)
    return Context(g, eq)


def save_json(obj, name):
    with open(os.path.join(RES, name), "w", newline="\n") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save_pickle(obj, name):
    with open(os.path.join(RES, name), "wb") as f:
        pickle.dump(obj, f)


def load_pickle(name):
    with open(os.path.join(RES, name), "rb") as f:
        return pickle.load(f)
