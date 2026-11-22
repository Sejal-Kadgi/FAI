"""Compute every result number quoted in the report TEXT from the saved
results and write them as LaTeX macros to results/numbers.tex (plus a
readable copy in results/numbers.json). Tables are written by plots.py.

Run via `python plots.py` (repository root). Nothing here is typed by hand:
if the results change, the report text changes with them.
"""
import json, math
import numpy as np
from common import *
import make_figures as P

N = {}          # macro name (letters only) -> LaTeX string


def pct(x): return f"{100 * x:.0f}"
def f3(x): return f"{x:.3f}"
def sci(x):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f}\\times10^{{{e}}}"
def pm(d):
    return f"{d.mean():.4f}\\pm{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.4f}"


# ----------------------------------------------------------------------- E1
e1 = load_json("e1.json")
sz = {f"K{s['K']}m{s['m']}h{s['h']}T{s['T']}": s for s in e1["sizes"]}
S_, L_ = sz["K3m2h2T6"], sz["K4m2h3T4"]
N["nodesL"] = f"{L_['nodes'] / 1e5:.1f}\\times10^5"
N["nodesS"] = f"{S_['nodes'] / 1e4:.1f}\\times10^4"
N["epsS"] = sci(S_["eps"]); N["epsL"] = sci(L_["eps"])
N["secS"] = f"{S_['seconds']:.0f}"; N["itersS"] = str(S_["iters"])
N["msIterS"] = f"{1000 * S_['seconds'] / S_['iters']:.0f}"
N["msIterL"] = f"{1000 * L_['seconds'] / L_['iters']:.0f}"
fb = {(tuple(f["cfg"]), f["full_deal"]): f for f in e1["feedback1"]}
prov = lambda f: f["provably_false"] / f["lie_rate"]
N["fullProvT"] = pct(prov(fb[((3, 2, 3, 4), True)]))
N["fullProvSix"] = pct(prov(fb[((3, 2, 3, 6), True)]))
part = [f for f in e1["feedback1"] if not f["full_deal"]]
N["partLieLo"] = pct(min(f["lie_rate"] for f in part)); N["partLieHi"] = pct(max(f["lie_rate"] for f in part))
N["partProvLo"] = pct(min(prov(f) for f in part)); N["partProvHi"] = pct(max(prov(f) for f in part))

# ----------------------------------------------------------------------- E2
e2 = load_pickle("e2.pkl")
R2 = e2["results"]
pas = [r for r in R2 if r["mode"] == "passive"]
act = [r for r in R2 if r["mode"] == "active"]
for g_, tag_ in ((10, "Ten"), (50, "Fifty")):
    N["gamErr" + tag_] = f3(np.mean([r["err_gamma"][g_ - 1] for r in pas]))
    N["betErr" + tag_] = f3(np.mean([r["err_beta"][g_ - 1] for r in pas]))
N["revPas"] = f"{np.mean([r['reveals'][-1] / len(r['reveals']) for r in pas]):.2f}"
N["revAct"] = f"{np.mean([r['reveals'][-1] / len(r['reveals']) for r in act]):.2f}"
loss, th = e2["loss"], e2["thetas"]
eqset = {nm: int((loss[:, th.index(t)] <= e2["eta"]).sum()) for nm, t in zip(P.ARCH, [(0.1, 0.1), (0.9, 0.1), (0.3, 0.9), (0.5, 0.5), (0.9, 0.9)])}
N["eqSetSize"] = str(eqset["Suspicious caller"])
assert eqset["Suspicious caller"] == eqset["Random"] == eqset["Paranoid liar"]
hits = []
for r in [r for r in pas if r["name"] == "Aggressive bluffer"]:
    h = np.nonzero(r["b_eff"] >= 0.9)[0]
    if len(h): hits.append(h[0] + 1)
N["tauAggr"] = f"{np.median(hits):.0f}"

# ----------------------------------------------------------------------- E3
def e3_numbers(which, sfx):
    d = load_pickle(f"e3_{which}.pkl")
    s = load_json(f"e3_{which}_summary.json")
    P.ALPHAS_ = np.array(d["alphas"])
    alphas = P.ALPHAS_
    N["linErr" + sfx] = sci(max(s["lin_err_max"], 1e-16))
    N["curv" + sfx] = sci(s["max_second_diff"])
    N["bracket" + sfx] = sci(d["eps_eq"])                 # eps_max(x*) = 2 eps_avg(x*) = bracket width
    orc = {o["name"]: o for o in d["oracle"]}
    summ = {(r["name"], str(r["n"])): r for r in s["summary"]}
    share = {}
    for nm in P.ARCH:
        Gs = orc[nm]["u_or"] - orc[nm]["u_eq"]
        mix = summ[(nm, "inf")]["F05"] / Gs
        seat = summ[(nm, "inf")]["Fs05"] / Gs
        rnr = max(r["u"] - orc[nm]["u_eq"] for r in s["rnr"] if r["name"] == nm and r["eps"] <= EPS_BAR) / Gs
        share[nm] = (mix, seat, rnr)
    # total regret per game at eps_bar (Lemma 2), mean over archetypes
    reg = {}
    for n in d["nlist"]:
        tot = []
        for nm in P.ARCH:
            Gs = orc[nm]["u_or"] - orc[nm]["u_eq"]
            for r in [r for r in d["res"] if r["name"] == nm and r["n"] == n]:
                a = P.a_max(alphas, r["eps_s"].max(1), EPS_BAR)
                tot.append((1 - a) * Gs + a * (orc[nm]["u_or"] - r["u_br"]))
        reg[str(n)] = np.mean(tot)
    rat = [x["Fc05"] / x["F05"] for x in s["summary"] if x["F05"] > 0]
    return d, s, orc, summ, share, reg, rat


d, s, orc, summ, share, reg, rat = e3_numbers("small", "S")
for nm, key in (("Honest", "Hon"), ("Aggressive bluffer", "Agg"), ("Paranoid liar", "Par")):
    N["mix" + key], N["seat" + key], N["rnr" + key] = (pct(v) for v in share[nm])
N["regZero"] = f3(reg["0"]); N["regInf"] = f3(reg["inf"])
N["epsPrior"] = f3(np.mean([summ[(nm, "0")]["eps_br"] for nm in P.ARCH]))
N["epsWorstS"] = f"{max(o['eps_or'] for o in d['oracle']):.2f}"
N["honZero"] = f3(summ[("Honest", "0")]["F05"]); N["honInf"] = f3(summ[("Honest", "inf")]["F05"])
N["honInfSeat"] = f3(summ[("Honest", "inf")]["Fs05"])
N["rnrEpsMaxS"] = sci(max(r["eps"] for r in s["rnr"] if r["p"] <= 0.75))
N["nSamplesS"] = str(len(d["res"]))
hs = summ[("Honest", "inf")]["eps_br_seats"]
N["honSeatOne"], N["honSeatTwo"] = f"{hs[0]:.2f}", f"{hs[1]:.2f}"
N["chordTightS"] = pct(np.mean(rat))

d, s, orc, summ, share, reg, rat = e3_numbers("large", "L")
mixL = [v[0] for v in share.values()]; seatL = [v[1] for v in share.values()]; rnrL = [v[2] for v in share.values()]
N["mixLlo"], N["mixLhi"] = pct(min(mixL)), pct(max(mixL))
N["rnrLlo"], N["rnrLhi"] = pct(min(rnrL)), pct(max(rnrL))
N["seatGainLlo"] = pct(min(b - a for a, b in zip(mixL, seatL))); N["seatGainLhi"] = pct(max(b - a for a, b in zip(mixL, seatL)))
N["chordTightL"] = pct(np.mean(rat)); N["chordTightLmin"] = pct(np.min(rat))
wl = [max(x["eps_br_seats"]) for x in s["summary"] if x["n"] == "inf"]
N["epsBRLlo"], N["epsBRLhi"] = f"{min(wl):.2f}", f"{max(wl):.2f}"
N["honZeroL"] = f3(summ[("Honest", "0")]["F05"]); N["honInfL"] = f3(summ[("Honest", "inf")]["F05"])

# ------------------------------------------------------------------- E4 / E5
def paired(R, opps, a, b):
    def tot(l, o):
        rr = sorted([r for r in R if r["learner"] == l and r["opp"] == o], key=lambda r: r["seed"])
        return np.array([np.mean(r["u"] - r["u_eq"]) for r in rr])
    return np.mean([tot(a, o) - tot(b, o) for o in opps], 0)


e4 = load_pickle("e4.pkl"); t4 = load_json("e4_summary.json")
R4, opps4 = e4["results"], e4["opps"]
orc4 = t4["Oracle"]["gain"]
sh = lambda l, t=t4, o=None: pct(t[l]["gain"] / (o or orc4))
N["greedyShare"] = sh("Bayes-greedy")
N["greedyRegret"] = f"{t4['Bayes-greedy']['regret']:.2f}"
mf = [t4[l]["gain"] / orc4 for l in ("UCB1", "Exp3", "SW-greedy")]
N["mfLo"], N["mfHi"] = pct(min(mf)), pct(max(mf))
N["greedyVsUCB"] = pm(paired(R4, opps4, "Bayes-greedy", "UCB1"))
N["safeVsUCB"] = pm(paired(R4, opps4, "Bayes-safe", "UCB1"))
N["seatVsSafe"] = pm(paired(R4, opps4, "Bayes-safe-seat", "Bayes-safe"))
N["rnrVsSeat"] = pm(paired(R4, opps4, "RNR-MAP", "Bayes-safe-seat"))
N["activeVsSafe"] = pm(-paired(R4, opps4, "Bayes-safe-active", "Bayes-safe"))
capped = ["Bayes-safe", "Bayes-safe-seat", "Bayes-safe-gated", "Bayes-safe-active"]
N["cappedGames"] = f"{sum(len(r['u']) for r in R4 if r['learner'] in capped):,}".replace(",", "{,}")
N["cappedMax"] = f"{max(max(r['eps0'].max(), r['eps1'].max()) for r in R4 if r['learner'] in capped):.4f}"
N["greedyEpsMax"] = f"{t4['Bayes-greedy']['eps_max']:.2f}"
N["greedyViol"] = pct(t4["Bayes-greedy"]["viol"])
N["safeShare"] = sh("Bayes-safe"); N["seatShare"] = sh("Bayes-safe-seat"); N["rnrShare"] = sh("RNR-MAP")
N["safeAlpha"] = f"{np.nanmean([r['alpha'].mean() for r in R4 if r['learner'] == 'Bayes-safe']):.2f}"
N["ucbEpsMax"] = f"{t4['UCB1']['eps_max']:.2f}"
N["rnrEpsMaxE"] = sci(t4["RNR-MAP"]["eps_max"])
rv = lambda l: np.mean([r["reveals"].mean() for r in R4 if r["learner"] == l])
N["revActive"], N["revSafe"] = f"{rv('Bayes-safe-active'):.2f}", f"{rv('Bayes-safe'):.2f}"
N["revIncrease"] = pct(rv("Bayes-safe-active") / rv("Bayes-safe") - 1)
ci4 = max(1.96 * np.std([np.mean(r["u"] - r["u_eq"]) for r in R4 if r["learner"] == l and r["opp"] == o], ddof=1) / np.sqrt(30)
          for l in e4["learners"] for o in opps4)
N["ciMaxE"] = f"{ci4:.3f}"

e5 = load_pickle("e5.pkl"); t5 = load_json("e5_summary.json")
R5, opps5 = e5["results"], e5["opps"]
orc5 = t5["Oracle"]["gain"]
offgrid = [o for o in opps5 if o.startswith("Off-grid")]
def regret(l, o): return np.mean([r["u_oracle"].sum() - r["u"].sum() for r in R5 if r["learner"] == l and r["opp"] == o])
N["offRegMax"] = f"{max(regret(l, o) for l in ('Bayes-greedy', 'RNR-MAP') for o in offgrid):.1f}"
eqr = [regret("Equilibrium", o) for o in offgrid]
N["offEqLo"], N["offEqHi"] = f"{min(eqr):.1f}", f"{max(eqr):.1f}"
N["sizeGreedy"] = f3(t5["Bayes-greedy"]["per_opp"]["Size-dependent"][0])
N["sizeOracle"] = f3(t5["Oracle"]["per_opp"]["Size-dependent"][0])


def worst_floor(l):
    # u in seat i minus v_i*; v_0* = game value of S (E1), v_1* = -v_0*
    v0 = S_["value"]
    rr = [r for r in R5 if r["learner"] == l]
    return min(min((r["u"][0::2] - v0).min(), (r["u"][1::2] + v0).min()) for r in rr)


N["floorCapped"] = f3(-min(worst_floor(l) for l in ("Bayes-safe", "Bayes-safe-seat", "Bayes-safe-gated")))
N["floorUCB"] = f"{-worst_floor('UCB1'):.2f}"
N["ucbVsEq"] = f3(-t5["UCB1"]["per_opp"]["Equilibrium"][0])
N["ucbMean"] = f3(t5["UCB1"]["gain"])
N["eqLossCapped"] = sci(max(-t5[l]["per_opp"]["Equilibrium"][0] for l in ("Bayes-safe", "Bayes-safe-seat", "Bayes-safe-gated")))
capped5 = ("Bayes-safe", "Bayes-safe-seat", "Bayes-safe-gated")
N["cappedWorstGainE"] = sci(-min(t5[l]["per_opp"][o][0] for l in capped5 for o in opps5))
N["rnrEpsMaxEfive"] = sci(t5["RNR-MAP"]["eps_max"])
N["seatShareE"] = sh("Bayes-safe-seat", t5, orc5); N["safeShareE"] = sh("Bayes-safe", t5, orc5)
N["rnrGainE"] = f3(t5["RNR-MAP"]["gain"]); N["rnrShareE"] = sh("RNR-MAP", t5, orc5)

# wall-clock time of the last full pipeline run (written by run_all.py)
ri = os.path.join(RES, "run_info.json")
if os.path.exists(ri):
    N["pipelineHours"] = f"{load_json('run_info.json')['total_seconds'] / 3600:.1f}"
else:
    N["pipelineHours"] = "a few"

# ----------------------------------------------------------------- write
with open(os.path.join(RES, "numbers.tex"), "w", newline="\n") as fh:
    fh.write("% generated by experiments/report_numbers.py -- do not edit\n")
    for k, v in N.items():
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")
save_json(N, "numbers.json")
print(f"report numbers: {len(N)} macros written to results/numbers.tex")
