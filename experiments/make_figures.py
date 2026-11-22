"""Figures and summary tables for all experiments (run after the experiments).

    python3 make_figures.py   # (called by `python plots.py` at the repository root)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

# reference categorical palette (fixed order, never cycled)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "savefig.dpi": 200, "savefig.bbox": "tight",
})
ARCH = ["Honest", "Aggressive bluffer", "Suspicious caller", "Random", "Paranoid liar"]


def ci(x, axis=0):
    x = np.asarray(x, float)
    n = x.shape[axis]
    return x.mean(axis), 1.96 * x.std(axis, ddof=1) / np.sqrt(n) if n > 1 else 0 * x.mean(axis)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"CreationDate": None})   # reproducible bytes
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


# ----------------------------------------------------------------------- E1
def e1():
    d = load_json("e1.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, (t, lab) in zip(axes, [("K3m2h2T4", "Bluff(3,2,2,4)"), ("K3m2h2T6", "S = Bluff(3,2,2,6)"),
                                   ("K4m2h3T4", "L = Bluff(4,2,3,4)")]):
        for j, (alg, nm) in enumerate([("cfr", "CFR"), ("cfr+", "CFR+")]):
            h = np.array(d["curves"][f"{t}_{alg}"])
            ax.loglog(h[:, 0], h[:, 1], color=C[j], label=nm)
            ax.annotate(nm, (h[-1, 0], h[-1, 1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")
        it = np.array([10, 2000]); ax.loglog(it, 0.1 / np.sqrt(it / 10), ls=":", color=GRAY, lw=1)
        ax.set_title(lab); ax.set_xlabel("iterations $N$")
    axes[0].set_ylabel(r"$\varepsilon_{\mathrm{avg}}(\bar\pi^N)$ (seat-averaged)")
    axes[0].legend(loc="lower left")
    axes[0].text(12, 0.012, r"$O(1/\sqrt{N})$", fontsize=7, color=GRAY)
    save(fig, "e1_cfr")
    rows = []
    for s in d["sizes"]:
        rows.append(f"Bluff({s['K']},{s['m']},{s['h']},{s['T']}) & {s['nodes']:,} & {s['infosets']:,} & "
                    f"{s['sequences']:,} & {s['value']:+.3f} & {s['eps']:.1e} & {s['seconds']:.0f}\\\\")
    rows2 = []
    for f in d["feedback1"]:
        K, m, h, T = f["cfg"]
        prov_share = f["provably_false"] / f["lie_rate"] if f["lie_rate"] > 0 else 0
        rows2.append(f"Bluff({K},{m},{h},{T}) & {'full' if f['full_deal'] else 'partial'} & {f['undealt']} & "
                     f"{f['opp_hands']:.1f} & {100*f['lie_rate']:.1f} & {100*prov_share:.0f} & "
                     f"{100*f['decidable']:.1f} & {100*f['call_rate']:.1f} & {f['value']:+.2f}\\\\")
    with open(os.path.join(RES, "e1_sizes.tex"), "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    with open(os.path.join(RES, "e1_fb1.tex"), "w", newline="\n") as fh:
        fh.write("\n".join(rows2) + "\n")
    print("E1 figures done")


# ----------------------------------------------------------------------- E2
def e2():
    d = load_pickle("e2.pkl")
    R = d["results"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for j, nm in enumerate(ARCH):
        for mode, ls in (("passive", "-"), ("active", "--")):
            rr = [r for r in R if r["name"] == nm and r["mode"] == mode]
            m, _ = ci([r["b_eff"] for r in rr])
            axes[0].plot(np.arange(1, len(m) + 1), m, color=C[j], ls=ls, lw=1.3,
                         label=nm if mode == "passive" else None)
    axes[0].set_xlabel("games"); axes[0].set_ylabel(r"$b_g(E_\eta(\theta^*))$")
    axes[0].set_title("Effective identification")
    axes[0].set_xscale("log")
    axes[0].plot([], [], color=GRAY, ls="-", label="passive"); axes[0].plot([], [], color=GRAY, ls="--", label="active")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=7, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
    # beta vs gamma error, passive, averaged over archetypes; vs games and vs reveals
    for k, (key, lab) in enumerate((("err_beta", r"$\beta$ (bluff)"), ("err_gamma", r"$\gamma$ (call)"))):
        rr = [r for r in R if r["mode"] == "passive"]
        E = np.array([r[key] for r in rr]); m, c = ci(E)
        g = np.arange(1, len(m) + 1)
        axes[1].plot(g, m, color=C[k], label=lab); axes[1].fill_between(g, m - c, m + c, color=C[k], alpha=.15, lw=0)
        # vs reveals: interpolate per run on a reveal grid
        grid = np.arange(0, 101)
        V = np.array([np.interp(grid, r["reveals"], r[key], left=np.nan, right=np.nan) for r in rr])
        mm = np.nanmean(V, 0)
        axes[2].plot(grid, mm, color=C[k], label=lab)
    axes[1].set_xscale("log"); axes[1].set_xlabel("games"); axes[1].set_ylabel("posterior-mean error")
    axes[1].set_title("Which propensity is learnt first?"); axes[1].legend()
    axes[2].set_xlabel("cumulative reveals"); axes[2].set_title("Same, per reveal"); axes[2].legend()
    fig.subplots_adjust(wspace=0.35)
    save(fig, "e2_ident")
    # loss matrix heat map
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    L = d["loss"]
    im = ax.imshow(L, cmap="Blues", vmin=0)
    labs = [f"({b},{c})" for b, c in d["thetas"]]
    ax.set_xticks(range(25)); ax.set_xticklabels(labs, rotation=90, fontsize=5)
    ax.set_yticks(range(25)); ax.set_yticklabels(labs, fontsize=5)
    ax.set_xlabel(r"true type $\theta^*=(\beta,\gamma)$"); ax.set_ylabel(r"best response to type $\theta$")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label("payoff loss", fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, "e2_loss")
    # table: tau
    rows = []
    for nm in ARCH:
        row = [nm]
        for mode in ("passive", "active"):
            rr = [r for r in R if r["name"] == nm and r["mode"] == mode]
            tg, tr, fin = [], [], []
            for r in rr:
                hit = np.nonzero(r["b_eff"] >= 0.9)[0]
                if len(hit):
                    tg.append(hit[0] + 1); tr.append(r["reveals"][hit[0]])
                fin.append(r["b_true"][-1])
            row += [f"{np.median(tg):.0f}" if len(tg) > len(rr) / 2 else ">150",
                    f"{np.median(tr):.0f}" if len(tr) > len(rr) / 2 else "--",
                    f"{100*len(tg)/len(rr):.0f}", f"{np.mean(fin):.2f}",
                    f"{np.mean([r['reveals'][-1] for r in rr])/len(rr[0]['reveals']):.2f}"]
        rows.append(" & ".join(row) + "\\\\")
    with open(os.path.join(RES, "e2_table.tex"), "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print("E2 figures done")


# ----------------------------------------------------------------------- E3
def a_max(alphas, eps, eb):
    """Largest alpha whose piecewise-linear interpolation of eps(alpha) is <= eb.
    By convexity the interpolation over-estimates eps, so this alpha is feasible."""
    if eps[0] > eb + 1e-12:
        return 0.0
    ok = np.nonzero(eps <= eb + 1e-12)[0]
    j = ok.max()                      # sublevel set is an interval [0, alpha_max]
    if j == len(alphas) - 1:
        return 1.0
    e0, e1 = eps[j], eps[j + 1]
    return alphas[j] + (alphas[j + 1] - alphas[j]) * (eb - e0) / (e1 - e0)


def emax(r):
    """eps_max(alpha) = max over seats of the certified per-seat exploitability."""
    return r["eps_s"].max(1)


def seat_gain(r, eb):
    """Best gain with a separate alpha per seat (per-seat constraints decouple)."""
    return 0.5 * sum((r["u_br_s"][i] - r["u_eq_s"][i]) * a_max(ALPHAS_, r["eps_s"][:, i], eb) for i in (0, 1))


def e3_frontiers(res, alphas, nlist, name, eps_grid):
    """F_n(eps_bar) for each n: mean over belief samples of the best feasible gain."""
    out = {}
    for n in nlist:
        rr = [r for r in res if r["name"] == name and r["n"] == n]
        F, Fc, Fs = [], [], []
        for r in rr:
            em = emax(r)
            row, rowc, rows_ = [], [], []
            e0, e1 = em[0], em[-1]
            for eb in eps_grid:
                row.append((r["u_br"] - r["u_eq"]) * a_max(alphas, em, eb))
                rows_.append(seat_gain(r, eb))
                ac = 1.0 if e1 <= eb else max(0.0, (eb - e0) / (e1 - e0))
                rowc.append((r["u_br"] - r["u_eq"]) * ac)
            F.append(row); Fc.append(rowc); Fs.append(rows_)
        out[n] = (np.array(F), np.array(Fc), np.array(Fs))
    return out


ALPHAS_ = None


def e3(which):
    global ALPHAS_
    d = load_pickle(f"e3_{which}.pkl")
    res, alphas, nlist = d["res"], np.array(d["alphas"]), d["nlist"]
    ALPHAS_ = alphas
    orc = {o["name"]: o for o in d["oracle"]}
    eps_grid = np.linspace(0, 0.3, 61)
    fig, axes = plt.subplots(1, 5, figsize=(8.0, 2.1), sharey=False)
    show = [1, 5, 20, "inf"]
    summary = []
    for ax, nm in zip(axes, ARCH):
        Fs = e3_frontiers(res, alphas, nlist, nm, eps_grid)
        Gs = orc[nm]["u_or"] - orc[nm]["u_eq"]
        for j, n in enumerate(show):
            m = Fs[n][0].mean(0)
            ax.plot(eps_grid, m, color=C[j], lw=1.3, label=f"n={n if n!='inf' else r'$\infty$'}")
        ax.plot(eps_grid, Fs["inf"][2].mean(0), color=C[3], lw=1.0, ls="--", label=r"n=$\infty$, per-seat $\alpha$")
        rn = sorted([r for r in d["rnr"] if r["name"] == nm], key=lambda r: r["eps"])
        ax.plot([r["eps"] for r in rn], [r["u"] - orc[nm]["u_eq"] for r in rn], color=INK, ls=":", lw=1,
                marker="o", ms=2.5, label="RNR (true model)")
        ax.axhline(Gs, color=GRAY, lw=0.6, ls="--")
        ax.set_title(nm, fontsize=8); ax.set_xlabel(r"safety level $\bar\varepsilon$")
        ax.set_xlim(0, 0.3)
        for n in nlist:
            F, Fc, Fsr = Fs[n]
            i5 = np.argmin(abs(eps_grid - EPS_BAR))
            summary.append(dict(name=nm, n=n, F05=F[:, i5].mean(), Fc05=Fc[:, i5].mean(), Fs05=Fsr[:, i5].mean(),
                                Gs=Gs, F_inf=F[:, -1].mean(),
                                eps_br=np.mean([emax(r)[-1] for r in res if r["name"] == nm and r["n"] == n]),
                                eps_br_seats=np.mean([r["eps_s"][-1] for r in res if r["name"] == nm and r["n"] == n], 0)))
    axes[0].set_ylabel(r"gain $F_n(\bar\varepsilon)$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.26))
    fig.subplots_adjust(wspace=0.5)
    save(fig, f"e3_frontier_{which}")

    # value of information + price decomposition at eps_bar = EPS_BAR (best feasible alpha)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    fig.subplots_adjust(wspace=0.32)
    ns = [n for n in nlist if n != "inf"]
    xs = [max(n, 0.5) for n in ns]
    for j, nm in enumerate(ARCH):
        Gs = orc[nm]["u_or"] - orc[nm]["u_eq"]
        v = [np.mean([(r["u_br"] - r["u_eq"]) for r in res if r["name"] == nm and r["n"] == n]) / Gs for n in ns]
        axes[0].plot(xs, v, color=C[j], marker="o", ms=3, label=nm)
    axes[0].set_xscale("log"); axes[0].set_xticks(xs); axes[0].set_xticklabels([str(n) for n in ns])
    axes[0].axhline(1, color=GRAY, lw=.6, ls="--")
    axes[0].set_xlabel("information level $n$ (reveals)"); axes[0].set_ylabel(r"$\mathcal{G}^{(n)}(1)\,/\,G^*$")
    axes[0].set_title("Gain of the posterior BR (unconstrained)")
    axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    # price decomposition averaged across archetypes, alpha = best feasible at EPS_BAR
    ps, pi = [], []
    for n in nlist:
        a_s, a_i = [], []
        for nm in ARCH:
            Gs = orc[nm]["u_or"] - orc[nm]["u_eq"]
            for r in [r for r in res if r["name"] == nm and r["n"] == n]:
                a = a_max(alphas, emax(r), EPS_BAR)
                Delta = orc[nm]["u_or"] - r["u_br"]
                a_s.append((1 - a) * Gs); a_i.append(a * Delta)
        ps.append(np.mean(a_s)); pi.append(np.mean(a_i))
    xl = [str(n) if n != "inf" else r"$\infty$" for n in nlist]
    x = np.arange(len(nlist))
    axes[1].bar(x, ps, color=C[0], width=.7, label="price of safety $(1-\\alpha)G^*$")
    axes[1].bar(x, pi, bottom=ps, color=C[1], width=.7, label="price of ignorance $\\alpha\\Delta$",
                edgecolor="white", linewidth=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(xl)
    axes[1].set_xlabel("information level $n$"); axes[1].set_ylabel("regret per game")
    axes[1].set_title(rf"Price of safety vs. ignorance at $\bar\varepsilon={EPS_BAR}$")
    axes[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    axes[1].set_ylim(0, max(np.array(ps) + np.array(pi)) * 1.05)
    save(fig, f"e3_voi_{which}")
    lin = max(r["lin_err"] for r in res)
    # convexity check of eps(alpha)
    conv = min((np.diff(r["eps_s"], 2, axis=0)).min() for r in res)
    curv = max((np.diff(r["eps_s"], 2, axis=0)).max() for r in res)
    save_json(dict(summary=summary, lin_err_max=lin, min_second_diff=conv, max_second_diff=curv,
                   oracle=d["oracle"], rnr=d["rnr"], eps_eq=d["eps_eq"]), f"e3_{which}_summary.json")
    print("E3", which, "figures done; max linearity error", lin, "min 2nd diff of eps", conv)


# ------------------------------------------------------------------- E4/E5
STY = {"Equilibrium": (GRAY, "-"), "Oracle": (INK, ":"), "Bayes-greedy": (C[0], "-"),
       "Bayes-MAP": (C[0], ":"), "Bayes-gated": (C[6], "-"), "Bayes-safe": (C[1], "-"), "Bayes-safe-seat": (C[4], "-"),
       "Bayes-safe-gated": (C[1], "--"), "Bayes-safe-active": (C[1], ":"), "UCB1": (C[2], "-"),
       "Exp3": (C[2], "--"), "SW-greedy": (C[2], ":"), "RNR-MAP": (C[3], "-")}


GROUPS = [
    ("Reference strategies", ["Equilibrium", "Oracle"]),
    ("Bayesian, unconstrained", ["Bayes-greedy", "Bayes-MAP", "Bayes-gated"]),
    ("Bayesian, safety-capped ($\\varepsilon_{\\max}\\le\\bar\\varepsilon$ enforced)",
     ["Bayes-safe", "Bayes-safe-seat", "Bayes-safe-gated", "Bayes-safe-active"]),
    ("Model-free (bandits over a portfolio)", ["UCB1", "Exp3", "SW-greedy"]),
    ("Robust game-theoretic (no cap enforced)", ["RNR-MAP"]),
]

OFFS = {"Bayes-safe-seat": (5, 3), "Oracle": (4, 3), "Bayes-greedy": (-5, 4), "Bayes-MAP": (4, -8), "Bayes-safe": (-5, 4),
        "Bayes-safe-active": (5, -2), "Bayes-safe-gated": (5, -6), "UCB1": (5, 2), "Exp3": (5, -7),
        "SW-greedy": (5, 2)}


def table_image(which, tab, learners, opps):
    """Render the grouped results table as an image (for the README), from the same data."""
    short = {"Aggressive bluffer": "Aggr.", "Suspicious caller": "Susp.", "Paranoid liar": "Paran.",
             "Size-dependent": "Size-dep.", "Noisy equilibrium": "Noisy eq.", "Equilibrium": "Equil."}
    cols = ["Learner"] + [short.get(o, o.replace("Off-grid ", "")) for o in opps] + ["mean", "R_100", "eps_max mean", "eps_max max"]
    rows, colours = [], []
    for gname, members in GROUPS:
        ms = [l for l in members if l in learners]
        if not ms:
            continue
        rows.append([gname.split(" ($")[0]] + [""] * (len(cols) - 1)); colours.append("#e6e5e0")
        for l in ms:
            t = tab[l]
            rows.append(["  " + l] + [f"{t['per_opp'][o][0]:+.3f}" for o in opps] +
                        [f"{t['gain']:+.3f}", f"{t['regret']:.1f}", f"{t['eps_mean']:.3f}", f"{t['eps_max']:.3f}"])
            colours.append("white")
    fig, ax = plt.subplots(figsize=(1.0 + 0.78 * len(cols), 0.26 * (len(rows) + 1)))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="right", colLoc="right")
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d9d8d2"); cell.set_linewidth(0.4)
        if c == 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
        if r == 0:
            cell.set_text_props(weight="bold")
        elif colours[r - 1] != "white":
            cell.set_facecolor(colours[r - 1]); cell.set_text_props(style="italic")
    tb.auto_set_column_width(list(range(len(cols))))
    fig.savefig(os.path.join(FIG, f"{which}_table.png"))
    plt.close(fig)


def e45(which):
    d = load_pickle(f"{which}.pkl")
    R, learners, opps, G = d["results"], d["learners"], d["opps"], d["G"]
    # summary table: per learner, mean over opponents of (avg gain, avg regret, max eps, mean eps)
    rows, tab = [], {}
    for l in learners:
        gains, regs, epsm, epsx, viol = [], [], [], [], []
        per_opp = {}
        for o in opps:
            rr = [r for r in R if r["learner"] == l and r["opp"] == o]
            ga = np.array([np.mean(r["u"] - r["u_eq"]) for r in rr])
            rg = np.array([np.sum(r["u_oracle"] - r["u"]) for r in rr])
            per_opp[o] = (ga.mean(), 1.96 * ga.std(ddof=1) / np.sqrt(len(ga)))
            gains.append(ga.mean()); regs.append(rg.mean())
            epsm.append(np.mean([r["eps"].mean() for r in rr])); epsx.append(max(r["eps"].max() for r in rr))
            viol.append(np.mean([((r["eps0"] > d["eps_bar"] + 1e-9) | (r["eps1"] > d["eps_bar"] + 1e-9)).mean()
                                 for r in rr]))
        tab[l] = dict(gain=np.mean(gains), regret=np.mean(regs), eps_mean=np.mean(epsm),
                      eps_max=max(epsx), viol=np.mean(viol), per_opp=per_opp)
    save_json(tab, f"{which}_summary.json")
    # per-opponent gain table (LaTeX)
    ncol = len(opps) + 5
    with open(os.path.join(RES, f"{which}_table.tex"), "w", newline="\n") as fh:
        for gi, (gname, members) in enumerate(GROUPS):
            rows = [l for l in members if l in learners]
            if not rows:
                continue
            if gi:
                fh.write("\\midrule\n")
            fh.write(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{gname}}}}}\\\\\n")
            for l in rows:
                t = tab[l]
                cells = [f"{t['per_opp'][o][0]:+.3f}" for o in opps]
                fh.write(f"\\quad {l} & " + " & ".join(cells) + f" & {t['gain']:+.3f} & {t['regret']:.1f} & "
                         f"{t['eps_mean']:.3f} & {t['eps_max']:.3f}\\\\\n")
    # cumulative regret curves per opponent
    ncol = len(opps)
    fig, axes = plt.subplots(1, ncol, figsize=(1.55 * ncol, 2.3), sharey=False)
    show = [l for l in learners if l != "Oracle"]
    for ax, o in zip(axes, opps):
        for l in show:
            rr = [r for r in R if r["learner"] == l and r["opp"] == o]
            cum = np.cumsum([r["u_oracle"] - r["u"] for r in rr], axis=1)
            m = cum.mean(0)
            col, ls = STY[l]
            ax.plot(np.arange(1, G + 1), m, color=col, ls=ls, lw=1.1, label=l)
        ax.set_title(o, fontsize=7.5); ax.set_xlabel("games")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("cumulative regret $R_G$")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, -0.24))
    save(fig, f"{which}_regret")
    # gain vs exploitability scatter (one dot per learner, mean over opponents)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for l in learners:
        t = tab[l]; col, ls = STY[l]
        mk = "o" if ls == "-" else ("s" if ls == "--" else "^")
        ax.scatter(t["eps_mean"], t["gain"], color=col, marker=mk, s=28, edgecolor="white", linewidth=1, zorder=3)
        off = OFFS.get(l, (4, 2))
        ax.annotate(l, (t["eps_mean"], t["gain"]), xytext=off, textcoords="offset points", fontsize=6,
                    color=INK2, ha="right" if off[0] < 0 else "left")
    ax.axvline(d["eps_bar"], color=GRAY, ls="--", lw=.6)
    ax.set_xlabel(r"mean worst-seat exploitability $\varepsilon_{\max}$ of strategies played"); ax.set_ylabel("mean gain over $\\pi^*$ per game")
    ax.set_title("Gain vs. exploitability (mean over opponents)")
    save(fig, f"{which}_scatter")
    table_image(which, tab, learners, opps)
    print(which, "figures done")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["e1", "e2", "e3small", "e3large", "e4", "e5"]
    for t in todo:
        try:
            {"e1": e1, "e2": e2, "e3small": lambda: e3("small"), "e3large": lambda: e3("large"),
             "e4": lambda: e45("e4"), "e5": lambda: e45("e5")}[t]()
        except FileNotFoundError as ex:
            print("skip", t, "(no results yet)")
