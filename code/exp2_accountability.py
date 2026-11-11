"""
Experiment 2 -- Accountability decomposition (Theorem 5) and the effective-influence
identity / rubber-stamp equivalence (Proposition 1) with counterfactual replay under
common random numbers.

Outputs: exp2_results.json and fig_exp2.pdf (six panels):
  (a) exact population-level decomposition in two idealisation orders (stable regime);
  (b) layer loss against the layer's per-decision deviation (stable regime);
  (c) individual specification loss against the principal's value of time, with a
      static softmax prediction;
  (d) layer losses in the stable and the synchronised regime;
  (e) daily cost of the idealised and the actual system in the synchronised regime;
  (f) effective influence of a drifted objective against the approval probability.

Run `python exp2_accountability.py --plot-only` to redraw the figure from the saved JSON.
Set AGMS_USETEX=1 to render text with LaTeX (as in the paper).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import agms_core as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

USETEX = os.environ.get("AGMS_USETEX", "0") == "1"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "mathtext.fontset": "cm",
})
if USETEX:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"})

INK, MUTED = "#1f2328", "#5b6470"
BLUE, ORANGE, GREEN, PURPLE = "#1c5cab", "#c4501d", "#2f8f5b", "#6a4c93"
LAYER = {"S": PURPLE, "I": BLUE, "D": GREEN, "A": ORANGE}
LNAME = {"S": "specification", "I": "interpretation", "D": "deliberation", "A": "actuation"}

NET = A.Net([10, 14, 20], [110, 100, 150], [4, 1, 0])
N = 300
V, G = A.make_population(N)
BASE = dict(q_I=0.10, q_D=0.05, alpha=0.8, T=1.0, rho=0.1, L=5, days=200, disc=0.95)   # stable regime
SYNC = {**BASE, "rho": 0.5}                                                                # synchronised regime
IDEAL = dict(perfect_spec=True, faithful=True, no_halluc=True, full_exec=True)
R = 40
VOT_CUT = (13.0, 30.0)                      # phrase thresholds of the lossy specification (agms_core)


def layered(seed, J=True, key="W", **kw):
    """Values V0..V4 (top-down idealisation order: spec, interpretation, deliberation, actuation)."""
    p = {**BASE, **kw}
    f = lambda **o: A.run(NET, V, G, seed=seed, **{**p, **o})[key]
    v0 = f(perfect_spec=J, faithful=J, no_halluc=J, full_exec=J)
    v1 = f(faithful=J, no_halluc=J, full_exec=J)
    v2 = f(no_halluc=J, full_exec=J)
    v3 = f(full_exec=J)
    v4 = f()
    return np.array([v0, v1, v2, v3, v4])


def ci(x):
    x = np.asarray(x); m = x.mean(0); h = 1.96 * x.std(0, ddof=1) / np.sqrt(len(x)); return m, h


def phrase_vot(v):
    return np.where(v >= VOT_CUT[1], A.PHRASE_VOT[0], np.where(v >= VOT_CUT[0], A.PHRASE_VOT[1], A.PHRASE_VOT[2]))


def softmax_spec_loss(vgrid, tbar, T):
    """Static approximation of the individual specification loss: expected true generalised cost of a
    softmax chooser (temperature T) facing mean route times tbar, with the phrase-implied objective minus
    the same with the true objective."""
    out = []
    for v in vgrid:
        true_cost = tbar + 60.0 * NET.toll / v
        def expected(theta):
            s = -(tbar + theta * NET.toll) / T; s -= s.max(); p = np.exp(s); p /= p.sum()
            return float(p @ true_cost)
        out.append(expected(60.0 / phrase_vot(np.array(v))) - expected(60.0 / v))
    return np.array(out)


def compute():
    res = {}
    # ---- (0) synchronised regime: same decomposition, re-planning probability 0.5 ----
    Vsync = np.array([layered(s, rho=0.5) for s in range(R)])
    lsync = np.diff(Vsync, axis=1); ms, hs = ci(lsync); mts, hts = ci(Vsync[:, 4] - Vsync[:, 0])
    res["population_synchronised"] = {"V_mean": Vsync.mean(0).tolist(), "Lambda_S_I_D_A": ms.tolist(),
                                      "CI95": hs.tolist(), "total": float(mts), "total_CI95": float(hts)}
    tr_ref = A.run(NET, V, G, seed=3, **{**SYNC, **IDEAL})["daily_cost"]
    tr_act = A.run(NET, V, G, seed=3, **SYNC)["daily_cost"]
    tr_ref_s = A.run(NET, V, G, seed=3, **{**BASE, **IDEAL})["daily_cost"]
    res["daily_sd_after_day50"] = {"sync_reference": float(tr_ref[50:].std()), "sync_actual": float(tr_act[50:].std()),
                                   "stable_reference": float(tr_ref_s[50:].std())}
    res["traces_seed3"] = {"sync_reference": tr_ref.tolist(), "sync_actual": tr_act.tolist(),
                           "stable_reference": tr_ref_s.tolist()}
    # ---- (1) population-level decomposition at baseline (stable regime) ----
    Vs = np.array([layered(s) for s in range(R)])                     # R x 5 (cost, min)
    lam = np.diff(Vs, axis=1)                                         # Lambda_S, I, D, A (cost increases)
    tot = Vs[:, 4] - Vs[:, 0]
    m, h = ci(lam); mt, ht = ci(tot)
    res["population_baseline"] = {"V_mean": Vs.mean(0).tolist(),
                                  "Lambda_S_I_D_A": m.tolist(), "CI95": h.tolist(),
                                  "total": float(mt), "total_CI95": float(ht),
                                  "telescoping_max_abs_error": float(np.abs(lam.sum(1) - tot).max())}

    # order dependence: bottom-up order (actuation first)
    def bottom_up(seed):
        f = lambda **o: A.run(NET, V, G, seed=seed, **{**BASE, **o})["W"]
        w0 = f(**IDEAL)
        wa = f(perfect_spec=True, faithful=True, no_halluc=True)          # actual actuation first
        wd = f(perfect_spec=True, faithful=True)
        wi = f(perfect_spec=True)
        w4 = f()
        return np.array([wa - w0, wd - wa, wi - wd, w4 - wi])           # A, D, I, S
    bu = np.array([bottom_up(s) for s in range(R)])
    mb, hb = ci(bu)
    res["population_bottom_up_A_D_I_S"] = {"mean": mb.tolist(), "CI95": hb.tolist()}

    # ---- (2) layer sweeps: response to the layer's TV deviation ----
    sweeps = {}
    for name, par, vals, idx in [("interpretation", "q_I", [0, .05, .1, .2, .3, .4], 1),
                                 ("deliberation", "q_D", [0, .05, .1, .2, .3, .4], 2),
                                 ("actuation", "alpha", [1, .9, .8, .6, .4, .2], 3)]:
        rows = []
        for val in vals:
            Vv = np.array([layered(s, **{par: val}) for s in range(20)])
            lv = np.diff(Vv, axis=1)[:, idx]
            mm, hh = ci(lv)
            rows.append({"value": val, "tv": (val if par != "alpha" else 1 - val), "Lambda": float(mm), "CI95": float(hh)})
        sweeps[name] = rows
    res["sweeps"] = sweeps

    # ---- (3) individual-principal decomposition and the Lipschitz bound ----
    rng = np.random.default_rng(5)
    sample = rng.choice(N, 30, replace=False)
    disc = BASE["disc"]
    indiv = []
    for i in sample:
        J = np.zeros(N, bool); J[i] = True
        Vi = np.array([layered(s, J=J, key="Wi")[:, i] for s in range(12)])
        li = np.diff(Vi, axis=1).mean(0)
        # oscillation of i's daily cost over attainable states: route time range x toll range
        tt_max = NET.times(np.full(3, N))           # all on one route (upper bound per route)
        gc = np.concatenate([NET.t0 + 60 * NET.toll / V[i], tt_max + 60 * NET.toll / V[i]])
        osc = gc.max() - gc.min()
        delta = np.array([1.0, BASE["q_I"], BASE["q_D"], 1 - BASE["alpha"]])
        bound = disc * osc * delta / (1 - disc)     # normalised (W = (1-disc) V) version
        indiv.append({"i": int(i), "vot": float(V[i]), "phrase": A.PHRASES[G[i]],
                      "Lambda": li.tolist(), "bound": bound.tolist()})
    res["individual"] = indiv
    res["individual_bound_holds_all"] = bool(all(np.all(np.abs(r["Lambda"]) <= np.array(r["bound"]) + 1e-9) for r in indiv))
    ratio = np.array([np.array(r["bound"][1:]) / np.maximum(np.abs(r["Lambda"][1:]), 1e-9) for r in indiv])
    res["individual_bound_over_actual_median"] = np.median(ratio, axis=0).tolist()
    # static softmax prediction of the individual specification loss (mean route times, actual system, days 50+)
    tbar = np.mean([A.run(NET, V, G, seed=s, **BASE)["route_times"][50:].mean(0) for s in range(R)], axis=0)
    vgrid = np.linspace(8.0, 46.0, 381)
    pred = softmax_spec_loss(vgrid, tbar, BASE["T"])
    LS = np.array([r["Lambda"][0] for r in indiv]); VV = np.array([r["vot"] for r in indiv])
    res["individual_spec_prediction"] = {"mean_route_times": tbar.tolist(), "vot_grid": vgrid.tolist(),
                                         "Lambda_S_pred": pred.tolist(),
                                         "corr_sim_pred": float(np.corrcoef(LS, softmax_spec_loss(VV, tbar, BASE["T"]))[0, 1]),
                                         "n_negative_Lambda_S": int((LS < 0).sum()),
                                         "n_positive_Lambda_I": int(sum(r["Lambda"][1] > 0 for r in indiv))}

    # ---- (4) effective influence identity and rubber-stamp equivalence ----
    eff = []
    for alpha in [0.0, 0.5, 0.8, 1.0]:
        tot_v = tot_e = tot_s = 0
        for s in range(40):
            ev = A.run(NET, V, G, seed=s, record=True, **{**BASE, "alpha": alpha})["events"]
            tot_v += ev["drift_valid"]; tot_e += ev["drift_exec"]; tot_s += ev["drift_selfexec"]
        sh = tot_s / tot_v
        eff.append({"alpha": alpha, "P_exec_given_drift_empirical": tot_e / tot_v,
                    "formula_s_plus_(1-s)alpha": sh + (1 - sh) * alpha, "P_selfexec_given_drift": sh})
    res["effective_influence"] = eff
    same = all(np.array_equal(A.run(NET, V, G, seed=s, **{**BASE, "alpha": 1.0})["Wi"],
                              A.run(NET, V, G, seed=s, **{**BASE, "full_exec": True})["Wi"]) for s in range(10))
    res["rubber_stamp_pathwise_identical_10_seeds"] = bool(same)
    return res


# ------------------------------------------------------------------ figure
def tx(s_tex, s_plain):
    return s_tex if USETEX else s_plain


def panel_title(ax, letter, text):
    ax.set_title(rf"\textbf{{({letter})}} {text}" if USETEX else rf"$\bf{{({letter})}}$ {text}",
                 fontsize=7.8, loc="left", color=INK, pad=4)


def make_figure(res):
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.55))
    fig.subplots_adjust(left=0.088, right=0.99, top=0.885, bottom=0.085, wspace=0.44, hspace=0.62)
    keys = ["S", "I", "D", "A"]

    # (a) exact decomposition in two orders
    ax = axes[0, 0]
    td = dict(zip(keys, res["population_baseline"]["Lambda_S_I_D_A"]))
    bu = dict(zip(["A", "D", "I", "S"], res["population_bottom_up_A_D_I_S"]["mean"]))
    tot, totci = res["population_baseline"]["total"], res["population_baseline"]["total_CI95"]
    for y, order, vals in [(1, keys, td), (0, ["A", "D", "I", "S"], bu)]:
        left = 0.0
        for k in order:
            w = vals[k]
            ax.barh(y, w, left=left, height=0.52, color=LAYER[k], edgecolor="white", lw=0.6)
            if w > 0.05:
                ax.text(left + w / 2, y, f"{w:.3f}", ha="center", va="center", fontsize=6.0, color="white")
            left += w
    ax.axvspan(tot - totci, tot + totci, color=INK, alpha=0.10, lw=0)
    ax.axvline(tot, color=INK, lw=0.7, ls=(0, (2, 1.5)))
    ax.text(tot - 0.006, 1.86, tx(rf"total $\Lambda={tot:.3f}\pm{totci:.3f}$", rf"total $\Lambda={tot:.3f}\pm{totci:.3f}$"),
            ha="right", va="center", fontsize=6.2, color=INK)
    ax.text(0.0, 1.40, tx(r"top-down, $S\to I\to D\to A$", r"top-down, $S\to I\to D\to A$"), fontsize=6.2, color=INK, va="center")
    ax.text(0.0, 0.40, tx(r"bottom-up, $A\to D\to I\to S$", r"bottom-up, $A\to D\to I\to S$"), fontsize=6.2, color=INK, va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.35, 2.05); ax.set_xlim(0, 0.345)
    ax.set_xlabel("loss (min per traveller-day)")
    panel_title(ax, "a", "Exact, order-dependent split")

    # (b) sweeps
    ax = axes[0, 1]
    base_dev = {"interpretation": BASE["q_I"], "deliberation": BASE["q_D"], "actuation": 1 - BASE["alpha"]}
    for name, k in [("interpretation", "I"), ("deliberation", "D"), ("actuation", "A")]:
        rows = res["sweeps"][name]
        x = np.array([r["tv"] for r in rows]); y = np.array([r["Lambda"] for r in rows]); h = np.array([r["CI95"] for r in rows])
        ax.fill_between(x, y - h, y + h, color=LAYER[k], alpha=0.14, lw=0)
        ax.plot(x, y, color=LAYER[k], lw=1.1, marker="o", ms=2.6, mew=0)
        xb = base_dev[name]; yb = np.interp(xb, x, y)
        ax.plot([xb], [yb], marker="o", ms=4.2, mfc="white", mec=LAYER[k], mew=0.9, zorder=5)
        ax.text(x[-1] + 0.02, y[-1], tx(rf"$\Lambda_{k}$", rf"$\Lambda_{k}$"), color=LAYER[k], fontsize=7, va="center")
    ax.plot([], [], marker="o", ms=4.2, mfc="white", mec=INK, mew=0.9, ls="none", label="baseline")
    ax.legend(loc="upper left", fontsize=6.2, handletextpad=0.3, borderaxespad=0.2)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xlim(-0.02, 0.92)
    ax.set_xlabel(tx(r"deviation $\delta_\ell$ ($q_I$, $q_D$, $1-\alpha$)", r"deviation $\delta_\ell$ ($q_I$, $q_D$, $1-\alpha$)"))
    ax.set_ylabel(tx(r"$\Lambda_\ell$ (min)", r"$\Lambda_\ell$ (min)"))
    panel_title(ax, "b", "Loss grows with deviation")

    # (c) individual specification loss vs value of time
    ax = axes[0, 2]
    pr = res["individual_spec_prediction"]
    vg = np.array(pr["vot_grid"]); yp = np.array(pr["Lambda_S_pred"])
    for lo, hi, lab, shade in [(7, VOT_CUT[0], "cheapest", 0.0), (VOT_CUT[0], VOT_CUT[1], "balanced", 0.06),
                               (VOT_CUT[1], 48, "fastest", 0.0)]:
        if shade:
            ax.axvspan(lo, hi, color=INK, alpha=shade, lw=0)
        ax.text(np.sqrt(lo * hi), 0.88, tx(rf"\textit{{{lab}}}", lab), ha="center", va="center", fontsize=6.2,
                color=MUTED, style="italic")
    for pv in A.PHRASE_VOT:
        ax.axvline(pv, color=MUTED, lw=0.5, ls=(0, (1, 1.5)))
    for seg in [vg < VOT_CUT[0], (vg >= VOT_CUT[0]) & (vg < VOT_CUT[1]), vg >= VOT_CUT[1]]:
        ax.plot(vg[seg], yp[seg], color=INK, lw=0.9)
    LS = np.array([r["Lambda"][0] for r in res["individual"]]); VV = np.array([r["vot"] for r in res["individual"]])
    ax.scatter(VV, LS, s=11, color=PURPLE, edgecolor="white", lw=0.4, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.text(9.5, 0.50, "simulated", fontsize=6.2, color=PURPLE, ha="center", va="center")
    ax.annotate("", xy=(14.0, 0.515), xytext=(11.4, 0.50), arrowprops=dict(arrowstyle="-", lw=0.5, color=PURPLE))
    ax.text(20.0, -0.47, "softmax\nprediction", fontsize=6.2, color=INK, ha="center", va="center", linespacing=1.0)
    ax.annotate("", xy=(20.0, float(np.interp(20.0, vg, yp)) - 0.03), xytext=(20.0, -0.36),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_xscale("log"); ax.set_xlim(7, 48); ax.set_ylim(-0.62, 0.98)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([8, 13, 20, 30, 45]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["8", "13", "20", "30", "45"]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(tx(r"true value of time $v_i$ (\$/h)", r"true value of time $v_i$ (\$/h)"))
    ax.set_ylabel(tx(r"$\Lambda_{S,i}$ (min)", r"$\Lambda_{S,i}$ (min)"))
    panel_title(ax, "c", "Loss per principal")

    # (d) two regimes per layer
    ax = axes[1, 0]
    st, stc = res["population_baseline"]["Lambda_S_I_D_A"], res["population_baseline"]["CI95"]
    sy, syc = res["population_synchronised"]["Lambda_S_I_D_A"], res["population_synchronised"]["CI95"]
    st = st + [res["population_baseline"]["total"]]; stc = stc + [res["population_baseline"]["total_CI95"]]
    sy = sy + [res["population_synchronised"]["total"]]; syc = syc + [res["population_synchronised"]["total_CI95"]]
    cols = [LAYER[k] for k in keys] + [INK]
    for j, c in enumerate(cols):
        ax.errorbar(j - 0.16, st[j], yerr=stc[j], fmt="o", ms=4.0, mfc="white", mec=c, mew=0.9, ecolor=c, elinewidth=0.8, capsize=1.6)
        ax.errorbar(j + 0.16, sy[j], yerr=syc[j], fmt="o", ms=4.0, color=c, ecolor=c, elinewidth=0.8, capsize=1.6)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(3.5, color=MUTED, lw=0.4, ls=(0, (1, 1.5)))
    ax.set_xticks(range(5)); ax.set_xticklabels([tx(rf"$\Lambda_{k}$", rf"$\Lambda_{k}$") for k in keys] + [r"$\Lambda$"])
    ax.errorbar([], [], fmt="o", ms=4.0, mfc="white", mec=INK, mew=0.9, label=tx(r"stable, $\rho=0.1$", r"stable, $\rho=0.1$"))
    ax.errorbar([], [], fmt="o", ms=4.0, color=INK, label=tx(r"synchronised, $\rho=0.5$", r"synchronised, $\rho=0.5$"))
    ax.legend(loc="lower left", fontsize=6.2, handletextpad=0.2, borderaxespad=0.1)
    ax.set_ylim(-1.42, 0.45)
    ax.set_ylabel("loss (min per traveller-day)")
    panel_title(ax, "d", "Synchronisation flips signs")

    # (e) traces
    ax = axes[1, 1]
    tr = res["traces_seed3"]; sd = res["daily_sd_after_day50"]; d = np.arange(len(tr["sync_reference"]))
    sl = slice(60, 160)
    ax.plot(d[sl], np.array(tr["stable_reference"])[sl], color=MUTED, lw=0.8, ls=(0, (3, 1.5)),
            label=f"stable reference (s.d. {sd['stable_reference']:.2f})")
    ax.plot(d[sl], np.array(tr["sync_actual"])[sl], color=BLUE, lw=0.9, label=f"actual (s.d. {sd['sync_actual']:.1f})")
    ax.plot(d[sl], np.array(tr["sync_reference"])[sl], color=INK, lw=0.9, label=f"idealised (s.d. {sd['sync_reference']:.1f})")
    ax.set_ylim(18.5, 61)
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.6, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
    ax.set_xlabel(tx(r"day ($\rho=0.5$, one replication)", r"day ($\rho=0.5$, one replication)"))
    ax.set_ylabel("mean generalised cost (min)")
    panel_title(ax, "e", "Ideal agents herd")

    # (f) effective influence
    ax = axes[1, 2]
    eff = res["effective_influence"]
    a = np.array([e["alpha"] for e in eff]); emp = np.array([e["P_exec_given_drift_empirical"] for e in eff])
    s = np.array([e["P_selfexec_given_drift"] for e in eff]); form = s + (1 - s) * a
    ax.fill_between(a, 0, s, color=ORANGE, alpha=0.42, lw=0)
    ax.fill_between(a, s, form, color=ORANGE, alpha=0.16, lw=0)
    ax.plot(a, form, color=INK, lw=0.9, label=tx(r"$s_O+(1-s_O)\alpha$", r"$s_O+(1-s_O)\alpha$"))
    ax.plot(a, emp, ls="none", marker="o", ms=3.6, mfc="white", mec=INK, mew=0.9, label="simulated")
    ax.text(0.40, 0.11, tx(r"self-executed, $s_O$", r"self-executed, $s_O$"), fontsize=6.2, color="#6e2a0c", ha="center")
    ax.text(0.56, 0.40, "approved by\nthe reviewer", fontsize=6.2, color="#8a3a14", ha="center", linespacing=1.0)
    ax.annotate("rubber stamp:\nfull autonomy", xy=(1.0, 1.0), xytext=(0.77, 0.62), fontsize=6.2, color=INK,
                ha="center", va="center", linespacing=1.0,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkB=3))
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.4, handletextpad=0.3, borderaxespad=0.1)
    ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.06)
    ax.set_xlabel(tx(r"approval probability $\alpha$", r"approval probability $\alpha$"))
    ax.set_ylabel(tx(r"$\mathbb{P}$(drifted objective executed)", r"P(drifted objective executed)"))
    panel_title(ax, "f", "Oversight and influence")

    # shared layer legend
    hs = [plt.Rectangle((0, 0), 1, 1, color=LAYER[k]) for k in keys]
    fig.legend(hs, [f"{LNAME[k]} ({k})" for k in keys], loc="upper center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.0, handleheight=0.8, columnspacing=1.6, handletextpad=0.5)
    fig.savefig(os.path.join(OUT, "fig_exp2.pdf"))
    plt.close(fig)


def main():
    path = os.path.join(OUT, "exp2_results.json")
    if "--plot-only" in sys.argv:
        with open(path) as f:
            res = json.load(f)
    else:
        res = compute()
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k not in ("individual", "traces_seed3", "individual_spec_prediction")}, indent=1))
        print(json.dumps({k: v for k, v in res["individual_spec_prediction"].items() if k not in ("vot_grid", "Lambda_S_pred")}, indent=1))
    make_figure(res)


if __name__ == "__main__":
    main()
