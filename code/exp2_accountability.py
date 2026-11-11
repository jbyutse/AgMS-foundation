"""
Experiment 2 -- Accountability decomposition (Theorem 5) and the effective-influence
identity / rubber-stamp equivalence (Theorem 4) with counterfactual replay under
common random numbers.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import agms_core as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.6, "font.family": "serif"})
NET = A.Net([10, 14, 20], [110, 100, 150], [4, 1, 0])
N = 300
V, G = A.make_population(N)
BASE = dict(q_I=0.10, q_D=0.05, alpha=0.8, T=1.0, rho=0.1, L=5, days=200, disc=0.95)   # stable regime
SYNC = {**BASE, "rho": 0.5}                                                                # synchronised regime
R = 40


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


def main():
    res = {}
    # ---- (0) synchronised regime: same decomposition, re-planning probability 0.5 ----
    Vsync = np.array([layered(s, rho=0.5) for s in range(R)])
    lsync = np.diff(Vsync, axis=1); ms, hs = ci(lsync); mts, hts = ci(Vsync[:, 4] - Vsync[:, 0])
    res["population_synchronised"] = {"V_mean": Vsync.mean(0).tolist(), "Lambda_S_I_E_A": ms.tolist(),
                                      "CI95": hs.tolist(), "total": float(mts), "total_CI95": float(hts)}
    tr_ref = A.run(NET, V, G, seed=3, **{**SYNC, "perfect_spec": True, "faithful": True, "no_halluc": True, "full_exec": True})["daily_cost"]
    tr_act = A.run(NET, V, G, seed=3, **SYNC)["daily_cost"]
    tr_ref_s = A.run(NET, V, G, seed=3, **{**BASE, "perfect_spec": True, "faithful": True, "no_halluc": True, "full_exec": True})["daily_cost"]
    res["daily_sd_after_day50"] = {"sync_reference": float(tr_ref[50:].std()), "sync_actual": float(tr_act[50:].std()),
                                   "stable_reference": float(tr_ref_s[50:].std())}
    # ---- (1) population-level decomposition at baseline (stable regime) ----
    Vs = np.array([layered(s) for s in range(R)])                     # R x 5 (cost, min)
    lam = np.diff(Vs, axis=1)                                         # Lambda_S, I, E, A (cost increases)
    tot = Vs[:, 4] - Vs[:, 0]
    m, h = ci(lam); mt, ht = ci(tot)
    res["population_baseline"] = {"V_mean": Vs.mean(0).tolist(),
                                  "Lambda_S_I_E_A": m.tolist(), "CI95": h.tolist(),
                                  "total": float(mt), "total_CI95": float(ht),
                                  "telescoping_max_abs_error": float(np.abs(lam.sum(1) - tot).max())}
    # order dependence: bottom-up order (actuation first)
    def bottom_up(seed):
        p = BASE
        f = lambda **o: A.run(NET, V, G, seed=seed, **{**p, **o})["W"]
        w0 = f(perfect_spec=True, faithful=True, no_halluc=True, full_exec=True)
        wa = f(perfect_spec=True, faithful=True, no_halluc=True)          # actual actuation first
        we = f(perfect_spec=True, faithful=True)
        wi = f(perfect_spec=True)
        w4 = f()
        return np.array([wa - w0, we - wa, wi - we, w4 - wi])           # A, E, I, S
    bu = np.array([bottom_up(s) for s in range(R)])
    mb, hb = ci(bu)
    res["population_bottom_up_A_E_I_S"] = {"mean": mb.tolist(), "CI95": hb.tolist()}

    # ---- (2) layer sweeps: linear response in the layer's TV deviation ----
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

    # ---- (4) effective influence identity and rubber-stamp equivalence ----
    eff = []
    for alpha in [0.0, 0.5, 0.8, 1.0]:
        tot_d = tot_v = tot_e = tot_s = 0
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

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    labels = ["$\\Lambda_S$", "$\\Lambda_I$", "$\\Lambda_E$", "$\\Lambda_A$"]
    x = np.arange(4)
    axes[0].bar(x - 0.2, m, 0.38, yerr=h, color=C[0], label="stable ($\\rho$=0.1)", capsize=2)
    axes[0].bar(x + 0.2, ms, 0.38, yerr=hs, color=C[1], label="synchronised ($\\rho$=0.5)", capsize=2)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("loss (min/traveller/day)")
    axes[0].set_title("(a) layer losses, two regimes", fontsize=9)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    for (name, rows), col, mk in zip(sweeps.items(), [C[2], C[3], C[6]], ["o", "s", "^"]):
        tv = [r["tv"] for r in rows]; lm = [r["Lambda"] for r in rows]; hh = [r["CI95"] for r in rows]
        axes[1].errorbar(tv, lm, yerr=hh, color=col, marker=mk, ms=3.5, capsize=2, label=name)
    axes[1].set_xlabel("layer deviation $\\delta_\\ell$ (TV)")
    axes[1].set_ylabel("$\\Lambda_\\ell$ (min)")
    axes[1].set_title("(b) stable regime: sweep", fontsize=9)
    axes[1].legend(fontsize=6.5, frameon=False)
    axes[2].plot(np.arange(60, 160), tr_ref[60:160], color=C[1], lw=1.1, label="idealised profile")
    axes[2].plot(np.arange(60, 160), tr_act[60:160], color=C[0], lw=1.1, label="actual profile")
    axes[2].set_xlabel("day"); axes[2].set_ylabel("mean generalised cost (min)")
    axes[2].set_title("(c) synchronised regime", fontsize=9)
    axes[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_exp2.pdf")); plt.close(fig)
    with open(os.path.join(OUT, "exp2_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "individual"}, indent=1))


if __name__ == "__main__":
    main()
