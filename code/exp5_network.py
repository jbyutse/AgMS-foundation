"""
Experiment 5 -- The delegation-loss decomposition (Theorem 5) and herding on a road network.

Network: Sioux Falls (24 nodes, 76 links, 528 OD pairs; Transportation Networks for Research,
files in data/siouxfalls). The population is scaled: one agent per 100 trips of the OD table
(3,606 agents) and link capacities divided by 100, as in sampled agent-based simulation.
Each agent chooses among the three shortest paths of its OD pair by free-flow time, so routes
of different OD pairs overlap on shared links. Tolls of $2 are placed on the ten links with the
highest volume-to-capacity ratio in the reference user-equilibrium solution.

The agents are the surrogate agents of Experiments 1-3 (agms_core): lossy phrase, interpretation
drift q_I, logit deliberation at temperature T over remembered route times from a shared
travel-information tool (window L), re-planning probability rho, invalid outputs q_D (current
route kept), and approval alpha of switches. Each layer draws from its own random stream, so the
layered systems M0..M4 of Theorem 5 are compared under common random numbers.

Outputs: results/exp5_results.json and results/fig_exp5.pdf.
Run `python exp5_network.py --plot-only` to redraw the figure from the saved JSON.
"""
import json, os, sys
from itertools import islice
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import agms_core as A

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "siouxfalls")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

SCALE = 100.0          # trips per agent
K = 3                  # routes per OD pair
TOLL, N_TOLLED = 2.0, 10
BASE = dict(q_I=0.10, q_D=0.05, alpha=0.8, T=1.0, rho=0.1, L=5, days=200, disc=0.95)
IDEAL = dict(perfect_spec=True, faithful=True, no_halluc=True, full_exec=True)
R = 20


def _rows(path, n):
    out = []
    for ln in open(path):
        t = ln.strip().rstrip(";").split()
        if len(t) >= n and t[0].replace(".", "").isdigit():
            out.append(t)
    return out


def load():
    L = np.array([[float(x) for x in r[:10]] for r in _rows(os.path.join(DATA, "SiouxFalls_net.tntp"), 10)])
    a, b = L[:, 0].astype(int), L[:, 1].astype(int)
    cap, fft, B, pw = L[:, 2] / SCALE, L[:, 4], L[:, 5], L[:, 6]
    # OD table
    od, cur = {}, None
    for ln in open(os.path.join(DATA, "SiouxFalls_trips.tntp")):
        s = ln.strip()
        if s.startswith("Origin"):
            cur = int(s.split()[1]); continue
        if cur is None or ":" not in s:
            continue
        for item in s.split(";"):
            if ":" in item:
                d, q = item.split(":"); d, q = int(d), float(q)
                if q > 0 and d != cur:
                    od[(cur, d)] = q
    # reference UE flows -> tolled links
    ue = {}
    for r in _rows(os.path.join(DATA, "SiouxFalls_flow.tntp"), 4):
        ue[(int(r[0]), int(r[1]))] = float(r[2])
    vc = np.array([ue[(i, j)] / (c * SCALE) for i, j, c in zip(a, b, cap)])
    toll = np.zeros(len(a)); toll[np.argsort(-vc)[:N_TOLLED]] = TOLL
    xy = {int(r[0]): (float(r[1]), float(r[2])) for r in _rows(os.path.join(DATA, "SiouxFalls_node.tntp"), 3)}
    return dict(a=a, b=b, cap=cap, fft=fft, B=B, pw=pw, toll=toll, od=od, xy=xy, vc_ue=vc)


def build(net):
    G = nx.DiGraph()
    idx = {}
    for k, (i, j) in enumerate(zip(net["a"], net["b"])):
        G.add_edge(int(i), int(j), w=float(net["fft"][k])); idx[(int(i), int(j))] = k
    routes, base, n_agents, od_list = [], [], [], []
    for (o, d), q in sorted(net["od"].items()):
        paths = list(islice(nx.shortest_simple_paths(G, o, d, weight="w"), K))
        assert len(paths) == K
        base.append(len(routes))
        for p in paths:
            routes.append([idx[(p[m], p[m + 1])] for m in range(len(p) - 1)])
        n_agents.append(int(round(q / SCALE))); od_list.append((o, d))
    Rt, Ln = len(routes), len(net["a"])
    Inc = np.zeros((Rt, Ln))
    for r, links in enumerate(routes):
        Inc[r, links] = 1.0
    base = np.array(base); n_agents = np.array(n_agents)
    agent_od = np.repeat(np.arange(len(base)), n_agents)
    return dict(Inc=Inc, base=base, agent_od=agent_od, od_list=od_list, routes=routes,
                rtoll=Inc @ net["toll"], rfft=Inc @ net["fft"])


def run(net, S, v, g, days=200, disc=0.95, L=5, T=1.0, rho=0.1, q_I=0.1, q_D=0.05, alpha=0.8,
        perfect_spec=False, faithful=False, no_halluc=False, full_exec=False, seed=0, keep_links=False,
        init=None, return_state=False):
    N = len(v)
    qI = 0.0 if faithful else q_I
    qD = 0.0 if no_halluc else q_D
    ss = np.random.SeedSequence(seed)
    r_init, r_int, r_hal, r_cho, r_ine, r_app, _, _ = [np.random.default_rng(s) for s in ss.spawn(8)]
    ridx = S["base"][S["agent_od"]][:, None] + np.arange(K)[None, :]          # N x K global route ids
    rtoll = S["rtoll"][ridx]
    loc = r_init.integers(0, K, N)
    Rt = S["Inc"].shape[0]
    hist = np.tile(S["rfft"][:, None], (1, L))                               # shared tool: route-time window
    if init is not None:                                                     # common warm start
        loc, hist = init[0].copy(), init[1].copy()
    ptr = 0
    theta_spec = 60.0 / v if perfect_spec else 60.0 / A.PHRASE_VOT[g]
    total = wsum = 0.0
    daily, links, switches = [], [], []
    for d in range(days):
        theta = np.full(N, theta_spec) if np.isscalar(theta_spec) else theta_spec.copy()
        u_int = r_int.random(N); alt = r_int.integers(1, 3, N)
        drift = u_int < qI
        theta = np.where(drift, 60.0 / A.PHRASE_VOT[(g + alt) % 3], theta)
        est = hist.mean(axis=1)[ridx]
        z = -(est + theta[:, None] * rtoll) / T
        z -= z.max(axis=1, keepdims=True); p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        prop = np.minimum((p.cumsum(axis=1) < r_cho.random(N)[:, None]).sum(axis=1), K - 1)
        replan = r_ine.random(N) < rho
        prop = np.where(replan, prop, loc)
        invalid = (r_hal.random(N) < qD) & replan
        u_app = r_app.random(N)
        needs_ok = (prop != loc) & ~invalid
        approved = full_exec | (u_app < alpha)
        newloc = np.where(invalid, loc, np.where(needs_ok & ~approved, loc, prop))
        switches.append(float((newloc != loc).mean()))
        loc = newloc
        rflow = np.bincount(ridx[np.arange(N), loc], minlength=Rt).astype(float)
        x = S["Inc"].T @ rflow
        t = net["fft"] * (1 + net["B"] * (x / net["cap"]) ** net["pw"])
        rt = S["Inc"] @ t
        gc = rt[ridx[np.arange(N), loc]] + 60.0 * rtoll[np.arange(N), loc] / v
        w = disc ** d; total += w * gc.mean(); wsum += w
        daily.append(float(gc.mean()))
        if keep_links:
            links.append(x.copy())
        hist[:, ptr % L] = rt; ptr += 1
    out = {"W": total / wsum, "daily": np.array(daily), "switch": np.array(switches)}
    if return_state:
        out["state"] = (loc.copy(), np.roll(hist, -ptr % L, axis=1).copy())
    if keep_links:
        out["links"] = np.array(links)
    return out


WARM_DAYS = 150


def warm_state(net, S, v, g, seed):
    """Common initial state: idealized agents in the stable regime after WARM_DAYS days."""
    return run(net, S, v, g, seed=10_000 + seed, **{**BASE, **IDEAL, "days": WARM_DAYS}, return_state=True)["state"]


def layered(net, S, v, g, seed, **kw):
    kw.setdefault("init", warm_state(net, S, v, g, seed))
    p = {**BASE, **kw}
    f = lambda **o: run(net, S, v, g, seed=seed, **{**p, **o})["W"]
    return np.array([f(**IDEAL), f(faithful=True, no_halluc=True, full_exec=True),
                     f(no_halluc=True, full_exec=True), f(full_exec=True), f()])


def ci(x):
    x = np.asarray(x); return x.mean(0), 1.96 * x.std(0, ddof=1) / np.sqrt(len(x))


def compute():
    net = load(); S = build(net)
    N = len(S["agent_od"])
    v, g = A.make_population(N, seed=11)
    res = {"agents": int(N), "od_pairs": len(S["base"]), "routes": int(S["Inc"].shape[0]),
           "links": int(len(net["a"])), "tolled_links": [[int(net["a"][k]), int(net["b"][k])] for k in np.flatnonzero(net["toll"])],
           "phrases": np.bincount(g, minlength=3).tolist()}
    ridx = S["base"][S["agent_od"]][:, None] + np.arange(K)[None, :]
    res["share_agents_toll_relevant"] = float((np.ptp(S["rtoll"][ridx], axis=1) > 0).mean())
    res["warm_days"] = WARM_DAYS
    shared = (S["Inc"].sum(0) > 0)
    res["links_used"] = int(shared.sum())
    res["routes_per_link_mean"] = float(S["Inc"].sum(0)[shared].mean())
    for name, rho in [("stable", 0.1), ("synchronized", 0.5)]:
        Vs = np.array([layered(net, S, v, g, s, rho=rho) for s in range(R)])
        lam = np.diff(Vs, axis=1); m, h = ci(lam); mt, ht = ci(Vs[:, 4] - Vs[:, 0])
        res[name] = {"V_mean": Vs.mean(0).tolist(), "Lambda_S_I_D_A": m.tolist(), "CI95": h.tolist(),
                     "total": float(mt), "total_CI95": float(ht),
                     "telescoping_max_abs_error": float(np.abs(lam.sum(1) - (Vs[:, 4] - Vs[:, 0])).max())}
        w3 = warm_state(net, S, v, g, 3)
        ref = run(net, S, v, g, seed=3, **{**BASE, "rho": rho, **IDEAL}, keep_links=True, init=w3)
        act = run(net, S, v, g, seed=3, **{**BASE, "rho": rho}, keep_links=True, init=w3)
        def cv(L):
            X = L[50:]; mu = X.mean(0); return np.where(mu > 0, X.std(0) / np.maximum(mu, 1e-9), 0.0)
        cr, ca = cv(ref["links"]), cv(act["links"])
        res[name]["daily_sd_after_day50"] = {"reference": float(ref["daily"][50:].std()), "actual": float(act["daily"][50:].std())}
        res[name]["daily_mean_after_day50"] = {"reference": float(ref["daily"][50:].mean()), "actual": float(act["daily"][50:].mean())}
        res[name]["link_flow_cv_mean"] = {"reference": float(cr[shared].mean()), "actual": float(ca[shared].mean())}
        res[name]["link_flow_cv_max"] = {"reference": float(cr.max()), "actual": float(ca.max())}
        res[name]["switch_share_after_day50"] = {"reference": float(ref["switch"][50:].mean()), "actual": float(act["switch"][50:].mean())}
        res[name]["trace_seed3"] = {"reference": ref["daily"].tolist(), "actual": act["daily"].tolist()}
        res[name]["link_cv_seed3"] = {"reference": cr.tolist(), "actual": ca.tolist()}
    res["network"] = {"a": net["a"].tolist(), "b": net["b"].tolist(),
                      "xy": {str(k): list(val) for k, val in net["xy"].items()}, "toll": net["toll"].tolist()}
    with open(os.path.join(OUT, "exp5_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def plot(res):
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "mathtext.fontset": "cm"})
    INK, MUTED = "#1f2328", "#5b6470"
    BLUE, ORANGE, GREEN, PURPLE = "#1c5cab", "#c4501d", "#2f8f5b", "#6a4c93"
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})
    # (a) layer losses in both regimes
    lay = ["$\\Lambda_S$", "$\\Lambda_I$", "$\\Lambda_D$", "$\\Lambda_A$", "$\\Lambda$"]
    xs = np.arange(5); wdt = 0.36
    for k, (name, col, lab) in enumerate([("stable", BLUE, "stable ($\\rho=0.1$)"), ("synchronized", ORANGE, "synchronized ($\\rho=0.5$)")]):
        m = res[name]["Lambda_S_I_D_A"] + [res[name]["total"]]; h = res[name]["CI95"] + [res[name]["total_CI95"]]
        ax[0].bar(xs + (k - 0.5) * wdt, m, wdt, yerr=h, color=col, alpha=0.85, label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    ax[0].axhline(0, color=INK, lw=0.5); ax[0].set_xticks(xs); ax[0].set_xticklabels(lay)
    ax[0].set_ylabel("min per traveler per day"); ax[0].legend(loc="lower left", fontsize=6.5)
    ax[0].set_title("(a) delegation loss by layer", fontsize=7.5, loc="left")
    # (b) daily cost, synchronized regime
    tr = res["synchronized"]["trace_seed3"]; days = np.arange(len(tr["reference"]))
    ax[1].plot(days, tr["reference"], color=ORANGE, lw=0.7, label="idealized agents")
    ax[1].plot(days, tr["actual"], color=INK, lw=0.7, label="actual agents")
    st = res["stable"]["trace_seed3"]["reference"]
    ax[1].plot(days, st, color=BLUE, lw=0.7, ls="--", label="idealized, stable regime")
    ax[1].set_xlabel("day"); ax[1].set_ylabel("mean generalized cost (min)")
    ax[1].legend(loc="upper right", fontsize=6.5); ax[1].set_title("(b) daily cost, synchronized regime", fontsize=7.5, loc="left")
    # (c) network map, link-flow coefficient of variation (idealized agents, synchronized regime)
    nw = res["network"]; xy = {int(k): v for k, v in nw["xy"].items()}
    cvv = np.array(res["synchronized"]["link_cv_seed3"]["reference"])
    vmax = max(cvv.max(), 1e-9)
    cmap = plt.get_cmap("Oranges")
    for k, (i, j) in enumerate(zip(nw["a"], nw["b"])):
        (x1, y1), (x2, y2) = xy[i], xy[j]
        ax[2].plot([x1, x2], [y1, y2], color=cmap(0.2 + 0.8 * cvv[k] / vmax), lw=0.6 + 2.4 * cvv[k] / vmax,
                   solid_capstyle="round", zorder=1)
        if nw["toll"][k] > 0:
            ax[2].plot([(x1 + x2) / 2], [(y1 + y2) / 2], marker="s", ms=2.2, color=BLUE, zorder=3)
    X = np.array(list(xy.values()))
    ax[2].scatter(X[:, 0], X[:, 1], s=4, color=INK, zorder=2)
    ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].set_aspect("equal", adjustable="datalim")
    for sp in ax[2].spines.values():
        sp.set_visible(False)
    ax[2].set_title("(c) day-to-day variability of link flows", fontsize=7.5, loc="left")
    ax[2].text(0.0, -0.06, "line width and shade: coefficient of variation\n(idealized agents, $\\rho=0.5$); squares: tolled links",
               transform=ax[2].transAxes, fontsize=6, color=MUTED, va="top")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig_exp5.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        res = json.load(open(os.path.join(OUT, "exp5_results.json")))
    else:
        res = compute()
    plot(res)
    for name in ["stable", "synchronized"]:
        r = res[name]
        print(name, "Lambda S,I,D,A =", np.round(r["Lambda_S_I_D_A"], 3), "+-", np.round(r["CI95"], 3),
              "total", round(r["total"], 3), "+-", round(r["total_CI95"], 3))
        print("   daily sd after day 50:", r["daily_sd_after_day50"], "link CV mean:", r["link_flow_cv_mean"])
