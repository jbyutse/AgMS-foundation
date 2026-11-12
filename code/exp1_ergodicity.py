"""
Experiment 1 -- Well-posedness and long-run behaviour (Theorem 3, Example 2).

Part A: exact lifted Markov chain for a symmetric two-route network with N identical
        language-model-like (softmax) agents that observe yesterday's route costs
        (memory window L = 1) and re-plan with probability rho.
        n' = n - A + B,  A ~ Bin(n, rho (1 - p(n))),  B ~ Bin(N - n, rho p(n)).
        Outputs: sample paths, the invariant law across temperatures (synchronisation
        transition), and a mixing-time phase diagram over (temperature, inertia).
Part B: agent-level simulation with private, own-experience memories:
        forgetful sliding window (with a fixed fallback belief) vs non-forgetful
        cumulative memory, from two initial conditions, across temperatures.
        Outputs: trajectories with between-run bands, the belief mechanism behind
        lock-in, and a lock-in diagram (late-period share against temperature).

Set AGMS_USETEX=1 to render text with LaTeX (as in the paper); the default uses
matplotlib's Computer Modern mathtext and needs no LaTeX installation.
"""
import json, os
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

USETEX = os.environ.get("AGMS_USETEX", "0") == "1"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "mathtext.fontset": "cm",
})
if USETEX:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})

INK, MUTED = "#1f2328", "#5b6470"
BLUE, ORANGE, GREEN = "#1c5cab", "#c4501d", "#2f8f5b"
CM_PI = LinearSegmentedColormap.from_list("pi", ["#ffffff", "#dae7f7", "#8fb4e6", "#3a72c4", "#123d78"])
CM_MIX = LinearSegmentedColormap.from_list("mix", ["#fdf6f0", "#f6d2b8", "#e59a6c", "#c4501d", "#6e2a0c"])

N = 100
A0, B0 = 10.0, 20.0           # c_r(f) = A0 + B0 * f / N   (minutes)


def panel_title(ax, letter, text):
    ax.set_title(rf"$\bf{{({letter})}}$ {text}" if not USETEX else rf"\textbf{{({letter})}} {text}",
                 fontsize=8.2, loc="left", color=INK, pad=4)


def costs(n):
    return A0 + B0 * n / N, A0 + B0 * (N - n) / N


def p_route1(n, T):
    c1, c2 = costs(n)
    if T == 0:
        if c1 < c2:
            return 1.0
        if c1 > c2:
            return 0.0
        return None                       # tie -> keep current route
    z = np.clip((c1 - c2) / T, -700, 700)
    return 1.0 / (1.0 + np.exp(z))


def transition_matrix(T, rho):
    P = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        p = p_route1(n, T)
        if p is None:
            P[n, n] = 1.0
            continue
        a = binom.pmf(np.arange(n + 1), n, rho * (1 - p))            # leaving route 1
        b = binom.pmf(np.arange(N - n + 1), N - n, rho * p)          # joining route 1
        conv = np.convolve(a[::-1], b)                               # index k -> n' = k
        P[n, :len(conv)] = conv[:N + 1]
    return P


def stationary(P):
    w, v = np.linalg.eig(P.T)
    k = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, k]); pi = np.abs(pi) / np.abs(pi).sum()
    return pi


def lazy_chain_check(P, pi, iters=20000):
    """Cross-check of the eigenvector solution: iterate the lazy chain (P + I) / 2 from the uniform law."""
    lazy = (P + np.eye(len(P))) / 2
    v = np.full(len(P), 1.0 / len(P))
    for _ in range(iters):
        v = v @ lazy
    return float(np.abs(v - pi).max())


def dobrushin(Pk):
    # delta(P) = 1 - min_{z,z'} sum_y min(P(z,y), P(z',y))  (computed exactly)
    m = np.inf
    for i in range(Pk.shape[0]):
        ov = np.minimum(Pk[i][None, :], Pk).sum(axis=1)
        m = min(m, ov.min())
    return 1 - m


def tv_curve(P, pi, n0=0, days=200):
    mu = np.zeros(N + 1); mu[n0] = 1.0
    out = []
    for _ in range(days):
        out.append(0.5 * np.abs(mu - pi).sum())
        mu = mu @ P
    return np.array(out)


def mixing_time(P, pi, eps=0.01, cap=1000, n0=0):
    mu = np.zeros(N + 1); mu[n0] = 1.0
    for t in range(cap + 1):
        if 0.5 * np.abs(mu - pi).sum() < eps:
            return t
        mu = mu @ P
    return np.inf


def worst_mixing_time(P, pi, eps=0.01, cap=1024):
    """Smallest n with max_z TV(P^n(z,.), pi) < eps (worst initial state); inf if above cap."""
    pows = [P]
    while 2 ** len(pows) <= cap:
        pows.append(pows[-1] @ pows[-1])

    def worst_tv(n):
        M = np.eye(P.shape[0]); k = 0
        while n:
            if n & 1:
                M = M @ pows[k]
            n >>= 1; k += 1
        return 0.5 * np.abs(M - pi[None, :]).sum(axis=1).max()

    if worst_tv(cap) >= eps:
        return np.inf
    lo, hi = 0, cap
    while lo < hi:                      # worst-case TV is non-increasing in n
        mid = (lo + hi) // 2
        if worst_tv(mid) < eps:
            hi = mid
        else:
            lo = mid + 1
    return lo


def sample_path(T, rho, n0, days, rng):
    traj = [n0]
    for _ in range(days):
        n = traj[-1]
        p = p_route1(n, T)
        if p is None:
            traj.append(n); continue
        A = rng.binomial(n, rho * (1 - p)); B = rng.binomial(N - n, rho * p)
        traj.append(n - A + B)
    return np.array(traj) / N


def part_a():
    res = {}
    # ---- numbers reported in the text (unchanged definitions) ----
    tv_tab = {}
    for (T, rho) in [(2.0, 1.0), (0.5, 1.0), (2.0, 0.3), (0.5, 0.3)]:
        P = transition_matrix(T, rho); pi = stationary(P)
        tv = tv_curve(P, pi)
        t01 = int(np.argmax(tv < 0.01)) if (tv < 0.01).any() else None
        d2 = dobrushin(np.linalg.matrix_power(P, 2))
        s = np.arange(N + 1) / N
        tv_tab[f"T={T},rho={rho}"] = {"days_to_TV<0.01": t01, "TV_day50": float(tv[50]),
                                      "TV_day199": float(tv[-1]), "dobrushin_P2": float(d2),
                                      "pi_mean_share": float(s @ pi),
                                      "pi_sd_share": float(np.sqrt((s ** 2) @ pi - (s @ pi) ** 2)),
                                      "pi_check_lazy_chain_max_abs_diff": lazy_chain_check(P, pi)}
    res["tv"] = tv_tab
    res["deterministic_attractors"] = "fixed point at 50 and 2-cycle {0,100}"

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.19, top=0.87, wspace=0.62)
    s = np.arange(N + 1) / N

    # (a) one-day response map: conditional mean and 5-95% band of tomorrow's share, with cobwebs
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
    rng = np.random.default_rng(2026)
    for rho, col, lab in [(1.0, ORANGE, r"$\rho=1$"), (0.3, BLUE, r"$\rho=0.3$")]:
        P = transition_matrix(0.5, rho)
        mean = P @ s
        cdf = np.cumsum(P, axis=1)
        lo = s[np.argmax(cdf >= 0.05, axis=1)]; hi = s[np.argmax(cdf >= 0.95, axis=1)]
        ax.fill_between(s, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(s, mean, color=col, lw=1.1, label=lab)
        path = sample_path(0.5, rho, 0, 14, rng)
        xs, ys = [path[0]], [path[0]]
        for u, v in zip(path[:-1], path[1:]):
            xs += [u, v]; ys += [v, v]
        ax.plot(xs[1:], ys[1:], color=col, lw=0.5, alpha=0.9)
    ax.plot([0.5], [0.5], marker="o", ms=3.2, mfc="white", mec=INK, mew=0.7, zorder=5)
    ax.annotate("equilibrium", (0.5, 0.5), xytext=(0.56, 0.28), fontsize=6.0, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"share today"); ax.set_ylabel(r"share tomorrow")
    ax.legend(fontsize=6.0, loc="upper right", handlelength=1.4, borderaxespad=0.2, labelspacing=0.2)
    panel_title(ax, "a", r"response map, $T=0.5$")

    # (b) invariant law across temperature: synchronisation transition (rho = 1) vs inertia (rho = 0.3)
    ax = axes[1]
    Ts = np.logspace(-1, 2, 70)
    img = np.zeros((N + 1, len(Ts)))
    q03 = np.zeros((3, len(Ts)))
    for j, T in enumerate(Ts):
        img[:, j] = stationary(transition_matrix(T, 1.0))
        pi03 = stationary(transition_matrix(T, 0.3)); cdf = np.cumsum(pi03)
        q03[:, j] = [s[np.searchsorted(cdf, q)] for q in (0.05, 0.5, 0.95)]
    logT = np.log10(Ts)
    im = ax.imshow(np.maximum(img, 1e-4), origin="lower", aspect="auto", cmap=CM_PI,
                   norm=LogNorm(vmin=1e-4, vmax=1.0),
                   extent=(logT[0], logT[-1], -0.03, 1.03), interpolation="nearest")
    ax.plot(logT, q03[0], color=ORANGE, lw=0.9, ls=(0, (3, 1.5)))
    ax.plot(logT, q03[2], color=ORANGE, lw=0.9, ls=(0, (3, 1.5)),
            label=r"$\rho=0.3$ (5--95\%)" if USETEX else r"$\rho=0.3$ (5–95%)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([-1, 0, 1, 2]); ax.set_xticklabels([r"$0.1$", r"$1$", r"$10$", r"$100$"])
    ax.set_xlabel(r"temperature $T$ (min)")
    ax.set_ylabel("share on route 1")
    ax.legend(fontsize=5.8, loc="center left", handlelength=1.8, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.82))
    cb = fig.colorbar(im, ax=ax, pad=0.04, fraction=0.07)
    cb.ax.tick_params(labelsize=5.6, width=0.4, length=1.5); cb.outline.set_linewidth(0.4)
    cb.set_label(r"$\pi^*$, $\rho=1$", fontsize=6.2, labelpad=1)
    panel_title(ax, "b", r"invariant law")

    # (c) mixing-time phase diagram over (T, rho), worst initial state
    ax = axes[2]
    Tg = np.logspace(-1, 1, 25); Rg = np.linspace(0.05, 1.0, 20)
    cap = 1024
    M = np.zeros((len(Rg), len(Tg)))
    for i, rho in enumerate(Rg):
        for j, T in enumerate(Tg):
            P = transition_matrix(T, rho); pi = stationary(P)
            M[i, j] = worst_mixing_time(P, pi, cap=cap)
    Mplot = np.where(np.isinf(M), cap * 1.5, np.maximum(M, 1))
    im2 = ax.pcolormesh(np.log10(Tg), Rg, Mplot, cmap=CM_MIX, norm=LogNorm(vmin=1, vmax=cap * 1.5), shading="nearest")
    cs = ax.contour(np.log10(Tg), Rg, Mplot, levels=[10, 100], colors=[INK], linewidths=0.5)
    ax.clabel(cs, fmt={10: r"10", 100: r"100"}, fontsize=5.6, inline=True, inline_spacing=1)
    if np.isinf(M).any():
        ax.contourf(np.log10(Tg), Rg, np.isinf(M).astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
        plt.rcParams["hatch.linewidth"] = 0.4
    marks = {}
    for (T, rho) in [(2.0, 0.3), (0.5, 0.3)]:
        P = transition_matrix(T, rho); pi = stationary(P)
        tm = worst_mixing_time(P, pi, cap=cap); marks[f"T={T},rho={rho}"] = tm
        ax.plot(np.log10(T), rho, marker="o", ms=3.2, mfc="white", mec=INK, mew=0.7, zorder=5)
        ax.annotate(f"{int(tm)} d", (np.log10(T), rho), xytext=(4, -9), textcoords="offset points",
                    fontsize=6.0, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    ax.text(np.log10(0.13), 0.93, r"no mixing within" + "\n" + r"1024 days", fontsize=5.8, color="white",
            va="top", ha="left", zorder=6)
    ax.set_xticks([-1, 0, 1]); ax.set_xticklabels([r"$0.1$", r"$1$", r"$10$"])
    ax.set_xlabel(r"temperature $T$ (min)"); ax.set_ylabel(r"re-planning probability $\rho$")
    cb2 = fig.colorbar(im2, ax=ax, pad=0.04, fraction=0.07)
    cb2.ax.tick_params(labelsize=5.6, width=0.4, length=1.5); cb2.outline.set_linewidth(0.4)
    cb2.set_label(r"mixing time (days)", fontsize=6.2, labelpad=1)
    panel_title(ax, "c", r"mixing time")
    fig.savefig(os.path.join(OUT, "fig_exp1a.pdf"))
    plt.close(fig)
    res["worst_case_mixing_marked"] = {k: (None if np.isinf(v) else int(v)) for k, v in marks.items()}
    res["mixing_grid"] = {"T": Tg.tolist(), "rho": Rg.tolist(),
                          "worst-case days to TV<0.01 (None = not within 1024)": [[None if np.isinf(v) else int(v) for v in row] for row in M]}
    trans = [float(Ts[j]) for j in range(len(Ts)) if img[N // 2 - 5:N // 2 + 6, j].sum() > img[[0, N], j].sum()]
    res["rho1_unimodal_from_T"] = trans[0] if trans else None
    return res


# ---------------- Part B: agent-level, private memories ----------------
def simulate_private(T, memory, ic, days=400, rho=0.3, L=5, seed=0, prior_w=5, track=False):
    rng = np.random.default_rng(seed)
    ff = np.array([A0, A0])
    if ic == "A":          # everyone on route 1, pessimistic belief about route 2
        route = np.zeros(N, dtype=int)
        init_belief = np.array([A0, 40.0])
    else:                  # random split, neutral beliefs
        route = rng.integers(0, 2, N)
        init_belief = ff.copy()
    S = np.tile(init_belief * prior_w, (N, 1)); Wt = np.full((N, 2), float(prior_w))
    win = np.full((N, 2, L), np.nan); win[:, :, :] = init_belief[None, :, None]
    ptr = np.zeros((N, 2), dtype=int)
    share, belief2, cost2 = [], [], []
    for d in range(days):
        n1 = int((route == 0).sum())
        c = np.array(costs(n1))
        share.append(n1 / N)
        idx = np.arange(N)
        if memory == "cumulative":
            S[idx, route] += c[route]; Wt[idx, route] += 1
            est = S / Wt
        else:
            win[idx, route, ptr[idx, route]] = c[route]
            ptr[idx, route] = (ptr[idx, route] + 1) % L
            est = np.nanmean(win, axis=2)
        if memory == "window":
            other = 1 - route
            win[idx, other, ptr[idx, other]] = ff[other]
            ptr[idx, other] = (ptr[idx, other] + 1) % L
            est = np.nanmean(win, axis=2)
        if track:
            belief2.append(float(est[:, 1].mean())); cost2.append(float(c[1]))
        replan = rng.random(N) < rho
        z = np.clip((est[:, 0] - est[:, 1]) / T, -700, 700)
        p1 = 1.0 / (1.0 + np.exp(z))
        newr = np.where(rng.random(N) < p1, 0, 1)
        route = np.where(replan, newr, route)
    if track:
        return np.array(share), np.array(belief2), np.array(cost2)
    return np.array(share)


def part_b():
    res = {}
    R = 20
    seeds = [100 * r + 7 for r in range(R)]
    # numbers reported in the text (same seeds as before)
    for memory in ("window", "cumulative"):
        for T in (2.0, 1.0, 0.5):
            for ic in ("A", "B"):
                runs = np.array([simulate_private(T, memory, ic, seed=sd) for sd in seeds])
                late = runs[:, 200:].mean(axis=1)
                res[f"{memory}|T={T}|IC={ic}"] = {"late_mean_share": float(late.mean()),
                                                  "late_sd_across_runs": float(late.std(ddof=1)),
                                                  "min": float(late.min()), "max": float(late.max())}

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.25))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.86, wspace=0.36)
    Tshow = 1.0
    days = np.arange(400)
    # (a) trajectories with between-run bands
    ax = axes[0]
    traj = {}
    for memory, col, lab in [("window", BLUE, "forgetful"), ("cumulative", ORANGE, "non-forgetful")]:
        outs = [simulate_private(Tshow, memory, "A", seed=sd, track=True) for sd in seeds]
        sh = np.array([o[0] for o in outs]); b2 = np.array([o[1] for o in outs]); c2 = np.array([o[2] for o in outs])
        traj[memory] = (sh, b2, c2)
        lo, med, hi = np.percentile(sh, [10, 50, 90], axis=0)
        ax.fill_between(days, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(days, med, color=col, lw=1.1, label=f"{lab}, IC-A")
        shB = np.array([simulate_private(Tshow, memory, "B", seed=sd) for sd in seeds])
        ax.plot(days, np.median(shB, axis=0), color=col, lw=0.7, ls=(0, (1.5, 1.5)), label=f"{lab}, IC-B")
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_ylim(0.3, 1.05); ax.set_xlim(0, 399)
    ax.set_xlabel("day"); ax.set_ylabel("share on route 1")
    ax.legend(fontsize=5.8, loc="center right", handlelength=1.6, labelspacing=0.2, bbox_to_anchor=(1.0, 0.70))
    panel_title(ax, "a", rf"share on route 1, $T={Tshow:g}$")
    # (b) the mechanism: stale beliefs about route 2 are self-confirming without forgetting
    ax = axes[1]
    for memory, col, lab in [("window", BLUE, "forgetful"), ("cumulative", ORANGE, "non-forgetful")]:
        sh, b2, c2 = traj[memory]
        ax.plot(days, b2.mean(axis=0), color=col, lw=1.1, label=f"belief, {lab}")
        ax.plot(days, c2.mean(axis=0), color=col, lw=0.7, ls=(0, (1.5, 1.5)), label=f"realised, {lab}")
    ax.set_xlim(0, 399); ax.set_ylim(5, 44)
    ax.set_xlabel("day"); ax.set_ylabel("route-2 travel time (min)")
    ax.legend(fontsize=5.8, loc="center right", handlelength=1.6, labelspacing=0.2, bbox_to_anchor=(1.0, 0.62))
    panel_title(ax, "b", "beliefs about route 2")
    res["mechanism_T1"] = {m: {"belief_route2_day399": float(traj[m][1][:, -1].mean()),
                               "realised_route2_day399": float(traj[m][2][:, -1].mean())} for m in traj}
    # (c) lock-in diagram: late share against temperature, both memories and initial conditions
    ax = axes[2]
    Tgrid = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
    lock = {}
    for memory, col, mk in [("window", BLUE, "o"), ("cumulative", ORANGE, "s")]:
        curves = {}
        for ic, fill in [("A", True), ("B", False)]:
            m, sd = [], []
            for T in Tgrid:
                runs = np.array([simulate_private(T, memory, ic, seed=s_)[200:].mean() for s_ in seeds])
                m.append(runs.mean()); sd.append(runs.std(ddof=1))
            m, sd = np.array(m), np.array(sd)
            curves[ic] = m
            ax.errorbar(Tgrid, m, yerr=sd, color=col, lw=0.9, marker=mk, ms=3.0,
                        mfc=col if fill else "white", mec=col, mew=0.8, capsize=1.5, elinewidth=0.6,
                        ls="-" if fill else (0, (1.5, 1.5)),
                        label=f"{'forgetful' if memory == 'window' else 'non-forgetful'}, IC-{ic}")
        if memory == "cumulative":
            ax.fill_between(Tgrid, curves["B"], curves["A"], color=ORANGE, alpha=0.12, lw=0)
            ax.annotate("lock-in gap (A4)", xy=(0.27, 0.92), fontsize=6.2, color=ORANGE, ha="left")
        lock[memory] = {"T": Tgrid, "IC-A": curves["A"].tolist(), "IC-B": curves["B"].tolist()}
    ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_ylim(0.42, 1.05)
    ax.set_xlabel(r"temperature $T$ (min)"); ax.set_ylabel("late-period share (days 200--399)" if USETEX else "late-period share (days 200–399)")
    ax.legend(fontsize=5.6, loc="center left", handlelength=1.8, labelspacing=0.2, bbox_to_anchor=(0.0, 0.5),
              frameon=True, framealpha=0.85, edgecolor="none")
    panel_title(ax, "c", "lock-in diagram")
    res["lockin_diagram"] = lock
    fig.savefig(os.path.join(OUT, "fig_exp1b.pdf"))
    plt.close(fig)
    return res


if __name__ == "__main__":
    out = {"part_a": part_a(), "part_b": part_b()}
    with open(os.path.join(OUT, "exp1_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: (v if k != "part_a" else {kk: vv for kk, vv in v.items() if kk != "mixing_grid"})
                      for k, v in out.items()}, indent=1)[:3000])
