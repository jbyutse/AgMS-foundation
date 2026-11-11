"""
Experiment 1 -- Well-posedness and long-run behaviour (Theorem 2, Example 1).

Part A: exact lifted Markov chain for a symmetric two-route network with N identical
        language-model-like (softmax) agents that observe yesterday's route costs
        (memory window L = 1) and re-plan with probability rho.
        n' = n - A + B,  A ~ Bin(n, rho (1 - p(n))),  B ~ Bin(N - n, rho p(n)).
Part B: agent-level simulation with private, own-experience memories:
        forgetful sliding window (with a fixed fallback belief) vs non-forgetful
        cumulative memory, from two initial conditions, several temperatures.
"""
import json, os
import numpy as np
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.6,
                     "font.family": "serif"})

N = 100
A0, B0 = 10.0, 20.0           # c_r(f) = A0 + B0 * f / N   (minutes)


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
        # a[::-1][j] = P(A = n - j); n' = n - A + B = j + B -> conv index j + B
        P[n, :len(conv)] = conv[:N + 1]
    return P


def stationary(P):
    w, v = np.linalg.eig(P.T)
    k = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, k]); pi = np.abs(pi) / np.abs(pi).sum()
    return pi


def dobrushin(Pk):
    # delta(P) = 1 - min_{z,z'} sum_y min(P(z,y), P(z',y))  (computed exactly)
    m = np.inf
    for i in range(Pk.shape[0]):
        ov = np.minimum(Pk[i][None, :], Pk).sum(axis=1)
        m = min(m, ov.min())
    return 1 - m


def part_a():
    res = {}
    days = 200
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    # (i) deterministic T = 0, rho = 1: two attractors
    P0 = transition_matrix(0, 1.0)
    for n0, col, lab in [(50, C[0], "start at 50/50"), (51, C[1], "start at 51/49"), (100, C[2], "start at 100/0")]:
        traj = [n0]
        for _ in range(12):
            traj.append(int(np.argmax(P0[traj[-1]])))
        axes[0].plot(range(len(traj)), np.array(traj) / N, marker="o", ms=3, color=col, label=lab)
    axes[0].set_title("(a) $T=0$, $\\rho=1$: deterministic", fontsize=9)
    axes[0].set_xlabel("day"); axes[0].set_ylabel("share on route 1")
    axes[0].set_ylim(-0.05, 1.42)
    axes[0].legend(fontsize=6.3, frameon=False, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.2)
    res["deterministic_attractors"] = "fixed point at 50 and 2-cycle {0,100}"

    # (ii) TV distance to stationarity from n0 = 0
    tv_tab = {}
    for (T, rho), col, ls in [((2.0, 1.0), C[0], ":"), ((0.5, 1.0), C[1], "-"),
                              ((2.0, 0.3), C[0], "--"), ((0.5, 0.3), C[1], "--")]:
        P = transition_matrix(T, rho)
        pi = stationary(P)
        mu = np.zeros(N + 1); mu[0] = 1.0
        tv = []
        for t in range(days):
            tv.append(0.5 * np.abs(mu - pi).sum())
            mu = mu @ P
        tv = np.array(tv)
        t01 = int(np.argmax(tv < 0.01)) if (tv < 0.01).any() else None
        # Dobrushin coefficient of P^k for k = 2
        d2 = dobrushin(np.linalg.matrix_power(P, 2))
        tv_tab[f"T={T},rho={rho}"] = {"days_to_TV<0.01": t01, "TV_day50": float(tv[50]),
                                       "TV_day199": float(tv[-1]), "dobrushin_P2": float(d2),
                                       "pi_mean_share": float((np.arange(N + 1) / N) @ pi),
                                       "pi_sd_share": float(np.sqrt(((np.arange(N + 1) / N) ** 2) @ pi - ((np.arange(N + 1) / N) @ pi) ** 2))}
        axes[1].semilogy(tv, color=col, ls=ls, label=f"$T$={T}, $\\rho$={rho}")
    axes[1].set_title("(b) TV distance to $\\pi^*$ from $n_0=0$", fontsize=9)
    axes[1].annotate("$\\rho=1$: TV stays at 0.5\n(metastable flip-flop)", xy=(100, 0.5), xytext=(60, 2e-2), fontsize=6.5,
                     arrowprops=dict(arrowstyle="->", lw=0.6))
    axes[1].set_xlabel("day"); axes[1].set_ylim(1e-6, 1.5)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower left")
    res["tv"] = tv_tab

    # (iii) stationary laws
    for (T, rho), col, ls in [((2.0, 0.3), C[0], "--"), ((0.5, 0.3), C[1], "--"), ((0.5, 1.0), C[1], "-")]:
        pi = stationary(transition_matrix(T, rho))
        axes[2].plot(np.arange(N + 1) / N, pi, color=col, ls=ls, label=f"$T$={T}, $\\rho$={rho}")
    axes[2].set_title("(c) stationary law $\\pi^*$", fontsize=9)
    axes[2].set_xlabel("share on route 1"); axes[2].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_exp1a.pdf")); plt.close(fig)
    return res


# ---------------- Part B: agent-level, private memories ----------------
def simulate_private(T, memory, ic, days=400, rho=0.3, L=5, seed=0, prior_w=5):
    rng = np.random.default_rng(seed)
    ff = np.array([A0, A0])
    if ic == "A":          # everyone on route 1, pessimistic belief about route 2
        route = np.zeros(N, dtype=int)
        init_belief = np.array([A0, 40.0])
    else:                  # random split, neutral beliefs
        route = rng.integers(0, 2, N)
        init_belief = ff.copy()
    # cumulative memory: running sums with prior pseudo-observations
    S = np.tile(init_belief * prior_w, (N, 1)); Wt = np.full((N, 2), float(prior_w))
    # window memory: last L own observations per route (nan = empty), initialised with beliefs
    win = np.full((N, 2, L), np.nan); win[:, :, :] = init_belief[None, :, None]
    ptr = np.zeros((N, 2), dtype=int)
    share = []
    for d in range(days):
        n1 = int((route == 0).sum())
        c = np.array(costs(n1))
        share.append(n1 / N)
        # each agent observes only its own route's realised cost
        idx = np.arange(N)
        if memory == "cumulative":
            S[idx, route] += c[route]; Wt[idx, route] += 1
            est = S / Wt
        else:
            win[idx, route, ptr[idx, route]] = c[route]
            ptr[idx, route] = (ptr[idx, route] + 1) % L
            # entries older than L of the *other* route decay to the fixed reference belief
            # (forgetful): track age of the other route
            est = np.nanmean(win, axis=2)
        if memory == "window":
            # forget: an unobserved route's window is overwritten with the fixed reference
            # belief (free-flow) one slot per day -> after L days the initial belief is gone
            other = 1 - route
            win[idx, other, ptr[idx, other]] = ff[other]
            ptr[idx, other] = (ptr[idx, other] + 1) % L
            est = np.nanmean(win, axis=2)
        replan = rng.random(N) < rho
        z = np.clip((est[:, 0] - est[:, 1]) / T, -700, 700)
        p1 = 1.0 / (1.0 + np.exp(z))
        newr = np.where(rng.random(N) < p1, 0, 1)
        route = np.where(replan, newr, route)
    return np.array(share)


def part_b():
    res = {}
    R = 20
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, memory, title in [(axes[0], "window", "(a) forgetful memory (window $L=5$)"),
                              (axes[1], "cumulative", "(b) non-forgetful (cumulative) memory")]:
        for T, col in [(2.0, C[0]), (1.0, C[2]), (0.5, C[1])]:
            for ic, ls in [("A", "-"), ("B", ":")]:
                runs = np.array([simulate_private(T, memory, ic, seed=100 * r + 7) for r in range(R)])
                m = runs.mean(axis=0)
                ax.plot(m, color=col, ls=ls, lw=1.3, label=f"$T$={T}, IC-{ic}")
                late = runs[:, 200:].mean(axis=1)
                res[f"{memory}|T={T}|IC={ic}"] = {"late_mean_share": float(late.mean()),
                                                   "late_sd_across_runs": float(late.std(ddof=1)),
                                                   "min": float(late.min()), "max": float(late.max())}
        ax.set_title(title, fontsize=9); ax.set_xlabel("day")
    axes[0].set_ylabel("share on route 1 (mean of 20 runs)")
    axes[1].legend(fontsize=6.3, frameon=False, ncol=2, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_exp1b.pdf")); plt.close(fig)
    return res


if __name__ == "__main__":
    out = {"part_a": part_a(), "part_b": part_b()}
    with open(os.path.join(OUT, "exp1_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
