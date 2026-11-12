"""
AgMS reference implementation (Yu & Eslami, "Mathematical Foundations of Agentic
Mobility Systems: Human and Machine Agency, Autonomy, and Trustworthy
Delegation").

Parameter names follow the paper: q_I (interpretation drift), q_D (invalid deliberation
outputs), alpha (approval probability), T (sampling temperature), rho (re-planning
probability), L (memory window), disc (discount factor beta in the paper).

Minimal reference implementation of an Agentic Mobility System (AgMS) with
language-mediated surrogate agents, written to mirror the primitives of the paper:

  principal p   : true value of time v_p (true preference U_p)
  specification : phrase g_p in a finite goal language G = {fastest, balanced, cheapest}
  interpretation: Phi_i(g) -> operative objective theta (money-to-minutes weight)
  deliberation  : Pi_i(theta, memory) -> proposed route (softmax, inertia, invalid output w.p. q_D)
  actuation     : self-execution set E_i = {keep current route}; switches need human approval (alpha)
  shield        : sigma_i maps invalid / inadmissible proposals to an admissible default
  substrate     : static BPR network loading of route flows, one tick = one day
  memory        : sliding window (L days) of observed route travel times (forgetful)

Every stochastic layer draws from its own random stream so that counterfactual
replays (kernel substitution) use common random numbers.
"""
import numpy as np

PHRASES = ["fastest", "balanced", "cheapest"]
PHRASE_VOT = np.array([45.0, 20.0, 8.0])      # $/h implied by the faithful interpretation


class Net:
    def __init__(self, t0, cap, toll, restricted=None):
        self.t0 = np.asarray(t0, float); self.cap = np.asarray(cap, float)
        self.toll = np.asarray(toll, float); self.R = len(t0)
        self.restricted = restricted

    def times(self, flows):
        return self.t0 * (1 + 0.15 * (flows / self.cap) ** 4)


def make_population(N, seed=11, median_vot=20.0, sdlog=0.5):
    rng = np.random.default_rng(seed)
    v = median_vot * np.exp(sdlog * rng.standard_normal(N))
    g = np.where(v >= 30, 0, np.where(v >= 13, 1, 2))            # lossy specification
    return v, g


def _mask(x, N):
    return np.broadcast_to(np.asarray(x, bool), (N,)).copy()


def run(net, v, g, days=200, disc=0.95, L=5, T=1.0, rho=0.5,
        q_I=0.1, q_D=0.05, alpha=0.8, perfect_spec=False, faithful=False,
        no_halluc=False, full_exec=False, seed=0, record=False,
        permits=None, misaligned_share=0.0, shield="validity",
        humans=0, kappa=0.0, persuade=False):
    """Layer-reference flags (perfect_spec, faithful, no_halluc, full_exec) may be bools or
    per-agent boolean masks (kernel substitution for a subset J of agents).
    Returns discounted average daily generalized cost (true VOT), per-agent values, diagnostics."""
    N = len(v)
    perfect_spec = _mask(perfect_spec, N); faithful = _mask(faithful, N)
    no_halluc = _mask(no_halluc, N); full_exec = _mask(full_exec, N)
    qI = np.where(faithful, 0.0, q_I); qD = np.where(no_halluc, 0.0, q_D)
    ss = np.random.SeedSequence(seed)
    r_init, r_int, r_hal, r_cho, r_ine, r_app, r_per, r_hum = [np.random.default_rng(s) for s in ss.spawn(8)]
    R = net.R
    route = r_init.integers(0, R, N)
    mem = np.tile(net.t0[None, :, None], (N, 1, L)).astype(float)   # memory window
    ptr = 0
    # specification -> operative objective (money weight in minutes per $)
    theta_spec = np.where(perfect_spec, 60.0 / v, 60.0 / PHRASE_VOT[g])
    acc = np.zeros(N)
    mis = np.zeros(N, bool)
    if misaligned_share > 0:
        mis[: int(round(misaligned_share * N))] = True
    # unmediated human drivers (habitual, follow agent advice with prob kappa if persuaded)
    h_route = r_init.integers(0, R - 1, humans) if humans else np.zeros(0, int)
    if humans and net.restricted is not None and net.restricted != R - 1:
        raise ValueError("convention: restricted route is the last route")
    total, wsum = 0.0, 0.0
    viol_days, excess = 0, []
    ev = {"drift": 0, "drift_exec": 0, "drift_selfexec": 0, "decisions": 0}
    daily_cost, route_times = [], []
    for d in range(days):
        # ---- cognition (Phi then Pi) ----
        theta = theta_spec.copy()
        u_int = r_int.random(N); alt = r_int.integers(1, 3, N)
        drift = u_int < qI
        theta = np.where(drift, 60.0 / PHRASE_VOT[(g + alt) % 3], theta)
        est = mem.mean(axis=2)                                         # N x R
        score = -(est + theta[:, None] * net.toll[None, :])
        if misaligned_share > 0 and net.restricted is not None:       # misaligned objective
            score[mis] = -est[mis] * 0.0; score[mis, net.restricted] = 50.0
        z = score / T; z -= z.max(axis=1, keepdims=True); pr = np.exp(z); pr /= pr.sum(axis=1, keepdims=True)
        u_cho = r_cho.random(N)
        prop = (pr.cumsum(axis=1) < u_cho[:, None]).sum(axis=1)
        prop = np.minimum(prop, R - 1)
        replan = r_ine.random(N) < rho
        prop = np.where(replan, prop, route)
        u_hal = r_hal.random(N)
        invalid = (u_hal < qD) & replan
        # ---- actuation: self-execution set = {keep current}; switches need approval ----
        u_app = r_app.random(N)
        needs_ok = (prop != route) & ~invalid
        approved = full_exec | (u_app < alpha)
        proposal_exec = np.where(needs_ok, approved, True)
        dec = np.where(invalid, -1, np.where(proposal_exec, prop, route))
        # ---- shield ----
        newroute = np.where(dec < 0, route, dec)                       # validity shield
        if shield == "quota" and net.restricted is not None:
            if d == 0:
                holder = np.zeros(N, bool)
                holder[r_per.permutation(N)[:permits]] = True           # product-form admissible sets (fixed permits)
            bad = (newroute == net.restricted) & ~holder
            sc = score.copy(); sc[:, net.restricted] = -np.inf
            default = sc.argmax(axis=1)                         # minimally invasive: best admissible route
            newroute = np.where(bad, default, newroute)
        if record:
            ev["decisions"] += int(replan.sum())
            dr = drift & replan
            ev["drift"] += int(dr.sum())
            ev["drift_valid"] = ev.get("drift_valid", 0) + int((dr & ~invalid).sum())
            ev["drift_exec"] += int((dr & ~invalid & proposal_exec).sum())
            ev["drift_selfexec"] += int((dr & ~invalid & ~needs_ok).sum())
        route = newroute
        # ---- unmediated humans ----
        if humans:
            follow = r_hum.random(humans) < (kappa if persuade else 0.0)
            hr = np.where(follow, net.restricted if net.restricted is not None else 0, h_route)
            hflow = np.bincount(hr, minlength=R)
        else:
            hflow = np.zeros(R)
        # ---- substrate ----
        flows = np.bincount(route, minlength=R) + hflow
        tt = net.times(flows)
        gc = tt[route] + 60.0 * net.toll[route] / v
        w = disc ** d
        total += w * gc.mean(); wsum += w; acc += w * gc
        daily_cost.append(gc.mean()); route_times.append(tt)
        if net.restricted is not None and permits is not None:
            ex = flows[net.restricted] - permits
            if d >= 20:
                viol_days += int(ex > 0); excess.append(max(ex, 0))
        # ---- memory update (window of observed route times; travel-information tool) ----
        mem[:, :, ptr % L] = tt[None, :]; ptr += 1
        if humans:
            hc = tt + 60 * net.toll / 20.0
            if net.restricted is not None:
                hc = hc.copy(); hc[net.restricted] = np.inf        # compliant habit: never the restricted link
            h_route = np.where(r_hum.random(humans) < 0.2, int(np.argmin(hc)), h_route)
    out = {"W": total / wsum, "Wi": acc / wsum, "daily_cost": np.array(daily_cost),
           "route_times": np.array(route_times)}
    if record:
        out["events"] = ev
    if permits is not None:
        out["viol_rate"] = viol_days / max(1, days - 20); out["mean_excess"] = float(np.mean(excess)) if excess else 0.0
    return out
