"""
Experiment 4 -- Language-model agents in the loop, and the fidelity of the surrogate agents
(Corollary 4, surrogate fidelity).

Every traveller's agent is a real language-model call. On each day on which an agent
re-plans, the model reads the traveller's instruction in plain words, picks an objective
profile (interpretation, Phi) and then a route (deliberation, Pi). Everything else is the
AgMS of Experiment 2: lossy specification, sliding-window memory, human approval of route
switches, validity shield, BPR network loading.

The script then measures, on the inputs the system actually visited,
  * the interpretation kernel of the model and its drift rate (q_I),
  * the rate of invalid outputs (q_D),
  * how the model's route choices compare with a softmax surrogate (fitted temperature T),
  * the total-variation distance between model and surrogate, from repeated calls,
  * system outcomes of the model-in-the-loop run against surrogate runs on the same seeds.

Usage
  python exp4_llm_agents.py --model MODEL_NAME     real run (key read from OPENAI_API_KEY)
  python exp4_llm_agents.py --list-models          show the model names your key can use
  python exp4_llm_agents.py --mock                 offline stand-in for the model (testing)
  python exp4_llm_agents.py --analyse-only         recompute the results from the saved log

The API key is read from the environment variable OPENAI_API_KEY and is never written to
any file. Every call is appended to a log (results/exp4_llm_log_<tag>.jsonl); if the run
stops, run the same command again and it resumes where it stopped. Language-model calls
are not reproducible, so the log is the record from which the results are computed.
"""
import argparse, hashlib, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import stats
import agms_core as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

PROFILES = ["A", "B", "C"]                       # time-first, balanced, cost-first
PROFILE_NAME = ["time-first", "balanced", "cost-first"]
THETA = 60.0 / A.PHRASE_VOT                      # minutes per dollar under each profile

# Goal language: free-text instructions. A principal whose value of time puts it in class k
# (the lossy specification of agms_core.make_population) issues one text of class k.
# Semantics [[g]]: the faithful interpretation of a text of class k is profile k.
GOAL_TEXTS = [
    ["Get me to work as fast as possible. I don't care about tolls.",
     "I'm always running late. Take the quickest way, whatever it costs.",
     "Time matters much more to me than money. Minimise my travel time.",
     "Fastest route, please. Tolls are fine.",
     "Every minute counts for me. Pay whatever toll saves time.",
     "Just make it quick."],
    ["Find a reasonable balance between travel time and tolls.",
     "I don't mind paying a small toll if it saves real time, but don't overspend.",
     "Balanced, please: not the slowest route, not the priciest.",
     "Save time when the toll is worth it; otherwise keep it cheap.",
     "I want a sensible trade-off between cost and time.",
     "Keep it moderate on both time and money."],
    ["Keep it as cheap as possible. I'm not in a hurry.",
     "Avoid tolls unless the detour is enormous.",
     "Money is tight this month. Cheapest way, please.",
     "I have plenty of time. Don't spend on tolls.",
     "Lowest cost matters most to me; a longer drive is fine.",
     "No rush. Spend as little as you can."],
]

SYSTEM = (
    "You are a personal travel agent. You act for one traveller and choose the route for "
    "their commute today.\n"
    "Step 1. Decide what the traveller wants from their instruction by picking exactly one "
    "objective profile:\n"
    "  A = time-first: the traveller values travel time at $45 per hour\n"
    "  B = balanced: the traveller values travel time at $20 per hour\n"
    "  C = cost-first: the traveller values travel time at $8 per hour\n"
    "Step 2. Choose the route that is best for the traveller under that profile, trading "
    "off expected travel time against the toll.\n"
    "Reply with exactly one line in the form  PROFILE=<A|B|C>; ROUTE=<1|2|3>  and nothing else."
)


def user_prompt(text, est, toll, cur, L):
    rows = "\n".join(f"  Route {r + 1}: {est[r]:.1f} min, toll ${toll[r]:.2f}" for r in range(len(est)))
    return (f"Traveller's instruction: \"{text}\"\n"
            f"Routes (expected travel time from the last {L} days; toll):\n{rows}\n"
            f"Current route: {cur + 1}")


def parse(reply, R):
    """Returns (profile index, route index) or None if the reply does not follow the format."""
    if not isinstance(reply, str):
        return None
    p = re.search(r"PROFILE\s*=\s*<?\s*([ABC])\b", reply, re.I)
    r = re.search(r"ROUTE\s*=\s*<?\s*(\d+)", reply, re.I)
    if not p or not r:
        return None
    route = int(r.group(1)) - 1
    if not 0 <= route < R:
        return None
    return PROFILES.index(p.group(1).upper()), route


# ----------------------------------------------------------------------------- model back ends
class OpenAIBackend:
    def __init__(self, model, api, effort, timeout):
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set.\n"
                     "  PowerShell:  $env:OPENAI_API_KEY = \"sk-...\"\n"
                     "  cmd:         set OPENAI_API_KEY=sk-...\n"
                     "  bash/zsh:    export OPENAI_API_KEY=sk-...\n"
                     "Then run the same command again in that window.")
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("The 'openai' package is missing. Install it with:  pip install openai")
        self.client = OpenAI(timeout=timeout)       # the key is taken from the environment
        self.model, self.api, self.effort = model, api, effort
        self.name = model

    def list_models(self):
        return sorted(m.id for m in self.client.models.list())

    def __call__(self, system, user, rng=None):
        if self.api == "responses":
            kw = dict(model=self.model, instructions=system, input=user)
            if self.effort:
                kw["reasoning"] = {"effort": self.effort}
            r = self.client.responses.create(**kw)
            text = r.output_text
            u = getattr(r, "usage", None)
            usage = (getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0)
        else:
            kw = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                  {"role": "user", "content": user}])
            if self.effort:
                kw["reasoning_effort"] = self.effort
            r = self.client.chat.completions.create(**kw)
            text = r.choices[0].message.content
            u = getattr(r, "usage", None)
            usage = (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        return text, usage


class MockBackend:
    """Offline stand-in with known behaviour, used to test the pipeline: it drifts to another
    profile with probability q_I, returns an unusable reply with probability q_D, and otherwise
    samples a route from a softmax at temperature T."""
    name = "mock (q_I=0.04, q_D=0.03, T=0.5)"
    q_I, q_D, T = 0.04, 0.03, 0.5

    def __call__(self, system, user, rng):
        text = re.search(r'instruction: "(.*)"', user).group(1)
        k = next(c for c in range(3) if text in GOAL_TEXTS[c])
        rows = re.findall(r"Route \d+: ([\d.]+) min, toll \$([\d.]+)", user)
        est = np.array([float(a) for a, _ in rows]); toll = np.array([float(b) for _, b in rows])
        if rng.random() < self.q_D:
            return "I recommend the scenic route.", (0, 0)
        if rng.random() < self.q_I:
            k = (k + rng.integers(1, 3)) % 3
        z = -(est + THETA[k] * toll) / self.T; z -= z.max()
        p = np.exp(z); p /= p.sum()
        return f"PROFILE={PROFILES[k]}; ROUTE={rng.choice(len(est), p=p) + 1}", (0, 0)


# ----------------------------------------------------------------------------- call log
class Log:
    def __init__(self, path, meta):
        self.path, self.lock, self.rec = path, threading.Lock(), {}
        self.calls = 0; self.tokens = [0, 0]
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("kind") == "meta":
                        changed = [k for k in meta if k in r and r[k] != meta[k]]
                        if changed:
                            what = "; ".join("the instructions to the model" if k == "instructions"
                                             else f"{k}: {r[k]} in the log, {meta[k]} now" for k in changed)
                            sys.exit(f"The log {os.path.basename(path)} was made with different settings ({what}).\n"
                                     "Use another --tag, or delete the log.")
                    else:
                        self.rec[tuple(r["key"])] = r
        else:
            with open(path, "w") as f:
                f.write(json.dumps({"kind": "meta", **meta}) + "\n")

    def get(self, key, digest):
        r = self.rec.get(key)
        if r is not None and r["digest"] != digest:
            sys.exit("A logged call does not match the current run (the run is not a continuation "
                     "of the log). Use another --tag, or delete the log.")
        return r

    def put(self, key, digest, reply, usage, user):
        r = {"key": list(key), "digest": digest, "user": user, "reply": reply}
        with self.lock:
            self.rec[key] = r; self.calls += 1
            self.tokens[0] += usage[0]; self.tokens[1] += usage[1]
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")
        return r


def ask(backend, log, key, user, seed_for_mock, retries=6):
    """One model call, served from the log if it is already there."""
    digest = hashlib.sha1(user.encode()).hexdigest()[:12]
    r = log.get(key, digest)
    if r is not None:
        return r["reply"]
    rng = np.random.default_rng(seed_for_mock)
    wait = 2.0
    for attempt in range(retries):
        try:
            reply, usage = backend(SYSTEM, user, rng)
            return log.put(key, digest, reply, usage, user)["reply"]
        except SystemExit:
            raise
        except Exception as e:                      # rate limits, time-outs, server errors
            if attempt == retries - 1:
                raise RuntimeError(f"The model call failed {retries} times ({type(e).__name__}: {e}). "
                                   "Progress is saved; run the same command again to resume.") from e
            time.sleep(wait); wait = min(wait * 2, 60.0)


# ----------------------------------------------------------------------------- the AgMS run
def setup(args):
    scale = args.agents / 300.0                                    # same congestion regime as Experiment 2
    net = A.Net([10, 14, 20], [110 * scale, 100 * scale, 150 * scale], [4, 1, 0])
    v, g = A.make_population(args.agents)
    text_id = np.random.default_rng(2026).integers(0, len(GOAL_TEXTS[0]), args.agents)
    return net, v, g, text_id


def run_llm(args, backend, log, net, v, g, text_id, seed, pool):
    """One replication with the model in the loop. Initial routes, re-planning days and human
    approvals use the same random streams as agms_core.run with the same seed."""
    N, R, L = len(v), net.R, args.L
    ss = np.random.SeedSequence(seed)
    r_init, _, _, _, r_ine, r_app, _, _ = [np.random.default_rng(s) for s in ss.spawn(8)]
    route = r_init.integers(0, R, N)
    mem = np.tile(net.t0[None, :, None], (N, 1, L)).astype(float); ptr = 0
    total = wsum = 0.0
    daily, shares, records = [], [], []
    for d in range(args.days):
        est = mem.mean(axis=2)
        replan = r_ine.random(N) < args.rho
        idx = np.flatnonzero(replan)
        users = [user_prompt(GOAL_TEXTS[g[i]][text_id[i]], est[i], net.toll, route[i], L) for i in idx]
        jobs = [(("loop", seed, d, int(i), 0), u, [seed, d, int(i), 0]) for i, u in zip(idx, users)]
        replies = list(pool.map(lambda j: ask(backend, log, *j), jobs))
        prop = route.copy(); invalid = np.zeros(N, bool)
        for i, u, rep in zip(idx, users, replies):
            out = parse(rep, R)
            records.append(dict(seed=seed, day=d, agent=int(i), cls=int(g[i]), text=int(text_id[i]),
                                est=[round(float(x), 1) for x in est[i]], cur=int(route[i]),
                                profile=None if out is None else out[0],
                                route=None if out is None else out[1], user=u))
            if out is None:
                invalid[i] = True
            else:
                prop[i] = out[1]
        u_app = r_app.random(N)
        needs_ok = (prop != route) & ~invalid                      # switches need human approval
        executed = np.where(needs_ok, u_app < args.alpha, True)
        route = np.where(invalid | ~executed, route, prop)         # validity shield keeps the current route
        flows = np.bincount(route, minlength=R)
        tt = net.times(flows)
        gc = tt[route] + 60.0 * net.toll[route] / v
        w = args.disc ** d
        total += w * gc.mean(); wsum += w
        daily.append(float(gc.mean())); shares.append(flows / N)
        mem[:, :, ptr % L] = tt[None, :]; ptr += 1
    return dict(W=total / wsum, daily=daily, shares=np.array(shares)), records


# ----------------------------------------------------------------------------- analysis
def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def softmax_rows(score, T):
    z = score / T; z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); return p / p.sum(axis=1, keepdims=True)


def fit_temperature(score, choice):
    """Maximum-likelihood temperature of a softmax over the scores (grid on a log scale)."""
    grid = np.exp(np.linspace(np.log(0.02), np.log(50.0), 400))
    ll = [np.log(softmax_rows(score, T)[np.arange(len(choice)), choice] + 1e-300).sum() for T in grid]
    return float(grid[int(np.argmax(ll))])


def surrogate_joint(cls, score3, q_I, q_D, T):
    """Law of the surrogate over (profile, route) plus 'invalid', for one input.
    score3[k] is the score vector under profile k. Returns a vector of length 3*R + 1."""
    R = score3.shape[1]
    out = np.zeros(3 * R + 1)
    for k in range(3):
        pk = (1 - q_I) if k == cls else q_I / 2
        out[k * R:(k + 1) * R] = pk * (1 - q_D) * softmax_rows(score3[k][None, :], T)[0]
    out[-1] = q_D
    return out


def analyse(args, net, v, g, runs, records, rep_records, backend_name, log):
    R = net.R
    valid = [r for r in records if r["profile"] is not None]
    n, nv = len(records), len(valid)
    cls = np.array([r["cls"] for r in valid]); prof = np.array([r["profile"] for r in valid])
    choice = np.array([r["route"] for r in valid]); est = np.array([r["est"] for r in valid])
    res = {"model": backend_name, "calls_in_loop": n, "settings": {k: getattr(args, k) for k in
           ("agents", "days", "seeds", "rho", "alpha", "L", "disc", "rep_inputs", "rep_calls")}}

    # interpretation: kernel by class, drift rate
    kern = np.zeros((3, 3), int)
    for c, p in zip(cls, prof):
        kern[c, p] += 1
    drift = int((cls != prof).sum())
    res["interpretation"] = {
        "kernel_counts_class_by_profile": kern.tolist(),
        "drift_rate": wilson(drift, nv),
        "drift_rate_by_class": [wilson(int(kern[c].sum() - kern[c, c]), int(kern[c].sum())) for c in range(3)],
        "drift_rate_by_text": [[wilson(sum(1 for r in valid if r["cls"] == c and r["text"] == t and r["profile"] != c),
                                       sum(1 for r in valid if r["cls"] == c and r["text"] == t))
                                for t in range(len(GOAL_TEXTS[c]))] for c in range(3)],
    }
    res["invalid_outputs"] = {"rate": wilson(n - nv, n)}

    # deliberation, relative to the profile the model stated
    score = -(est + THETA[prof][:, None] * net.toll[None, :])
    best = score.argmax(axis=1)
    regret = score.max(axis=1) - score[np.arange(nv), choice]
    hit = regret < 1e-9                                            # a tie for the best route is not an error
    cur = np.array([r["cur"] for r in valid])
    tolls, times = net.toll[choice], est[np.arange(nv), choice]
    dearer_faster = (~hit) & (tolls > net.toll[best]) & (times < est[np.arange(nv), best])
    T_hat = fit_temperature(score, choice)
    rng = np.random.default_rng(0)
    boot = [fit_temperature(score[b], choice[b]) for b in (rng.integers(0, nv, nv) for _ in range(200))]
    gap = np.sort(score, axis=1)[:, -1] - np.sort(score, axis=1)[:, -2]       # margin of the best route
    bins = [0, 0.5, 1, 2, 4, 1e9]; calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gap >= lo) & (gap < hi)
        if m.any():
            calib.append({"margin_min": [lo, None if hi > 1e8 else hi], "n": int(m.sum()),
                          "model_picks_best": float(hit[m].mean()),
                          "softmax_T1": float(softmax_rows(score[m], 1.0).max(axis=1).mean()),
                          "softmax_fitted": float(softmax_rows(score[m], T_hat).max(axis=1).mean())})
    res["deliberation"] = {"picks_best_route": wilson(int(hit.sum()), nv),
                           "mean_regret_min": float(regret.mean()),
                           "errors": {"n": int((~hit).sum()),
                                      "by_stated_profile": [{"profile": PROFILE_NAME[k], "decisions": int((prof == k).sum()),
                                                             "errors": int(((prof == k) & ~hit).sum())} for k in range(3)],
                                      "choose_a_faster_dearer_route": int(dearer_faster.sum()),
                                      "keep_the_current_route": int(((~hit) & (choice == cur)).sum()),
                                      "median_regret_min": float(np.median(regret[~hit])) if (~hit).any() else 0.0},
                           "fitted_temperature": [T_hat, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           "stays_on_current_route": float(np.mean(choice == cur)),
                           "by_margin": calib}
    q_I_hat, q_D_hat = res["interpretation"]["drift_rate"][0], res["invalid_outputs"]["rate"][0]
    fitted = dict(q_I=q_I_hat, q_D=q_D_hat, T=T_hat)
    default = dict(q_I=0.10, q_D=0.05, T=1.0)
    res["surrogate_parameters"] = {"default": default, "fitted_to_model": fitted}

    # distance of the default surrogate from the surrogate fitted to the model, exact, on every visited input
    cls_all = np.array([r["cls"] for r in records]); est_all = np.array([r["est"] for r in records])
    tv_fit = []
    for c, e in zip(cls_all, est_all):
        sc3 = -(e[None, :] + THETA[:, None] * net.toll[None, :])
        tv_fit.append(0.5 * np.abs(surrogate_joint(c, sc3, **default) - surrogate_joint(c, sc3, **fitted)).sum())
    res["default_vs_fitted_surrogate"] = {"mean_tv_per_decision": float(np.mean(tv_fit)),
                                          "max_tv_per_decision": float(np.max(tv_fit))}

    # total-variation distance between model and surrogate on visited inputs (repeated calls)
    if rep_records:
        tv = {"default": [], "fitted_to_model": []}; floor = {"default": [], "fitted_to_model": []}
        rng = np.random.default_rng(1)
        same = modal_best = rep_drift = rep_invalid = 0; modal_share = []
        for base, reps in rep_records:
            outs = [parse(x, R) for x in reps]
            K = len(outs); emp = np.zeros(3 * R + 1)
            vals, cnts = np.unique([-1 if o is None else o[0] * R + o[1] for o in outs], return_counts=True)
            top = int(vals[cnts.argmax()]); modal_share.append(cnts.max() / K); same += int(len(vals) == 1)
            if top >= 0:
                sc = -(np.array(base["est"]) + THETA[top // R] * net.toll)
                modal_best += int(sc.max() - sc[top % R] < 1e-9)
            rep_drift += sum(1 for o in outs if o is not None and o[0] != base["cls"])
            rep_invalid += sum(1 for o in outs if o is None)
            for o in outs:
                emp[-1 if o is None else o[0] * R + o[1]] += 1.0 / K
            sc3 = -(np.array(base["est"])[None, :] + THETA[:, None] * net.toll[None, :])
            for name, par in (("default", default), ("fitted_to_model", fitted)):
                q = surrogate_joint(base["cls"], sc3, **par)
                tv[name].append(0.5 * np.abs(emp - q).sum())
                sim = rng.multinomial(K, q, size=40) / K          # what K calls of the surrogate itself would give
                floor[name].append(0.5 * np.abs(sim - q[None, :]).sum(axis=1).mean())
        n_rep = sum(len(x[1]) for x in rep_records)
        res["repeated_calls"] = {"inputs": len(rep_records), "calls_per_input": len(rep_records[0][1]),
                                 "inputs_with_one_answer_in_all_calls": same,
                                 "mean_share_of_most_frequent_answer": float(np.mean(modal_share)),
                                 "most_frequent_answer_is_a_best_route": modal_best,
                                 "profile_other_than_the_instruction_class": [rep_drift, n_rep],
                                 "invalid_outputs": [rep_invalid, n_rep]}
        res["distance_to_surrogate"] = {
            "inputs": len(rep_records), "calls_per_input": len(rep_records[0][1]),
            **{name: {"mean_tv": float(np.mean(tv[name])), "se": float(np.std(tv[name], ddof=1) / np.sqrt(len(tv[name]))) if len(tv[name]) > 1 else float("nan"),
                      "sampling_floor": float(np.mean(floor[name]))} for name in tv}}

    # system outcomes on the same seeds (common random numbers for initial routes, re-planning, approvals)
    kw = dict(days=args.days, disc=args.disc, L=args.L, rho=args.rho, alpha=args.alpha)
    ideal = dict(perfect_spec=True, faithful=True, no_halluc=True, full_exec=True)
    burn = min(10, args.days // 2)                                  # days left out of the day-to-day statistics
    rows = {"model": [], "surrogate_default": [], "surrogate_fitted": [], "reference": []}
    sd = {k: [] for k in rows}; share = {k: [] for k in rows}
    for s, run in zip(range(args.seeds), runs):
        outs = {"model": run,
                "surrogate_default": A.run(net, v, g, seed=s, **kw, **default),
                "surrogate_fitted": A.run(net, v, g, seed=s, **kw, **fitted),
                "reference": A.run(net, v, g, seed=s, **kw, **default, **ideal)}
        for k, o in outs.items():
            rows[k].append(float(o["W"]))
            dc = np.asarray(o["daily"] if k == "model" else o["daily_cost"])
            sd[k].append(float(dc[burn:].std()))
    W = {k: np.array(x) for k, x in rows.items()}
    ci = lambda x: [float(np.mean(x)), float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))
                    if len(x) > 1 else float("nan")]             # mean and half-width of the 95% Student-t interval
    res["system"] = {
        "W_per_seed": rows,
        "W": {k: ci(x) for k, x in W.items()},
        "delegation_loss": {k: ci(W[k] - W["reference"]) for k in ("model", "surrogate_default", "surrogate_fitted")},
        "intervals": "mean and half-width of the 95% Student-t interval over replications (paired by seed for differences)",
        "model_minus_surrogate_default": ci(W["model"] - W["surrogate_default"]),
        "model_minus_surrogate_fitted": ci(W["model"] - W["surrogate_fitted"]),
        "burn_in_days": burn,
        "sd_daily_cost_after_burn_in": {k: float(np.mean(x)) for k, x in sd.items()},
        "model_route_shares_after_burn_in": np.mean([r["shares"][burn:].mean(axis=0) for r in runs], axis=0).tolist(),
    }

    # Corollary 4 in normalised form: |W - W_sur| <= beta * osc(c) / (1 - beta) * (expected deviation per day)
    if rep_records:
        lo = np.mean((net.t0[None, :] + 60 * net.toll[None, :] / v[:, None]).min(axis=1))
        hi = np.mean((net.times(np.full(R, float(len(v))))[None, :] + 60 * net.toll[None, :] / v[:, None]).max(axis=1))
        calls_per_day = n / (args.seeds * args.days)
        tvm = {name: res["distance_to_surrogate"][name]["mean_tv"] for name in ("default", "fitted_to_model")}
        res["corollary_4"] = {"osc_c_min": float(hi - lo), "model_calls_per_day": calls_per_day,
                              **{name: float(args.disc * (hi - lo) / (1 - args.disc) * calls_per_day * tvm[name])
                                 for name in tvm},
                              # the bound is below the trivial bound osc(c) only if the summed deviation per day
                              # is below (1 - beta) / beta
                              "summed_deviation_per_day": {name: float(calls_per_day * tvm[name]) for name in tvm},
                              "single_agent_deviation_per_day": {name: float(args.rho * tvm[name]) for name in tvm},
                              "informative_below": float((1 - args.disc) / args.disc)}
    res["usage"] = {"new_calls_this_run": log.calls, "input_tokens": log.tokens[0], "output_tokens": log.tokens[1],
                    "calls_in_log": len(log.rec)}
    return res


def report(res):
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    pm = lambda x, sign="": f"{x[0]:{sign}.3f}" + (f" +- {x[1]:.3f}" if x[1] == x[1] else " (one replication, no interval)")
    print("\n================ Experiment 4: language-model agents in the loop ================")
    print(f"model: {res['model']}    calls in the loop: {res['calls_in_loop']}")
    print(f"interpretation drift rate   {f(res['interpretation']['drift_rate'])}   (surrogate default 0.100)")
    print(f"invalid outputs             {f(res['invalid_outputs']['rate'])}   (surrogate default 0.050)")
    d = res["deliberation"]
    print(f"picks a best route          {f(d['picks_best_route'])}   mean regret {d['mean_regret_min']:.3f} min")
    e = d["errors"]
    print("  errors by stated profile  " + ", ".join(f"{x['profile']} {x['errors']}/{x['decisions']}" for x in e["by_stated_profile"])
          + f"; {e['choose_a_faster_dearer_route']} of {e['n']} choose a faster, dearer route; "
          f"{e['keep_the_current_route']} keep the current route")
    print(f"fitted softmax temperature  {d['fitted_temperature'][0]:.2f} [{d['fitted_temperature'][1]:.2f}, "
          f"{d['fitted_temperature'][2]:.2f}]   (surrogate default 1.00)")
    print(f"default surrogate vs surrogate fitted to the model: TV per decision "
          f"{res['default_vs_fitted_surrogate']['mean_tv_per_decision']:.3f}")
    if "repeated_calls" in res:
        q = res["repeated_calls"]
        print(f"repeated calls: the same answer in all {q['calls_per_input']} calls for "
              f"{q['inputs_with_one_answer_in_all_calls']} of {q['inputs']} inputs; most frequent answer has share "
              f"{q['mean_share_of_most_frequent_answer']:.3f}; other profile in "
              f"{q['profile_other_than_the_instruction_class'][0]} of {q['profile_other_than_the_instruction_class'][1]} calls")
    if "distance_to_surrogate" in res:
        for name, x in res["distance_to_surrogate"].items():
            if isinstance(x, dict):
                print(f"distance to surrogate ({name}): TV {x['mean_tv']:.3f} +- {1.96 * x['se']:.3f}, "
                      f"sampling floor {x['sampling_floor']:.3f}")
    s = res["system"]
    for k, x in s["W"].items():
        print(f"W {k:18s} {pm(x)} min")
    print(f"model - surrogate (default) {pm(s['model_minus_surrogate_default'], '+')}")
    print(f"model - surrogate (fitted)  {pm(s['model_minus_surrogate_fitted'], '+')}")
    print(f"delegation loss: model {pm(s['delegation_loss']['model'])}, surrogate (default) "
          f"{pm(s['delegation_loss']['surrogate_default'])}")
    if "corollary_4" in res:
        c = res["corollary_4"]
        print(f"Corollary 4 with the measured distance: {c['default']:.0f} min (default surrogate), against "
              f"osc(c) = {c['osc_c_min']:.0f} min; summed deviation per day {c['summed_deviation_per_day']['default']:.2f} "
              f"(one agent {c['single_agent_deviation_per_day']['default']:.3f}), informative below {c['informative_below']:.3f}")
    u = res["usage"]
    if res.get("run", {}).get("recomputed_from_log"):
        print(f"recomputed from the log; the run that made the calls finished {res['run'].get('finished')}: "
              f"{u['new_calls_this_run']} calls, tokens in/out {u['input_tokens']}/{u['output_tokens']}")
    else:
        print(f"new calls this run: {u['new_calls_this_run']}  tokens in/out: {u['input_tokens']}/{u['output_tokens']}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model name, exactly as your account lists it")
    ap.add_argument("--api", choices=["responses", "chat"], default="responses", help="OpenAI endpoint to use")
    ap.add_argument("--reasoning-effort", default=None, help="optional, passed to the model if it supports it")
    ap.add_argument("--mock", action="store_true", help="offline stand-in for the model")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", help="no new calls; needs a complete log")
    ap.add_argument("--tag", default=None, help="name of the log and result files (default: the model name)")
    ap.add_argument("--agents", type=int, default=100); ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=6); ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.8); ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--disc", type=float, default=0.95)
    ap.add_argument("--rep-inputs", type=int, default=60, help="visited inputs that are called again")
    ap.add_argument("--rep-calls", type=int, default=20, help="repeated calls per such input")
    ap.add_argument("--workers", type=int, default=8, help="calls made in parallel")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if args.list_models:
        print("\n".join(OpenAIBackend("-", args.api, None, args.timeout).list_models())); return
    if not args.mock and not args.model and not args.analyse_only:
        ap.error("give --model MODEL_NAME (see --list-models), or --mock")
    tag = args.tag or ("mock" if args.mock else re.sub(r"[^A-Za-z0-9_.-]", "_", args.model or "run"))
    backend = MockBackend() if args.mock else (None if args.analyse_only else
                                               OpenAIBackend(args.model, args.api, args.reasoning_effort, args.timeout))
    meta = {k: getattr(args, k) for k in ("agents", "days", "rho", "alpha", "L", "disc")}
    meta["model"] = "mock" if args.mock else args.model
    meta["instructions"] = SYSTEM
    log_path = os.path.join(OUT, f"exp4_llm_log_{tag}.jsonl")
    if args.analyse_only:
        if not os.path.exists(log_path):
            sys.exit(f"No log at {log_path}. Give the --tag (or --model) of the run to analyse.")
        with open(log_path) as fh:
            meta["model"] = json.loads(fh.readline()).get("model")

        class NoCalls:
            name = meta["model"]

            def __call__(self, *a):
                raise SystemExit("The log is incomplete: run without --analyse-only to finish it.")
        backend = NoCalls()
    log = Log(log_path, meta)
    net, v, g, text_id = setup(args)

    expected = int(args.agents * args.rho * args.days * args.seeds + args.rep_inputs * args.rep_calls)
    print(f"model: {backend.name}   about {expected} calls in total; {len(log.rec)} already in the log")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if not args.mock and not args.analyse_only and len(log.rec) == 0:       # three trial calls before the full run
            trial = [user_prompt(GOAL_TEXTS[c][0], net.t0 + c, net.toll, 1, args.L) for c in range(3)]
            got = [ask(backend, log, ("trial", 0, 0, c, 0), u, [0, c]) for c, u in enumerate(trial)]
            for u, r in zip(trial, got):
                print("  trial reply:", repr(r), "->", parse(r, net.R))
            if all(parse(r, net.R) is None for r in got):
                sys.exit("None of the three trial replies followed the required format, so the run was stopped "
                         "before spending more calls. Try another model, or --api chat.")
        runs, records = [], []
        for s in range(args.seeds):
            t0 = time.time()
            run, rec = run_llm(args, backend, log, net, v, g, text_id, s, pool)
            runs.append(run); records += rec
            print(f"  replication {s + 1}/{args.seeds}: W = {run['W']:.3f} min, {len(rec)} decisions, {time.time() - t0:.0f} s")
        # repeated calls on a sample of visited inputs
        rng = np.random.default_rng(7)
        pick = rng.choice(len(records), size=min(args.rep_inputs, len(records)), replace=False) if args.rep_calls > 0 else []
        jobs = [(("rep", int(i), 0, 0, k), records[i]["user"], [7, int(i), k]) for i in pick for k in range(args.rep_calls)]
        replies = list(pool.map(lambda j: ask(backend, log, *j), jobs))
        rep_records = [(records[i], replies[a * args.rep_calls:(a + 1) * args.rep_calls]) for a, i in enumerate(pick)]
    res = analyse(args, net, v, g, runs, records, rep_records, backend.name, log)
    res["run"] = {"endpoint": "none (stand-in)" if args.mock else args.api, "reasoning_effort": args.reasoning_effort,
                  "finished": time.strftime("%Y-%m-%d %H:%M")}
    path = os.path.join(OUT, f"exp4_llm_results_{tag}.json")
    if args.analyse_only:                                # keep the record of the run that made the calls
        res["run"] = {"endpoint": "unknown", "reasoning_effort": None, "finished": "unknown"}
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
            res["run"] = old.get("run", res["run"]); res["usage"] = old.get("usage", res["usage"])
        res["run"]["recomputed_from_log"] = time.strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    report(res)
    print(f"\nsaved: {os.path.relpath(path)}\n       {os.path.relpath(log_path)}  (the record of every call)")


if __name__ == "__main__":
    main()
