"""
Experiment 3 -- Shielded safety (Theorem 6): agency-independent invariance under
complete mediation with product-form admissible sets, and its two failure modes
(individual-only validation; unmediated human channels reachable by messages).
Safe set: daily flow on the restricted bridge route <= 90 vehicles.
"""
import json, os
import numpy as np
import agms_core as A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
NET = A.Net([10, 14, 12], [110, 100, 120], [4, 1, 0], restricted=2)
N = 300
V, G = A.make_population(N)
BASE = dict(q_I=0.10, q_D=0.05, alpha=1.0, T=1.0, rho=0.1, L=5, days=200, disc=0.95, permits=90)
R = 20


def stats(**kw):
    vr, ex, W = [], [], []
    for s in range(R):
        o = A.run(NET, V, G, seed=s, **{**BASE, **kw})
        vr.append(o["viol_rate"]); ex.append(o["mean_excess"]); W.append(o["W"])
    return {"viol_rate": float(np.mean(vr)), "viol_rate_max": float(np.max(vr)),
            "mean_excess": float(np.mean(ex)), "W": float(np.mean(W))}


def main():
    rows = []
    for shield in ["validity", "quota"]:
        for qD in [0.0, 0.1, 0.3]:
            for mis in [0.0, 0.2, 1.0]:
                r = stats(shield=shield, q_D=qD, misaligned_share=mis)
                rows.append({"shield": shield, "q_D": qD, "misaligned": mis, **r})
    hum = []
    for mis in [0.2, 1.0]:
        for kappa in [0.0, 0.05, 0.1, 0.25, 0.5]:
            r = stats(shield="quota", humans=100, kappa=kappa, persuade=True, misaligned_share=mis)
            hum.append({"misaligned": mis, "kappa": kappa, **r})
    res = {"agents_only": rows, "with_unmediated_humans": hum}
    with open(os.path.join(OUT, "exp3_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    for r in rows: print(r)
    for r in hum: print(r)


if __name__ == "__main__":
    main()
