"""
9.7 lever: turn the search-headroom mechanism from a TWO-POINT association into a
DEMONSTRATED functional relationship. We vary the MOMKP capacity tightness (a knob
on how much the feasibility-preserving decoder leaves for search), and at each
level measure (i) headroom = gap from random restart to the best method, and
(ii) the real-coded swarm's capture fraction = how much of that gap it recovers.
If capture rises monotonically with headroom, "the random-key swarm's
competitiveness tracks search headroom" stops being a two-point fit and becomes a
measured curve.

Output: app/data/results/headroom_sweep.json
"""
import json, time
from pathlib import Path
import numpy as np

from app.core.mohho import compute_hypervolume
import second_problem as sp

RESULTS = Path("app/data/results")
FRACS = [0.2, 0.35, 0.5, 0.65, 0.8]   # capacity / total-weight: tighter -> more headroom
S = 10
REF = sp.REF


def make_problem(frac):
    p = sp.MOMKP(seed=7)
    p.cap = frac * p.weight.sum(axis=0)   # retune the only knob
    return p


def mean_hv(fn, prob, seeds):
    return float(np.mean([compute_hypervolume(fn(prob, s), REF) for s in seeds]))


def main():
    t0 = time.time(); seeds = list(range(1, S + 1)); out = []
    for frac in FRACS:
        prob = make_problem(frac)
        rnd = mean_hv(sp.run_random, prob, seeds)
        hho = mean_hv(sp.run_hho_realcoded, prob, seeds)     # the random-key swarm
        best = mean_hv(sp.run_permnsga, prob, seeds)         # representation-matched best
        headroom = (best - rnd) / best if best > 0 else 0.0
        capture = (hho - rnd) / (best - rnd) if best > rnd else float("nan")
        out.append({"frac": frac, "random": rnd, "swarm": hho, "best": best,
                    "headroom_pct": 100 * headroom, "swarm_capture_pct": 100 * capture})
        print(f"frac={frac}: headroom={100*headroom:5.1f}%  swarm_capture={100*capture:5.1f}%  "
              f"(rnd={rnd:.4f} hho={hho:.4f} best={best:.4f})  ({time.time()-t0:.0f}s)")
    # monotonicity check (Spearman of capture vs headroom)
    from scipy.stats import spearmanr
    hr = [r["headroom_pct"] for r in out]; cap = [r["swarm_capture_pct"] for r in out]
    rho, p = spearmanr(hr, cap)
    json.dump({"fracs": FRACS, "seeds": S, "sweep": out,
               "spearman_rho": float(rho), "spearman_p": float(p),
               "elapsed_s": time.time() - t0}, open(RESULTS / "headroom_sweep.json", "w"), indent=2)
    print(f"\nSpearman(capture vs headroom) rho={rho:.3f} p={p:.3f}  "
          f"({'MONOTONE' if rho>0.7 else 'weak'})  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
