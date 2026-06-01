"""
Omnibus multi-method statistics (journal-grade rigor).
  - Visa (base) problem: the six methods now share seeds 1-30 where available;
    this historical script still reports the independent-sample Kruskal-Wallis
    diagnostic used before the paired omnibus script became canonical.
  - MOMKP (second problem): all methods share seeds 1-30 -> PAIRED -> Friedman
    test + Nemenyi critical difference (the Demsar standard).
Output: app/data/results/omnibus_stats.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare, rankdata

R = Path("app/data/results")
# Nemenyi q_0.05 for infinite df, by number of treatments k (Demsar 2006, Table 5)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def holm(pairs):
    order = sorted(pairs, key=lambda x: x[2])
    m = len(order); out = []
    for i, (a, b, p) in enumerate(order):
        out.append((a, b, p, min(1.0, p * (m - i))))
    return out


def visa_kruskal():
    st = json.load(open(R / "stats_test.json"))
    ctl = json.load(open(R / "controls.json"))
    dm = json.load(open(R / "discrete_mohho.json"))
    pn = json.load(open(R / "perm_nsga.json"))
    md = json.load(open(R / "perm_moead.json"))
    groups = {
        "NSGA-II": st["nsga2_hv"], "Random restart": ctl["random_restart"]["per_seed_hv"],
        "MOHHO": st["mohho_hv"], "Discrete-MOHHO": dm["per_run_hv"],
        "perm-MOEA/D": md["per_run_hv"], "perm-NSGA-II": pn["per_run_hv"],
    }
    names = list(groups); data = [np.array(groups[n]) for n in names]
    H, p = kruskal(*data)
    # average rank over pooled data (higher HV -> better -> we rank so rank1=best)
    pooled = np.concatenate(data)
    rk = rankdata(-pooled)  # rank 1 = largest HV
    idx = 0; avg = {}
    for n, d in zip(names, data):
        avg[n] = float(np.mean(rk[idx:idx + len(d)])); idx += len(d)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            _, pp = mannwhitneyu(data[i], data[j], alternative="two-sided")
            pairs.append((names[i], names[j], float(pp)))
    return {"test": "Kruskal-Wallis", "H": float(H), "p": float(p),
            "mean_pooled_rank": avg,
            "holm_pairwise": [{"a": a, "b": b, "p_raw": p0, "p_holm": ph}
                              for a, b, p0, ph in holm(pairs)]}


def momkp_friedman():
    f = R / "second_problem.json"
    if not f.exists():
        return None
    M = json.load(open(f))["methods"]
    names = list(M)
    X = np.array([M[n]["per_run_hv"] for n in names])  # k x N (HV, larger better)
    # per-block (column) ranks, rank 1 = best (largest HV)
    ranks = np.zeros_like(X)
    for col in range(X.shape[1]):
        ranks[:, col] = rankdata(-X[:, col])
    avg = {names[i]: float(ranks[i].mean()) for i in range(len(names))}
    chi, p = friedmanchisquare(*[X[i] for i in range(len(names))])
    k, Nn = len(names), X.shape[1]
    CD = Q05[k] * np.sqrt(k * (k + 1) / (6 * Nn))
    return {"test": "Friedman + Nemenyi", "chi2": float(chi), "p": float(p),
            "k": k, "N": Nn, "avg_rank": avg, "nemenyi_CD": float(CD)}


def main():
    out = {"visa": visa_kruskal(), "momkp": momkp_friedman()}
    json.dump(out, open(R / "omnibus_stats.json", "w"), indent=2)
    v = out["visa"]
    print("=== VISA: Kruskal-Wallis ===")
    print(f"H={v['H']:.1f}, p={v['p']:.2e}")
    print("mean pooled rank (lower=better):")
    for n, r in sorted(v["mean_pooled_rank"].items(), key=lambda x: x[1]):
        print(f"  {n:16s} {r:6.1f}")
    if out["momkp"]:
        m = out["momkp"]
        print(f"\n=== MOMKP: Friedman chi2={m['chi2']:.1f}, p={m['p']:.2e}, CD={m['nemenyi_CD']:.3f} ===")
        for n, r in sorted(m["avg_rank"].items(), key=lambda x: x[1]):
            print(f"  {n:22s} avg rank {r:.2f}")
    else:
        print("\n(MOMKP not ready yet -- rerun after second_problem.json exists)")


if __name__ == "__main__":
    main()
