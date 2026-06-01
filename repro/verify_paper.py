"""
verify_paper.py — FIREWALL DE REPRODUCIBILIDAD. Cruza cada numero declarado en el
paper contra results/*.json y FALLA si hay discrepancia > tolerancia.

Operacionaliza el principio "cero hardcode": si un numero del .tex no se puede
trazar a un JSON regenerado, es un FALLO.

USO:
  python verify_paper.py [--tex DIR] [--results DIR]
- CLAIMS: registro semilla con numeros VERIFICADOS contra el codigo (extiendelo
  con el resto de cifras del paper).
- tex_number_inventory(): lista todos los tokens numericos del .tex para cablearlos.

NOTA: los paper_value de abajo provienen del PDF MICAI. Ajusta/expande segun el
.tex real. n_mismatch>0 => hay cifras sin respaldo o stale.
"""
import os, re, json, argparse
from pathlib import Path
import _bootstrap

# ---------- navegacion de JSON: "a.b[0].c" ----------
def jget(obj, path):
    cur = obj
    for tok in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if tok.startswith("[") and tok.endswith("]"):
            cur = cur[int(tok[1:-1])]
        else:
            cur = cur[tok]
    return cur

# ---------- REGISTRO SEMILLA (extender) ----------
# kind: "rel" (tolerancia relativa) | "abs" (absoluta) | "exact"
CLAIMS = [
    # name, paper_value, json_file, json_path, tol, kind
    ("degeneration_ratio", 1.27, "expB_structural_collapse.json",
     "B2_decoder_degeneration.degeneration_ratio", 0.02, "abs"),
    ("distinct_objective_points", 39401, "expB_structural_collapse.json",
     "B2_decoder_degeneration.n_distinct_objective_points", 50, "abs"),
    ("pca_pc1_plus_pc2_mohho", 0.99, "expB_structural_collapse.json",
     "B1_effective_dimensionality.mohho_realcoded.pc1_plus_pc2", 0.01, "abs"),
    ("f1_vs_f2_range_ratio", 39.4, "expB_structural_collapse.json",
     "B2_decoder_degeneration.f1_vs_f2_range_ratio", 0.5, "abs"),
    ("decoder_margin_greedy_pct", 6.93, "expC_decoder_ladder.json",
     "separation_collapse.greedy_perm_minus_rk_pct", 0.1, "abs"),
    ("decoder_margin_C1_pct", 1.45, "expC_decoder_ladder.json",
     "separation_collapse.C1_perm_minus_rk_pct", 0.1, "abs"),
    ("decoder_margin_C2_pct", 4.27, "expC_decoder_ladder.json",
     "separation_collapse.C2_perm_minus_rk_pct", 0.1, "abs"),
    # ---- ladder per-run HV (Tabla 5/7), repo UNIFICADO a seeds 1-30 ----
    # MOHHO y NSGA: nsga2_comparison.json (estudio principal regenerado a 1-30).
    # random/Discrete/perm-*: ladder_v5.json (ya a 1-30).
    ("hv_mohho_mean", 302756, "nsga2_comparison.json", "mohho.hv_mean", 0.005, "rel"),
    ("hv_nsga2_mean", 293367, "nsga2_comparison.json", "nsga2.hv_mean", 0.005, "rel"),
    ("hv_random_restart", 310214, "ladder_v5.json", "methods.random_restart.hv_mean", 0.005, "rel"),
    ("hv_discrete_mohho", 316792, "ladder_v5.json", "methods.discrete_mohho.hv_mean", 0.005, "rel"),
    ("hv_perm_nsga2", 318151, "ladder_v5.json", "methods.perm_nsga2.hv_mean", 0.005, "rel"),
    ("hv_perm_moead", 314846, "ladder_v5.json", "methods.perm_moead.hv_mean", 0.005, "rel"),
    ("combined_hv_discrete", 321408, "ladder_v5.json", "methods.discrete_mohho.combined_front_hv", 0.005, "rel"),
    ("combined_hv_perm_nsga2", 321935, "ladder_v5.json", "methods.perm_nsga2.combined_front_hv", 0.005, "rel"),
    ("combined_size_discrete", 137, "ladder_v5.json", "methods.discrete_mohho.combined_front_size", 0, "exact"),
    ("mohho_igd_1to30", 0.0212, "nsga2_comparison.json", "mohho.igd", 0.002, "abs"),
    ("nsga_igd_1to30", 0.0071, "nsga2_comparison.json", "nsga2.igd", 0.002, "abs"),
    ("mohho_cv_1to30", 2.36, "ladder_v5.json", "methods.naive_mohho.cv_pct", 0.05, "abs"),
    ("discrete_cv_1to30", 0.71, "ladder_v5.json", "methods.discrete_mohho.cv_pct", 0.05, "abs"),
    ("perm_nsga_cv_1to30", 0.58, "ladder_v5.json", "methods.perm_nsga2.cv_pct", 0.05, "abs"),
    ("canonical_levy_fematched_1to30", 309180, "control_canonical_hho.json",
     "fe_matched.hv_mean", 0.01, "rel"),
    # ---- omnibus / mechanism ----
    ("omnibus_chi2", 112.6, "omnibus_visa_paired.json", "chi2", 0.5, "abs"),
    ("omnibus_CD", 1.38, "omnibus_visa_paired.json", "nemenyi_CD", 0.01, "abs"),
    ("tau_sbx", 0.99, "operator_order.json",
     "operators.SBX crossover (GA).mean_tau", 0.01, "abs"),
    # ---- FIFO baseline + extremes (Tabla 4) ----
    ("fifo_f1", 8.7891, "summary.json", "baseline.f1", 0.001, "abs"),
    ("fifo_f2", 13.0, "summary.json", "baseline.f2", 0.01, "abs"),
    ("fifo_f3", 1940, "summary.json", "baseline.f3", 1, "abs"),
    ("combined_pareto_size", 104, "summary.json", "combined_pareto_size", 0, "exact"),
    ("min_f1_sol_f1", 8.7884, "summary.json", "best_f1.[0]", 0.001, "abs"),
    ("min_f1_sol_f3", 680, "summary.json", "best_f1.[2]", 1, "abs"),
    ("min_f2_sol_f2", 2.0, "summary.json", "best_f2.[1]", 0.01, "abs"),
    # ---- Taguchi ----
    ("taguchi_grand_mean_sn", 109.46, "taguchi.json", "grand_mean_sn", 0.01, "abs"),
    # ---- policy of Fig.10 (f2 recomputed) ----
    ("policy_f2_years", 7.59, "policy_impact.json", "f2", 0.01, "abs"),
    ("equity_wait_std_fifo", 3.14, "equity_audit.json",
     "front_ranges.wait_std.fifo", 0.02, "abs"),
    ("equity_wait_std_front_best", 0.75, "equity_audit.json",
     "front_ranges.wait_std.min", 0.02, "abs"),
    ("equity_gini_fifo", 0.79, "equity_audit.json",
     "front_ranges.wait_gini.fifo", 0.02, "abs"),
    ("equity_gini_front_best", 0.17, "equity_audit.json",
     "front_ranges.wait_gini.min", 0.02, "abs"),
    ("equity_jain_fifo", 0.80, "equity_audit.json",
     "front_ranges.jain_inverse_wait.fifo", 0.02, "abs"),
    ("equity_jain_front_best", 0.94, "equity_audit.json",
     "front_ranges.jain_inverse_wait.max", 0.02, "abs"),
    # ---- Friedman ranks (Tabla 8, visa column) ----
    ("rank_perm_nsga2_visa", 1.60, "omnibus_visa_paired.json", "avg_rank.perm-NSGA-II", 0.01, "abs"),
    ("rank_discrete_visa", 2.23, "omnibus_visa_paired.json", "avg_rank.Discrete-MOHHO", 0.01, "abs"),
    ("rank_perm_moead_visa", 2.53, "omnibus_visa_paired.json", "avg_rank.perm-MOEA/D", 0.01, "abs"),
    ("rank_random_visa", 4.20, "omnibus_visa_paired.json", "avg_rank.Random restart", 0.01, "abs"),
    ("rank_mohho_visa", 4.67, "omnibus_visa_paired.json", "avg_rank.MOHHO", 0.01, "abs"),
    ("rank_nsga2_visa", 5.77, "omnibus_visa_paired.json", "avg_rank.NSGA-II", 0.01, "abs"),
    # ---- v5: competent MO-HHO + order-preservation ----
    ("competent_hv_mean", 316347, "ladder_v5.json", "methods.competent_mohho.hv_mean", 0.01, "rel"),
    ("competent_vs_random_pct", 2.0, "ladder_v5.json",
     "key_finding.competent_beats_random_pct", 0.6, "abs"),
    ("competent_beats_random", True, "ladder_v5.json",
     "key_finding.competent_beats_random", 0, "exact"),
    ("naive_beats_random", False, "ladder_v5.json",
     "key_finding.naive_beats_random", 0, "exact"),
    ("competent_zdt2_validation", 0.99, "competent_mohho_validation.json",
     "configs.[3].per_benchmark.ZDT2.hv_over_true", 0.03, "abs"),
    ("tau_nsga2", 0.99, "tau_by_method.json", "methods.nsga2_realcoded.tau_mean", 0.02, "abs"),
    ("hv_tau_spearman_rho", -0.21, "hv_vs_tau.json", "spearman_hv_vs_tau.rho", 0.05, "abs"),
    ("hv_tau_correlation_weak", True, "hv_vs_tau.json", "correlation_is_weak", 0, "exact"),
    # ---- v6: controlled 2x2 (two conditions, sec:twoconditions) ----
    ("c2x2_order_nds_hv", 315730, "factorial_2x2_conditions.json",
     "cells.order_nds.hv_mean", 0.01, "rel"),
    ("c2x2_order_gated_hv", 304126, "factorial_2x2_conditions.json",
     "cells.order_gated.hv_mean", 0.01, "rel"),
    ("c2x2_near_nds_hv", 305892, "factorial_2x2_conditions.json",
     "cells.near_nds.hv_mean", 0.01, "rel"),
    ("c2x2_near_gated_hv", 304760, "factorial_2x2_conditions.json",
     "cells.near_gated.hv_mean", 0.01, "rel"),
    ("c2x2_random_hv", 310214, "factorial_2x2_conditions.json",
     "random_restart.hv_mean", 0.01, "rel"),
    ("c2x2_order_nds_vs_random_pct", 1.78, "factorial_2x2_conditions.json",
     "cells.order_nds.vs_random_pct", 0.2, "abs"),
    ("c2x2_order_nds_A12", 0.791, "factorial_2x2_conditions.json",
     "cells.order_nds.A12_vs_random", 0.02, "abs"),
    ("c2x2_only_order_nds_wins", True, "factorial_2x2_conditions.json",
     "only_order_nds_wins", 0, "exact"),
    ("c2x2_eta2_operator", 0.076, "factorial_2x2_conditions.json",
     "anova.eta2_operator_A", 0.01, "abs"),
    ("c2x2_eta2_selection", 0.145, "factorial_2x2_conditions.json",
     "anova.eta2_selection_B", 0.01, "abs"),
    ("c2x2_eta2_interaction", 0.098, "factorial_2x2_conditions.json",
     "anova.eta2_interaction_AxB", 0.01, "abs"),
    ("c2x2_interaction_significant", True, "factorial_2x2_conditions.json",
     "anova.interaction_significant", 0, "exact"),
    # ---- v6 FASE 2: competent across 4 structures (structures_v6.json) ----
    ("struct_competent_knapsack_pos", 1, "structures_v6.json",
     "placement.knapsack.competent_position_of_7", 0, "exact"),
    ("struct_competent_knapsack_rank", 1.13, "structures_v6.json",
     "placement.knapsack.competent_avg_rank", 0.05, "abs"),
    ("struct_competent_knapsack_perm_best", False, "structures_v6.json",
     "placement.knapsack.perm_native_still_best", 0, "exact"),
    ("struct_competent_visa_pos", 2, "structures_v6.json",
     "placement.visa.competent_position_of_7", 0, "exact"),
    ("struct_competent_tsp_pos", 5, "structures_v6.json",
     "placement.TSP.competent_position_of_7", 0, "exact"),
    ("struct_competent_flowshop_pos", 2, "structures_v6.json",
     "placement.flow-shop.competent_position_of_7", 0, "exact"),
]


def check(results_dir):
    rows = []; n_mismatch = 0
    for name, pv, jf, jp, tol, kind in CLAIMS:
        fp = Path(results_dir) / jf
        if not fp.exists():
            rows.append({"name": name, "status": "JSON_MISSING", "file": jf,
                         "paper": pv, "json": None}); n_mismatch += 1; continue
        try:
            jv = jget(json.loads(fp.read_text()), jp)
        except Exception as e:
            rows.append({"name": name, "status": "PATH_ERROR", "file": jf,
                         "path": jp, "err": str(e), "paper": pv}); n_mismatch += 1; continue
        if kind == "exact":
            ok = (jv == pv)
        elif kind == "abs":
            ok = abs(float(jv) - float(pv)) <= tol
        else:
            ok = abs(float(jv) - float(pv)) <= tol * abs(float(pv))
        if not ok: n_mismatch += 1
        rows.append({"name": name, "status": "OK" if ok else "MISMATCH",
                     "paper": pv, "json": jv, "delta": (float(jv) - float(pv))
                     if isinstance(jv, (int, float)) else None})
    return rows, n_mismatch


def tex_number_inventory(tex_dir):
    inv = []
    p = Path(tex_dir)
    if not p.exists():
        return inv
    for tex in p.rglob("*.tex"):
        for ln, line in enumerate(tex.read_text(errors="ignore").splitlines(), 1):
            for m in re.finditer(r"(?<![\\A-Za-z])\d[\d,]*\.?\d*", line):
                tok = m.group(0)
                if len(tok.replace(",", "").replace(".", "")) >= 3:
                    inv.append({"file": tex.name, "line": ln, "token": tok,
                                "context": line.strip()[:120]})
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=_bootstrap.results_dir())
    ap.add_argument("--tex", default=None,
                    help="dir del paper LaTeX (MICAI/) para inventario de numeros")
    a = ap.parse_args()
    rows, n_mismatch = check(a.results)
    inv = tex_number_inventory(a.tex) if a.tex else []
    out = {"n_claims_checked": len(rows), "n_mismatch": n_mismatch,
           "rows": rows, "tex_inventory_count": len(inv),
           "tex_inventory_sample": inv[:40]}
    Path(a.results, "_verify_paper.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"  [{r['status']:12s}] {r['name']}: paper={r.get('paper')} json={r.get('json')}")
    print(f"\nn_mismatch = {n_mismatch} (de {len(rows)} claims cableados). "
          f"Inventario .tex: {len(inv)} tokens. -> _verify_paper.json")
    if n_mismatch:
        print("FALLO: hay cifras del paper sin respaldo o stale. Corrige el PAPER, no el JSON.")


if __name__ == "__main__":
    main()
