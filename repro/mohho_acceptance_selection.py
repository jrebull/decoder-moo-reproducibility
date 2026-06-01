"""
mohho_acceptance_selection.py — Decide que politica de aceptacion del MOHHO es SANA
en benchmarks con frente verdadero conocido (ZDT1, ZDT2, DTLZ2), y si NINGUNA sanea
ZDT2, lo reporta como patologia de operadores/archivo (no de la aceptacion).

NOTA CLAVE (hallazgo de auditoria): el colapso del MOHHO en ZDT2 (HV~0.11, archivo=1)
NO lo causa la regla de aceptacion -- las TRES politicas colapsan -- sino que los
operadores HHO con clip a [0,1] arrastran toda la poblacion a la esquina x=0 en
frentes concavos. La aceptacion es necesaria-pero-no-suficiente. Este script lo
DEMUESTRA y deja el veredicto en JSON para decidir el arreglo.

Salida: app/data/results/mohho_acceptance_selection.json
Presupuesto via env: POP (def 100), GEN (def 500), SEEDS (def 3).
Prueba rapida: POP=40 GEN=80 SEEDS=1 python mohho_acceptance_selection.py
"""
import os, json, time
from pathlib import Path
import numpy as np
import _bootstrap; _bootstrap.bootstrap_engine()
import benchmarks_moo as B   # bootstraps the engine on import

POP   = int(os.environ.get("POP", 100))
GEN   = int(os.environ.get("GEN", 500))
SEEDS = int(os.environ.get("SEEDS", 3))
RESULTS = Path(os.environ.get("RESULTS_DIR", _bootstrap.results_dir()))
RESULTS.mkdir(parents=True, exist_ok=True)
SANE_THRESHOLD = 0.90   # HV/true >= 0.90 on ALL benchmarks => "sano"


def main():
    t0 = time.time()
    seeds = list(range(1, 1 + SEEDS))
    results = {"budget": {"pop": POP, "gen": GEN, "evals": POP * GEN, "seeds": seeds},
               "sane_threshold_hv_over_true": SANE_THRESHOLD,
               "per_benchmark": {}}
    for name, b in B.BENCHMARKS.items():
        tf = B.true_front_hv(name)
        entry = {"M": b["M"], "dim": b["dim"], "ref": list(b["ref"]),
                 "true_front_hv": round(tf, 6), "policies": {}, "nsga2_reference": {}}
        # NSGA-II reference (sanity, like expA)
        ns = [B.run_nsga2_generic(b["fn"], b["dim"], b["M"], b["ref"], s, POP, GEN)["hv"]
              for s in seeds]
        entry["nsga2_reference"] = {"hv_mean": float(np.mean(ns)),
                                    "hv_over_true": round(float(np.mean(ns)) / tf, 4)}
        for pol in B.ACCEPTANCE:
            hv = []; mv = []; arch = []
            for s in seeds:
                r = B.run_mohho_generic(b["fn"], b["dim"], b["M"], b["ref"], s,
                                        POP, GEN, acceptance=pol)
                hv.append(r["hv"]); mv.append(r["moved_fraction_mean"]); arch.append(r["archive"])
            hv_mean = float(np.mean(hv))
            entry["policies"][pol] = {
                "hv_mean": hv_mean, "hv_over_true": round(hv_mean / tf, 4),
                "moved_fraction_mean": round(float(np.mean(mv)), 4),
                "archive_mean": round(float(np.mean(arch)), 1),
                "sane": bool(hv_mean / tf >= SANE_THRESHOLD)}
        results["per_benchmark"][name] = entry

    # verdict: a policy is globally sane iff sane on ALL benchmarks
    verdict = {}
    for pol in B.ACCEPTANCE:
        per = {n: results["per_benchmark"][n]["policies"][pol]["sane"]
               for n in B.BENCHMARKS}
        verdict[pol] = {"sane_per_benchmark": per, "globally_sane": all(per.values())}
    any_sane = any(v["globally_sane"] for v in verdict.values())
    zdt2_any = any(results["per_benchmark"]["ZDT2"]["policies"][p]["sane"]
                   for p in B.ACCEPTANCE)
    results["verdict"] = verdict
    results["DECISION"] = (
        "ALGUNA politica es globalmente sana: adoptarla y re-correr el ladder."
        if any_sane else
        ("NINGUNA politica de aceptacion sanea ZDT2 (todas colapsan a ~0.11). "
         "El colapso es PATOLOGIA DE OPERADORES/ARCHIVO en frentes concavos "
         "(clip a [0,1] crea un atractor de esquina x=0), NO de la aceptacion. "
         "Arreglo requerido es mas profundo (p.ej. reflexion-en-bordes en vez de "
         "clip, archivo con epsilon-dominancia, o amortiguar el tiro hacia el "
         "lider), O bien SCOPING honesto: declarar el MOHHO real-coded como "
         "baseline que colapsa en frentes concavos y NO apoyar la tesis en que "
         "sea un swarm fuerte. La aceptacion-gated ademas CONGELA la poblacion "
         "(ver diag_mohho_freeze) -- corregir el mecanismo del texto igualmente."))
    results["zdt2_fixable_by_acceptance"] = bool(zdt2_any)
    results["elapsed_s"] = round(time.time() - t0, 1)
    (RESULTS / "mohho_acceptance_selection.json").write_text(json.dumps(results, indent=2))
    # console summary
    for n in B.BENCHMARKS:
        e = results["per_benchmark"][n]
        print(f"== {n} (true={e['true_front_hv']:.3f}) nsga2={e['nsga2_reference']['hv_over_true']*100:.0f}% ==")
        for pol in B.ACCEPTANCE:
            pp = e["policies"][pol]
            print(f"   {pol:26s} HV/true={pp['hv_over_true']*100:5.1f}%  moved/iter={pp['moved_fraction_mean']:.3f}  arch={pp['archive_mean']:.0f}  sane={pp['sane']}")
    print("DECISION:", results["DECISION"][:90], "...")


if __name__ == "__main__":
    main()
