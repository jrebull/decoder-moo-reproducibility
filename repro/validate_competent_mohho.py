"""
validate_competent_mohho.py — Confirma que el MO-HHO competente (HHO+SBX+mut+NDS)
es SANO en ZDT1/ZDT2/DTLZ2 (HV/true >= 0.9), barriendo unas pocas configs para
elegir la adoptada. Escribe results/competent_mohho_validation.json.

Presupuesto via env: POP (def 100), GEN (def 500), SEEDS (def 5).
"""
import os, json, time
from pathlib import Path
import numpy as np
import _bootstrap; _bootstrap.bootstrap_engine()
import benchmarks_moo as B
import competent_mohho as C

POP=int(os.environ.get("POP",100)); GEN=int(os.environ.get("GEN",500))
SEEDS=int(os.environ.get("SEEDS",5)); RESULTS=Path(os.environ.get("RESULTS_DIR",_bootstrap.results_dir()))
RESULTS.mkdir(parents=True, exist_ok=True); THR=0.9
CONFIGS=[dict(pm=1/30,use_sbx=False),dict(pm=0.1,use_sbx=False),
         dict(pm=0.1,use_sbx=True),dict(pm=0.15,use_sbx=True)]

def main():
    t0=time.time(); seeds=list(range(1,1+SEEDS)); out={"budget":{"pop":POP,"gen":GEN,"seeds":seeds},
        "threshold_hv_over_true":THR,"configs":[]}
    best=None
    for cfg in CONFIGS:
        row={"cfg":cfg,"per_benchmark":{}}
        worst=1.0
        for name,b in B.BENCHMARKS.items():
            tf=B.true_front_hv(name)
            hvs=[C.run_competent_mohho(b['fn'],b['dim'],b['M'],
                  lambda F,ref=b['ref']:B.hv_any(F,ref),s,POP,GEN,**cfg)['hv'] for s in seeds]
            r=float(np.mean(hvs))/tf; worst=min(worst,r)
            row["per_benchmark"][name]={"hv_over_true":round(r,4),"sane":bool(r>=THR)}
        row["min_over_benchmarks"]=round(worst,4); row["sane_all"]=bool(worst>=THR)
        out["configs"].append(row)
        if row["sane_all"] and (best is None or worst>best["min_over_benchmarks"]):
            best=row
    out["adopted_config"]=best["cfg"] if best else None
    out["any_sane"]=best is not None
    out["verdict"]=("MO-HHO competente SANO en ZDT1/ZDT2/DTLZ2 -> baseline justo; "
        "el resultado 'random restart vs swarm' ya no esta confundido por un swarm roto."
        if best else "Ninguna config llega a 0.9 en las tres; subir GEN/seeds o pm y reintentar.")
    out["elapsed_s"]=round(time.time()-t0,1)
    (RESULTS/"competent_mohho_validation.json").write_text(json.dumps(out,indent=2))
    for row in out["configs"]:
        print("  cfg",row["cfg"],"-> min %.0f%%"%(row["min_over_benchmarks"]*100),"SANE" if row["sane_all"] else "")
    print("ADOPTED:",out["adopted_config"],"|",out["verdict"][:80])

if __name__=="__main__": main()
