"""
ladder_v5.py (FASE 1) — re-corre el ladder del visa con los 7 metodos, incluido el
MO-HHO COMPETENTE, a seed=1..30 estandar y 25,000 evals, con HV per-seed y tests
pareados. Escribe app/data/results/ladder_v5.json.

Metodos:
  nsga2_realcoded   (SBX, tau~0.99)        -- random-key, near-identity -> pierde
  naive_mohho       (gated, poblacion congelada) -- random-key -> pierde vs random
  competent_mohho   (HHO+NDS+poly-mut+SBX) -- random-key, low-tau + diversidad -> GANA
  random_restart    (muestreo ciego)
  perm_nsga2 / perm_moead / discrete_mohho (permutation-native, low-tau) -> ganan

Presupuesto via env: POP(50) GEN(500) SEEDS(30) PM(0.15) USE_SBX(1).
"""
import os, json, time, statistics as stx
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import _bootstrap; _bootstrap.bootstrap_engine()

from app.core.problem import VisaProblem
from app.core import mohho as M
import competent_mohho as C
# reuse the published runners
import sys; sys.path.insert(0, str(Path(_bootstrap._find_backend())))
from compare_nsga2 import run_nsga2, nondominated
from controls import run_random
from perm_nsga import run_perm_nsga
from perm_moead import run_perm_moead
from discrete_mohho import run_discrete_mohho

POP=int(os.environ.get("POP",50)); GEN=int(os.environ.get("GEN",500))
SEEDS=int(os.environ.get("SEEDS",30)); PM=float(os.environ.get("PM",0.15))
USE_SBX=os.environ.get("USE_SBX","1")=="1"
RESULTS=Path(_bootstrap.results_dir())
p=VisaProblem()
def ev(h): return M.evaluate_hawk(h,p)[1]
def HV(F): return M.compute_hypervolume([tuple(x) for x in F])

def m_nsga(s):    return run_nsga2(p,s)
def m_naive(s):   return M.run_mohho(p,s,POP,GEN,100)[1]
def m_comp(s):    return C.run_competent_mohho(ev,M.NUM_GROUPS,3,HV,s,POP,GEN,pm=PM,use_sbx=USE_SBX)['front']
def m_random(s):  return run_random(p,s,POP*GEN)
def m_pnsga(s):   return run_perm_nsga(p,s)
def m_pmoead(s):  return run_perm_moead(p,s)
def m_disc(s):    return run_discrete_mohho(p,s)

METHODS=[("nsga2_realcoded",m_nsga,"random_key"),("naive_mohho",m_naive,"random_key"),
         ("competent_mohho",m_comp,"random_key"),("random_restart",m_random,"random_key"),
         ("perm_nsga2",m_pnsga,"permutation"),("perm_moead",m_pmoead,"permutation"),
         ("discrete_mohho",m_disc,"permutation")]

def main():
    t0=time.time(); seeds=list(range(1,1+SEEDS))
    per_seed={}; combined={}; allfronts={}
    for name,fn,tier in METHODS:
        hvs=[]; comb=[]; ts=time.time()
        for s in seeds:
            F=fn(s); hvs.append(HV(F)); comb+=[tuple(map(float,x)) for x in F]
        cf=nondominated(comb)
        per_seed[name]={"tier":tier,"hv_per_seed":hvs,
            "hv_mean":round(stx.mean(hvs),1),"hv_std":round(stx.pstdev(hvs),1),
            "cv_pct":round(100*stx.pstdev(hvs)/stx.mean(hvs),3),
            "combined_front_hv":round(HV(cf),1),"combined_front_size":len(cf)}
        print(f"  {name:18s} HV {per_seed[name]['hv_mean']:>11,.0f} CV {per_seed[name]['cv_pct']:.2f}% "
              f"comb {per_seed[name]['combined_front_hv']:>11,.0f} ({len(cf)} sols) [{time.time()-ts:.0f}s]")
    # paired tests vs random_restart (same seeds)
    rnd=per_seed["random_restart"]["hv_per_seed"]
    pairs={}
    for name in per_seed:
        if name=="random_restart": continue
        x=per_seed[name]["hv_per_seed"]
        try: _,pg=wilcoxon(x,rnd,alternative="greater"); _,pl=wilcoxon(x,rnd,alternative="less")
        except ValueError: pg=pl=None
        better=sum(a>b for a,b in zip(x,rnd))
        pairs[name]={"mean_minus_random_pct":round(100*(stx.mean(x)-stx.mean(rnd))/stx.mean(rnd),2),
                     "p_greater_than_random":pg,"p_less_than_random":pl,"better_than_random_count":f"{better}/{SEEDS}"}
    # tier means
    rk=[per_seed[m]["hv_mean"] for m,_,t in METHODS if t=="random_key" and m!="random_restart"]
    pm_=[per_seed[m]["hv_mean"] for m,_,t in METHODS if t=="permutation"]
    out={"budget":{"pop":POP,"gen":GEN,"evals":POP*GEN,"seeds":seeds},
         "competent_cfg":{"pm":PM,"use_sbx":USE_SBX},
         "methods":per_seed,"paired_vs_random":pairs,
         "key_finding":{
            "competent_beats_random_pct":pairs["competent_mohho"]["mean_minus_random_pct"],
            "competent_beats_random":per_seed["competent_mohho"]["hv_mean"]>per_seed["random_restart"]["hv_mean"],
            "naive_beats_random":per_seed["naive_mohho"]["hv_mean"]>per_seed["random_restart"]["hv_mean"],
            "competent_p_greater_random":pairs["competent_mohho"]["p_greater_than_random"]},
         "elapsed_s":round(time.time()-t0,1)}
    (RESULTS/"ladder_v5.json").write_text(json.dumps(out,indent=2))
    kf=out["key_finding"]
    print(f"\ncompetent vs random: {kf['competent_beats_random_pct']:+.2f}% "
          f"(beats={kf['competent_beats_random']}, p_greater={kf['competent_p_greater_random']}); "
          f"naive_beats_random={kf['naive_beats_random']}")
    print(f"-> ladder_v5.json ({out['elapsed_s']:.0f}s)")

if __name__=="__main__": main()
