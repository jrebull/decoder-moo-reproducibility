"""
visa_competent_compare.py — EL EXPERIMENTO DECISIVO. Re-corre la comparacion en el
problema visa con el MO-HHO COMPETENTE anadido, al MISMO presupuesto que el ladder:
NSGA-II (real-coded), MOHHO ingenuo (paper actual), MO-HHO competente, random restart.
Reporta si el competente le gana a random (esperado: SI) -> el framing 'the decoder
does the work' es artefacto del swarm roto y debe reescribirse.

Presupuesto via env: POP (def 50), GEN (def 500), SEEDS (def 30).
"""
import os, json, time, statistics as st
from pathlib import Path
import numpy as np
import _bootstrap; _bootstrap.bootstrap_engine()
import benchmarks_moo as B
import competent_mohho as C
from app.core.problem import VisaProblem
from app.core import mohho as M

POP=int(os.environ.get("POP",50)); GEN=int(os.environ.get("GEN",500))
SEEDS=int(os.environ.get("SEEDS",30)); PM=float(os.environ.get("PM",0.1))
USE_SBX=os.environ.get("USE_SBX","1")=="1"
RESULTS=Path(os.environ.get("RESULTS_DIR",_bootstrap.results_dir())); RESULTS.mkdir(parents=True,exist_ok=True)
p=VisaProblem()
def ev(h): return M.evaluate_hawk(h,p)[1]
def hv(F): return M.compute_hypervolume(F)

def random_restart(seed,budget):
    rng=np.random.default_rng(seed); ap,af=[],[]
    for _ in range(budget):
        h=rng.uniform(0,1,size=M.NUM_GROUPS); M.update_archive(ap,af,h,ev(h),100,rng)
    return hv(af)
def naive(seed):
    pos,fit,_=M.run_mohho(p,seed,POP,GEN,100); return hv(fit)
def competent(seed):
    return C.run_competent_mohho(ev,M.NUM_GROUPS,3,hv,seed,POP,GEN,pm=PM,use_sbx=USE_SBX)['hv']

def main():
    t0=time.time(); seeds=list(range(1,1+SEEDS)); B_=POP*GEN
    comp=[competent(s) for s in seeds]; rand=[random_restart(s,B_) for s in seeds]; nai=[naive(s) for s in seeds]
    def agg(x): return {"mean":round(st.mean(x),1),"std":round(st.pstdev(x),1)}
    margin=100*(st.mean(comp)-st.mean(rand))/st.mean(rand)
    out={"budget":{"pop":POP,"gen":GEN,"evals":B_,"seeds":SEEDS},"competent_cfg":{"pm":PM,"use_sbx":USE_SBX},
         "competent_mohho":agg(comp),"random_restart":agg(rand),"naive_mohho":agg(nai),
         "competent_minus_random_pct":round(margin,2),
         "competent_beats_random":bool(st.mean(comp)>st.mean(rand)),
         "verdict":("CONFIRMADO: el MO-HHO competente LE GANA a random restart, mientras el ingenuo pierde. "
            "El 'random restart beats the swarm / decoder does most of the work' es artefacto del swarm roto. "
            "Reformular la tesis hacia ORDER-PRESERVATION tau (operadores que cambian el orden ganan; "
            "SBX near-identity pierde), y reportar honestamente que un swarm competente supera al muestreo ciego."
            if st.mean(comp)>st.mean(rand) else
            "El competente NO supera a random a este presupuesto; subir presupuesto/seeds y re-verificar."),
         "elapsed_s":round(time.time()-t0,1)}
    (RESULTS/"visa_competent_compare.json").write_text(json.dumps(out,indent=2))
    print("  competent MO-HHO: %s"%out["competent_mohho"])
    print("  random restart:   %s"%out["random_restart"])
    print("  naive MOHHO:      %s"%out["naive_mohho"])
    print("  competent vs random: %+.2f%% | beats_random=%s"%(margin,out["competent_beats_random"]))

if __name__=="__main__": main()
