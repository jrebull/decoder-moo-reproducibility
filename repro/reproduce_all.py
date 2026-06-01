"""
reproduce_all.py — Orquestador unico. Regenera resultados y corre el firewall.

MODOS:
  python reproduce_all.py --fast   # solo diagnosticos v4 + verify_paper (rapido)
  python reproduce_all.py --full   # ademas re-corre el ladder/Taguchi/etc (PESADO)

Corre cada script con CWD = backend/ para que `import app.core` y las rutas de
salida resuelvan. Best-effort: reporta exito/fallo por script sin abortar todo.
"""
import os, sys, time, subprocess, json
from pathlib import Path
import _bootstrap

BACKEND = Path(_bootstrap._find_backend() or ".")
KIT = Path(__file__).resolve().parent

V4_DIAGNOSTICS = [
    ("diag_mohho_freeze.py", KIT, {}),
    ("mohho_acceptance_selection.py", KIT, {}),
    ("diag_zdt2_collapse.py", KIT, {}),
]
# Scripts historicos del repo (PESADOS). Ajusta la lista si difiere en tu backend.
HEAVY_BACKEND = [
    "taguchi_doe.py", "compare_nsga2.py", "controls.py", "control_canonical_hho.py",
    "discrete_mohho.py", "perm_nsga.py", "perm_moead.py", "factorial_ops.py",
    "operator_order.py", "tau_trajectory.py", "eta_sweep.py", "omnibus_stats.py",
    "omnibus_visa_paired.py", "second_problem.py", "second_instance.py",
    "more_structures.py", "headroom_sweep.py", "regime.py",
]


def run(script, cwd, env_extra):
    env = dict(os.environ); env.update(env_extra)
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + str(KIT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(Path(cwd) / script)],
                           cwd=str(cwd), env=env, capture_output=True, text=True, timeout=36000)
        ok = (r.returncode == 0)
        return {"script": script, "ok": ok, "secs": round(time.time() - t0, 1),
                "stderr_tail": r.stderr[-500:] if not ok else ""}
    except Exception as e:
        return {"script": script, "ok": False, "secs": round(time.time() - t0, 1), "err": str(e)}


def main():
    full = "--full" in sys.argv
    log = []
    print("== v4 diagnostics ==")
    for s, cwd, ev in V4_DIAGNOSTICS:
        res = run(s, cwd, ev); log.append(res)
        print(f"  {'OK ' if res['ok'] else 'FAIL'} {s} ({res['secs']}s)")
    if full:
        print("== heavy backend regeneration ==")
        for s in HEAVY_BACKEND:
            if (BACKEND / s).exists():
                res = run(s, BACKEND, {}); log.append(res)
                print(f"  {'OK ' if res['ok'] else 'FAIL'} {s} ({res['secs']}s)")
            else:
                print(f"  SKIP {s} (no existe)")
    print("== firewall ==")
    res = run("verify_paper.py", KIT, {}); log.append(res)
    print(f"  {'OK ' if res['ok'] else 'FAIL'} verify_paper.py ({res['secs']}s)")
    Path(_bootstrap.results_dir(), "_reproduce_all_log.json").write_text(json.dumps(log, indent=2))
    print("\n-> _reproduce_all_log.json | modo:", "FULL" if full else "FAST")


if __name__ == "__main__":
    main()
