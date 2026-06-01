"""Shared engine bootstrap: makes `import app.core...` work from any CWD and
resolves the canonical results dir (backend/app/data/results)."""
import os, sys

def _find_backend():
    here = os.path.abspath(os.path.dirname(__file__))
    for up in range(0, 8):
        base = os.path.abspath(os.path.join(here, *([".."] * up))) if up else here
        for cand in (os.path.join(base, "backend"), base):
            if os.path.isdir(os.path.join(cand, "app", "core")):
                return cand
    return None

def bootstrap_engine():
    try:
        import app.core.hho  # noqa
        return
    except Exception:
        pass
    b = _find_backend()
    if not b:
        raise ImportError("No pude localizar backend/app/core. Corre desde el repo.")
    sys.path.insert(0, b)

def results_dir():
    b = _find_backend()
    d = os.path.join(b, "app", "data", "results") if b else "app/data/results"
    os.makedirs(d, exist_ok=True)
    return d
