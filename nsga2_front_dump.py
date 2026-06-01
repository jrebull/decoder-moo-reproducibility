"""Dump the NSGA-II combined non-dominated front points (30 runs, seeds 1-30)
so the paper can overlay MOHHO vs NSGA-II fronts. Reuses the exact NSGA-II
implementation from compare_nsga2.py. Output: results/nsga2_front.json."""
import json
from pathlib import Path
import numpy as np

from app.core.problem import VisaProblem
from compare_nsga2 import run_nsga2, nondominated, SEEDS

RESULTS = Path("app/data/results")


def main():
    problem = VisaProblem()
    allp = []
    for s in SEEDS:
        front = run_nsga2(problem, s)
        allp += front
        print(f"NSGA-II seed {s}: {len(front)} sols (cum {len(allp)})")
    front = nondominated(allp)
    json.dump({"front": [list(p) for p in front], "size": len(front)},
              open(RESULTS / "nsga2_front.json", "w"))
    print("combined NSGA-II front:", len(front), "-> nsga2_front.json")


if __name__ == "__main__":
    main()
