# Feasibility-Preserving MOHHO — Reproducibility Package

Code and precomputed results for a study of **decoder-based multi-objective
optimization** under hard constraints. The running case is the annual allocation
of roughly 140,000 employment-based immigrant visas across 21 countries and 5
categories (105 groups), cast as a three-objective integer program with six hard
constraints and solved with a multi-objective Harris Hawks Optimizer (MOHHO).

The central claim of the study is methodological: for a problem solved through a
feasibility-preserving decoder, performance is governed not by the *encoding* or
the *metaheuristic family* but by whether the search is **non-degenerate** — i.e.
its operators change the decoded order **and** its selection preserves population
diversity. A controlled seven-method ladder, a 2×2 operator/selection factorial,
and a replication on a knapsack, a TSP, and a flow-shop instantiate the claim.

This repository is provided for anonymous peer review. It contains everything
needed to **verify** every number reported in the paper and to **regenerate** the
underlying results from scratch.

## Repository layout

```
app/core/            MOHHO engine: problem, SPV+greedy decoder, HHO operators,
                     runner, and the FIFO baseline.
app/data/results/    Precomputed results (*.json, *.csv) — the exact values the
                     paper reports.
repro/               Verification and analysis entry points
                       verify_paper.py   reproducibility firewall (see below)
                       equity_audit.py   alternative fairness audit
                       _bootstrap.py      path/engine resolver (CWD-independent)
tests/               Unit tests for the decoder, problem, MOHHO, and baseline.
*.py                 Top-level scripts that regenerate the result JSON files
                     (ladder, permutation baselines, Taguchi DOE, omnibus stats,
                     second instance/problem, factorial, robustness sweeps, ...).
requirements.txt     Python dependencies.
```

The numerical core needs only `numpy` and `scipy`; `pytest` runs the tests.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

**1. Verify the paper's numbers (firewall).** Every figure quoted in the paper is
cross-checked against `app/data/results/*.json`; the script fails if any value is
stale or unsupported.

```bash
python repro/verify_paper.py --results app/data/results
# expected tail:  n_mismatch = 0 (... claims ...)
```

**2. Run the unit tests.**

```bash
python -m pytest tests
# expected:  all tests pass
```

**3. Regenerate results from scratch.** The top-level scripts and those in
`repro/` recompute the JSON files in `app/data/results/`. For example:

```bash
python perm_spea2.py        # SPEA2 permutation baseline
python repro/equity_audit.py
```

All randomized studies use fixed seeds — the diagnostic suite uses seed `1` and
the 30-run comparisons use the seed block `1..30` — so every reported number is
regenerated deterministically.

## Notes

- The case instance is a calibrated, partly synthetic single-fiscal-year model;
  it is a decision-support artifact, not a policy prescription.
- The spillover cascade is part of the formulation but is inert on the studied
  instances (every category is oversubscribed), so the effective per-category
  caps equal their base caps.
