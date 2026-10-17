#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3B (2026-09-29, GO/NO-GO gate for a CCF-A / TKDE submission): head-to-head
comparison against three published early-time-series-classification (ETSC) stopping
rules -- TEASER (Schafer & Leser 2020, DMKD), EconomyGamma (the Economy-gamma line,
Achenchabe et al. 2021, MLJ), and CALIMERA (Bilski & Jastrzebska 2023, IPM) -- using the
`ml_edm` reference implementation (Renault et al. 2024, arXiv:2408.12925).

Design (isolates the STOPPING RULE, exactly matching this paper's own E4
methodology): all three external triggers are fed the SAME leave-condition-out
cross-fitted posterior process (posterior_data_grouped.npz's P_group) that every
other stopping rule in Table III is evaluated on -- they do not get to train their
own classifier, and neither do we get to use a posterior favourable to us; this is
the best-case test for each external trigger given our best available probability
estimates. Cost structure matches this paper's J = lambda*tau + rho(p_tau) exactly
(rho: symmetric, L=1) via ml_edm's CostMatrices(misclf_cost=2, delay_cost=2*lambda*t,
alpha=0.5), which reduces to J for every organoid at every horizon (verified: diagonal
[correct] = lambda*t, off-diagonal [wrong] = lambda*t + 1).

Each trigger's OWN learned parameters (EconomyGamma's group thresholds/transition
matrices, CALIMERA's kernel-ridge halters, TEASER's one-class-SVM master
classifiers) are fit under leave-condition-out GroupKFold(5) over the same 18
conditions as every other headline number in the paper -- fitting and evaluating on
the same organoids would inflate these baselines unfairly, biasing the comparison
IN FAVOUR of the externals, so we do not do that.

Outputs:
  posterior/table_etsc_baselines_lambda.csv     raw (lambda, method, mean_time, err, J)
  posterior/table_etsc_baselines_frontier.csv   min-err-reachable + time-at-target-error,
                                                 same format as table_seq_baselines_matched.csv
"""
import random, os, sys, csv, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_external", "ml_edm", "src"))
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from sklearn.model_selection import GroupKFold
from ml_edm.cost_matrices import CostMatrices
from ml_edm.trigger import EconomyGamma, CALIMERA, TEASER

OUT = "posterior"
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(int); groups = d["groups"]
N, T = P.shape
probas = np.stack([1 - P, P], axis=-1)  # (N, T, 2)
print(f"N={N} T={T} conditions={len(set(groups.tolist()))}  (leave-condition-out posterior, same as Table III)")

LAMBDAS = np.round(np.logspace(np.log10(0.0005), np.log10(0.05), 14), 6)
folds = list(GroupKFold(5).split(P, y, groups))


def run_method(cls, kwargs, cm, name):
    """Leave-condition-out CV: fit trigger on train fold's probas, predict on test
    fold sequentially over the grid (mirrors ml_edm's own EarlyClassifier.score()
    loop), aggregate out-of-fold predictions/times across all 5 folds -> full-N
    result, exactly matching how P_group itself was constructed."""
    all_preds = np.full(N, -1, dtype=int)
    all_t = np.full(N, -1.0)
    for tr, te in folds:
        trig = cls(timestamps=GRID, **kwargs)
        # ml_edm takes the full series length from X.shape[1] (TEASER's earliness term needs it);
        # until Stage 3P this dummy had one column per horizon (13), which made every TEASER
        # setting that stops after 13 h look worse than useless and pinned it at 12 h.
        random.seed(0)
        trig.fit(np.zeros((len(tr), int(GRID[-1]))), probas[tr], y[tr], cm)
        triggered = np.zeros(len(te), bool)
        preds = np.full(len(te), -1, dtype=int); ts = np.full(len(te), -1.0)
        for t_idx, length in enumerate(GRID):
            if trig.require_past_probas:
                p_arg = probas[te, :t_idx + 1, :]
            else:
                p_arg = probas[te, t_idx, :]
            X_now = np.zeros((len(te), int(length)))
            fire = np.asarray(trig.predict(X_now, p_arg, cm))
            newly = fire & (~triggered)
            preds[newly] = probas[te, t_idx, :][newly].argmax(-1)
            ts[newly] = length
            triggered |= fire
            if triggered.all():
                break
        if not triggered.all():
            rem = ~triggered
            preds[rem] = probas[te, -1, :][rem].argmax(-1)
            ts[rem] = GRID[-1]
        all_preds[te] = preds; all_t[te] = ts
    err = float((all_preds != y).mean()); mt = float(all_t.mean())
    return mt, err


class TEASERCached(TEASER):
    """TEASER with its per-horizon one-class SVMs fitted once per training fold and reused across
    delay weights and objectives (they depend on neither); the search for v is ml_edm's own."""
    _cache = {}

    def _fit_master_clf(self, probas, masks_pos_probas):
        key = (probas.tobytes(), masks_pos_probas.tobytes())
        if key not in TEASERCached._cache:
            TEASERCached._cache[key] = super()._fit_master_clf(probas, masks_pos_probas)
        return TEASERCached._cache[key]


METHODS = [
    ("EconomyGamma", EconomyGamma, dict(nb_intervals=5)),
    ("CALIMERA", CALIMERA, {}),
    ("TEASER", TEASERCached, dict(n_jobs=1)),                              # as published: hmean of accuracy and earliness
    ("TEASER-cost", TEASERCached, dict(objective="avg_cost", n_jobs=1)),   # ml_edm's cost objective: v minimises J
]

rows_lambda = []
t_start = time.time()
for lam in LAMBDAS:
    cm = CostMatrices(timestamps=GRID, n_classes=2, misclf_cost=2.0,
                       delay_cost=(lambda t, L=lam: 2 * L * t), alpha=0.5)
    for name, cls, kwargs in METHODS:
        t0 = time.time()
        mt, err = run_method(cls, kwargs, cm, name)
        J = lam * mt + err
        rows_lambda.append((lam, name, round(mt, 3), round(err, 4), round(J, 4)))
        print(f"  lambda={lam:.5f}  {name:14s} mean_t={mt:6.2f}h  err={err:.4f}  J={J:.4f}  "
              f"({time.time()-t0:.1f}s)  [elapsed {time.time()-t_start:.0f}s]")

with open(os.path.join(OUT, "table_etsc_baselines_lambda.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["lambda", "method", "mean_time_h", "decision_error", "J"])
    for r in rows_lambda: w.writerow(r)
print("\nwrote table_etsc_baselines_lambda.csv")


# ---- derive an error-vs-time frontier per method (same envelope logic as seq_baselines.py) ----
def isotonic_lower_envelope(t, e):
    o = np.argsort(t); t = np.asarray(t)[o]; e = np.asarray(e)[o]
    return t, np.minimum.accumulate(e)


def frontier_time_at_error(t, e, targets):
    t2, e2 = isotonic_lower_envelope(t, e)
    lo, hi = e2.min(), e2.max()
    out = []
    for te_ in targets:
        out.append(np.nan if (te_ < lo or te_ > hi) else float(np.interp(-te_, -e2, t2)))
    return out


targets = np.array([0.12, 0.16, 0.20, 0.24])
rows_frontier = []
for name, _, _ in METHODS:
    sub = [(r[2], r[3]) for r in rows_lambda if r[1] == name]
    t_arr = np.array([s[0] for s in sub]); e_arr = np.array([s[1] for s in sub])
    times_at = frontier_time_at_error(t_arr, e_arr, targets)
    t2, e2 = isotonic_lower_envelope(t_arr, e_arr)
    rows_frontier.append((name, round(float(e2.min()), 4)) + tuple(
        "" if np.isnan(x) else round(x, 2) for x in times_at))
    print(f"{name:14s} min_err={e2.min():.4f}  time@err[0.12,0.16,0.20,0.24]="
          f"{['n/a' if np.isnan(x) else f'{x:.1f}h' for x in times_at]}")

with open(os.path.join(OUT, "table_etsc_baselines_frontier.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["policy", "min_achievable_err"] + [f"time_at_err_{t_:.2f}" for t_ in targets])
    for r in rows_frontier: w.writerow(r)
print("\nwrote table_etsc_baselines_frontier.csv")
print(f"\ntotal runtime {time.time()-t_start:.0f}s")
