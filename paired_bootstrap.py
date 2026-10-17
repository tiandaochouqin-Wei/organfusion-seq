#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3R: paired, condition-level bootstrap of the time differences behind Table 3.

Table 3 reports each rule's time to reach a target error; two separate intervals do not give the
interval of their difference. Here every rule is evaluated on the same organoids (the leave-condition-
out posterior of Table 3), the 18 conditions are resampled with replacement (B = 2000), each rule's
error-time frontier is recomputed on every replicate (lower envelope over its parameter, linear
interpolation at the target error, as in Table 3) and the DIFFERENCES are recorded:
    margin - CALIMERA,  margin - fixed horizon,  CALIMERA - fixed horizon,   at errors 0.12 ... 0.24.
CALIMERA is fitted out-of-fold by condition exactly as in etsc_baseline_comparison.py.
Output: posterior/table_paired_bootstrap.csv
"""
import os, sys, csv, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_external", "ml_edm", "src")); sys.path.insert(0, HERE)
import numpy as np
from sklearn.model_selection import GroupKFold
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from certify_any_trigger import fit_predict
from ml_edm.trigger import CALIMERA

OUT = "posterior"; B = 2000; TARGETS = (0.12, 0.16, 0.20, 0.24)
d = np.load(os.path.join(OUT, "posterior_data_grouped.npz"), allow_pickle=True)
P = d["P_group"]; y = d["y"].astype(int); GRID = d["GRID"].astype(float); groups = d["groups"]
N, T = P.shape; GI = GRID.astype(int)
conds = np.array(sorted(set(groups.tolist()))); cidx = [np.where(groups == c)[0] for c in conds]

# per-organoid (time, error) for every parameter value of every rule
fam = {}
mem = []
for m in np.linspace(0.0, 0.49, 50):
    ex = np.abs(P - 0.5) >= m; j = np.where(ex.any(1), ex.argmax(1), T - 1)
    mem.append((GRID[j], ((P[np.arange(N), j] >= 0.5).astype(int) != y).astype(float)))
fam["margin"] = mem
fam["fixed"] = [(np.full(N, GRID[j]), ((P[:, j] >= 0.5).astype(int) != y).astype(float)) for j in range(T)]
probas = np.stack([1 - P, P], axis=-1); folds = list(GroupKFold(5).split(P, y, groups))
mem = []
for lam in np.round(np.logspace(np.log10(0.0005), np.log10(0.05), 14), 6):        # Table 3's grid
    tau = np.zeros(N); call = np.zeros(N, int)
    for tr, te in folds:
        tau[te], call[te] = fit_predict(CALIMERA, {}, lam, GI, GI[-1], probas[tr], y[tr], probas[te])
    mem.append((tau, (call != y).astype(float)))
fam["CALIMERA"] = mem
TM = {k: np.array([t for t, _ in v]) for k, v in fam.items()}
ER = {k: np.array([e for _, e in v]) for k, v in fam.items()}


def time_at(w, k):
    """frontier of rule k under organoid weights w: time to reach each target error (nan if unreachable)."""
    t = (TM[k] * w).sum(1) / w.sum(); e = (ER[k] * w).sum(1) / w.sum()
    o = np.argsort(t); t, e = t[o], np.minimum.accumulate(e[o])
    out = []
    for tg in TARGETS:
        out.append(np.nan if (tg < e.min() or tg > e.max()) else float(np.interp(-tg, -e, t)))
    return np.array(out)


w0 = np.ones(N)
pt = {k: time_at(w0, k) for k in fam}
print("point estimates (h) at", TARGETS)
for k in fam: print(f"  {k:9s}", np.round(pt[k], 1))
rng = np.random.default_rng(0); boot = {k: [] for k in fam}
for b in range(B):
    w = np.zeros(N)
    for c in rng.integers(0, len(conds), len(conds)): w[cidx[c]] += 1
    for k in fam: boot[k].append(time_at(w, k))
boot = {k: np.array(v) for k, v in boot.items()}
rows = []
for a_, b_ in (("margin", "CALIMERA"), ("margin", "fixed"), ("CALIMERA", "fixed")):
    dlt = boot[a_] - boot[b_]
    for i, tg in enumerate(TARGETS):
        x = dlt[:, i]; x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        rows.append(dict(comparison=f"{a_} - {b_}", target_error=tg, diff_h=round(float(pt[a_][i] - pt[b_][i]), 2),
                         ci_lo=round(float(lo), 2), ci_hi=round(float(hi), 2), n_valid=len(x)))
        print(f"  {a_:>8} - {b_:<8} at {tg:.2f}: {pt[a_][i] - pt[b_][i]:+6.2f} h  [{lo:+.2f}, {hi:+.2f}]  (n={len(x)})")
with open(os.path.join(OUT, "table_paired_bootstrap.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote table_paired_bootstrap.csv")
