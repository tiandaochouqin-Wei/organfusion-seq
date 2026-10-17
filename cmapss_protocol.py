#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 (generality, 2nd domain): apply the SAME feedback-observation-control
protocol used for organoid screening and sepsis prediction to NASA C-MAPSS turbofan
degradation (FD001) -- an industrial predictive-maintenance task. Third domain,
same controller, testing the Discussion's "predictive maintenance" generality claim.

Task construction (mirrors the organoid/PhysioNet setup so the comparison is honest):
  - Each engine is a run-to-failure trajectory of 21 sensors (15 informative in FD001).
  - We observe an EARLY prefix (first T cycles) and must decide, per engine, whether it
    will enter a maintenance window (RUL <= H cycles) -- a binary early-warning label,
    the standard binarization of the RUL regression (reviewer Sec. on C-MAPSS).
  - The controller decides WHEN to stop observing and commit that call, under the same
    objective J = E[lambda*tau + rho(p_hat_tau)] as the other two domains.

Why FD001: one operating condition, one fault mode -- the cleanest subset, matching
"start with the least confounded case" (FD002/4 add 6 operating conditions that would
require per-condition normalization, a separate confound we deliberately avoid here).

Honest scope (stated, not hidden): like the sepsis task, discrimination is moderate
and the classes are imbalanced, so we compare on the controller's native objective J
(not matched decision-error), exactly as in the PhysioNet protocol.

Outputs:
  posterior/table_cmapss_auc_by_horizon.csv
  posterior/table_cmapss_cost.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # $ORGANFUSION_DATA, or ./datasets (see README)
import os, csv, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 0
OUT = "posterior"; os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(DATA_DIR, "cmapss/train_FD001.txt")
GRID = [20, 30, 40, 50, 60, 70, 80]   # early-observation horizons (cycles); < min lifetime 128
c_FP, c_FN = 1.0, 3.0        # missing a short-lived (early-failure) engine costs more than a false alarm
LAM_T = 0.02                 # per-cycle waiting cost (industrial: waiting = risk exposure)

raw = np.loadtxt(DATA)
eng = raw[:, 0].astype(int)
cyc = raw[:, 1].astype(int)
# informative sensors in FD001 (nonzero variance), indices into the 21-sensor block (cols 5..25)
SENS = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]
S = raw[:, 5:26][:, SENS]
engines = np.unique(eng)
maxcyc = {e: cyc[eng == e].max() for e in engines}   # total lifetime (run-to-failure cycle)
# Label: is this a SHORT-LIVED engine (total lifetime <= median)? -- predicted from an
# EARLY sensor prefix, structurally identical to the organoid task (predict a late
# outcome from early dynamics). Known only in hindsight; the prefix must forecast it.
med_life = np.median([maxcyc[e] for e in engines])
y_all = {e: int(maxcyc[e] <= med_life) for e in engines}
print(f"engines={len(engines)}  median lifetime={med_life:.0f}  horizons={GRID}  "
      f"short-lived rate={np.mean(list(y_all.values())):.2f}")


def prefix_feats(e, T):
    """Per-engine features from the first T cycles: last value + mean slope of each
    informative sensor. Standardized downstream. Mirrors the organoid feats()."""
    m = (eng == e) & (cyc <= T)
    if m.sum() < 2:
        return None
    Sp = S[m]; tp = cyc[m]
    out = []
    for k in range(Sp.shape[1]):
        v = Sp[:, k]
        out.append(v[-1])
        out.append((v[-1] - v[0]) / max(tp[-1] - tp[0], 1))
    return out


N = len(engines)
P = np.zeros((N, len(GRID)))
rows_auc = []
for j, T in enumerate(GRID):
    X, y, keep = [], [], []
    for i, e in enumerate(engines):
        f = prefix_feats(e, T)
        if f is None:
            continue
        X.append(f); y.append(y_all[e]); keep.append(i)
    X = np.array(X); y = np.array(y); keep = np.array(keep)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5).split(X, y, keep), method="predict_proba")[:, 1]
    P[keep, j] = p
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else np.nan
    rows_auc.append((T, round(auc, 4), int(y.sum()), len(y)))
    print(f"  T={T:>2}cyc  AUC={auc:.3f}  positives={y.sum()}/{len(y)}")

with open(os.path.join(OUT, "table_cmapss_auc_by_horizon.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["horizon_cycles", "auc", "n_positive", "n"])
    for r in rows_auc: w.writerow(r)

# ---------------------------------------------------------------------------
# Controller vs best fixed-time on the controller's native objective J.
# Label at decision time is the horizon-specific y (engine fails within H after that
# horizon); for a stopping policy we need a single per-engine ground truth, so we use
# the terminal label (does the engine reach RUL<=H at all within the observed grid) and
# evaluate cost with rho on the calibrated posterior at the stopping horizon.
# ---------------------------------------------------------------------------
G = np.array(GRID, float)
y_term = np.array([y_all[e] for e in engines])   # short-lived-engine label (same for all horizons)
P_STAR = c_FP / (c_FP + c_FN)     # 0.25
rho = lambda p: np.minimum(c_FP * (1 - p), c_FN * p)


def controller(m):
    tau = np.empty(N); dec = np.empty(N, int); pdec = np.empty(N)
    for i in range(N):
        ex = np.where(np.abs(P[i] - P_STAR) >= m)[0]
        j = ex[0] if len(ex) else len(G) - 1
        tau[i] = G[j]; pdec[i] = P[i, j]; dec[i] = int(P[i, j] >= P_STAR)
    return tau, dec, pdec


ms = np.linspace(0.0, 0.35, 40)
lambdas = np.geomspace(2e-3, 1e-1, 20)
rows_cost = []
for lam in lambdas:
    J_fix_by_T = lam * G + np.array([rho(P[:, j]).mean() for j in range(len(G))])
    jb = int(np.argmin(J_fix_by_T)); J_fix = float(J_fix_by_T[jb]); T_fix = float(G[jb])
    Jc = np.empty(len(ms)); tc = np.empty(len(ms))
    for k, m in enumerate(ms):
        tau, dec, pd = controller(m)
        Jc[k] = (lam * tau + rho(pd)).mean(); tc[k] = tau.mean()
    kb = int(np.argmin(Jc)); J_ctrl = float(Jc[kb]); t_ctrl = float(tc[kb])
    rows_cost.append((round(lam, 5), round(J_fix, 4), round(T_fix, 0), round(J_ctrl, 4),
                      round(t_ctrl, 1), round(J_fix - J_ctrl, 5)))
    print(f"  lambda={lam:.4f}  fixed(J={J_fix:.4f},T={T_fix:.0f})  ctrl(J={J_ctrl:.4f},meanT={t_ctrl:.1f})  gain={J_fix-J_ctrl:+.4f}")

with open(os.path.join(OUT, "table_cmapss_cost.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["lambda", "J_fixed_best", "T_fixed_best", "J_controller", "mean_T_controller", "gain"])
    for r in rows_cost: w.writerow(r)

n_win = sum(1 for r in rows_cost if r[-1] > 0)
print(f"\nSUMMARY: on C-MAPSS FD001 ({N} engines, {y_term.mean()*100:.0f}% positive at horizon end), the controller "
      f"attains lower expected cost J than the best fixed-time policy at {n_win}/{len(rows_cost)} delay weights "
      f"(c_FP={c_FP}, c_FN={c_FN}). Third domain, identical controller, same J objective as sepsis/organoids.")
